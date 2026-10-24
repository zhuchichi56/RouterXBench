#!/usr/bin/env python3
"""Independent replay / baseline entry point for StartLux RouterXBench scores.

This wrapper never re-scores anything and never changes the frozen scoring
algorithm or denominators: it validates an existing ``*.jsonl`` score file and
delegates the metric computation to ``evaluate_scores`` from
``run_startlux_routerxbench.py`` (same body as the original script).

Typical uses
------------
Replay a historical score file that lacks ``router_model``/``signal``::

    python replay_startlux_metrics.py --scores old.jsonl \
        --method-name Jev --router-model jev-1.13.0 --signal query

Verify a replay against a reference metrics CSV (54 main-table cells)::

    python replay_startlux_metrics.py --scores old.jsonl --method-name Jev \
        --compare-metrics reference.metrics.csv --check-json check.json

Partial files are never turned into a full ``.summary.json``: ``--allow-partial``
only writes a coverage check file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_startlux_routerxbench import (  # noqa: E402
    DEFAULT_EVAL_BUNDLE,
    DEFAULT_RUNS_ROOT,
    evaluate_scores,
    load_inputs,
    sha256_file,
)


def read_scores(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"{path}:{i}: invalid JSON: {exc}") from exc
    return out


def load_pairs(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out


def validate_records(records, pairs, expected_config_sha256):
    ids = []
    seen = set()
    config_hashes = set()
    models = set()
    signals = set()
    for r in records:
        sid = r.get("sample_id")
        if sid is None:
            raise SystemExit("Score record without sample_id")
        if sid not in pairs:
            raise SystemExit(f"Score record for unknown sample_id {sid!r}")
        if sid in seen:
            raise SystemExit(f"Duplicate sample_id {sid} in score file")
        seen.add(sid)
        ids.append(sid)
        ins = r.get("instruction")
        if not isinstance(ins, str):
            raise SystemExit(f"{sid}: missing instruction")
        if hashlib.sha256(ins.encode("utf-8")).hexdigest() != pairs[sid]["query_sha256"]:
            raise SystemExit(f"{sid}: instruction sha256 does not match frozen pairs")
        score = r.get("router_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise SystemExit(f"{sid}: invalid router_score {score!r}")
        probs = r.get("probabilities")
        if probs is not None:
            if set(probs) != {"local", "cloud"}:
                raise SystemExit(f"{sid}: invalid probabilities {probs!r}")
            if abs(float(probs["local"]) - float(score)) > 1e-12:
                raise SystemExit(f"{sid}: probabilities[local] != router_score")
        if r.get("config_sha256"):
            config_hashes.add(r["config_sha256"])
        if r.get("router_model"):
            models.add(r["router_model"])
        if r.get("signal"):
            signals.add(r["signal"])
    if expected_config_sha256:
        if config_hashes != {expected_config_sha256}:
            raise SystemExit(f"config_sha256 mismatch: file has {sorted(config_hashes)}, expected {expected_config_sha256}")
    if len(config_hashes) > 1:
        raise SystemExit(f"Mixed config_sha256 in score file: {sorted(config_hashes)}")
    if len(models) > 1:
        raise SystemExit(f"Mixed router_model in score file: {sorted(models)}")
    if len(signals) > 1:
        raise SystemExit(f"Mixed signal in score file: {sorted(signals)}")
    return ids, config_hashes, models, signals


def compare_metrics(produced: Path, reference: Path, tol: float):
    def load(p):
        with p.open(newline="") as f:
            return {(r["task"], int(r["level"])): r for r in csv.DictReader(f)}

    a = load(produced)
    b = load(reference)
    if set(a) != set(b):
        missing = sorted(set(b) - set(a))
        extra = sorted(set(a) - set(b))
        raise SystemExit(f"metrics shape mismatch: missing={missing[:5]} extra={extra[:5]}")
    cells = []
    maxdiff = 0.0
    cols = ["AUROC_pct", "BQ_pct", "TC_pct"]
    for key in sorted(b):
        for col in cols:
            va = a[key].get(col)
            vb = b[key].get(col)
            va = None if va in ("", None) else float(va)
            vb = None if vb in ("", None) else float(vb)
            if va is None or vb is None:
                diff = 0.0 if va is vb else float("inf")
            else:
                diff = abs(va - vb)
            maxdiff = max(maxdiff, diff)
            cells.append({"task": key[0], "level": key[1], "column": col, "produced": va, "reference": vb, "abs_diff": diff})
    passed = maxdiff <= tol
    return {"passed": passed, "n_cells": len(cells), "max_abs_diff": maxdiff, "tolerance": tol, "cells": cells}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--eval-bundle", type=Path, default=DEFAULT_EVAL_BUNDLE)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--split-file", type=Path)
    p.add_argument("--input-file", type=Path)
    p.add_argument("--small-file", type=Path)
    p.add_argument("--labels-file", type=Path)
    p.add_argument("--pairs-file", type=Path)
    p.add_argument("--manifest-file", type=Path)
    p.add_argument("--require-frozen-inputs", action="store_true")
    p.add_argument("--method-name", required=True)
    p.add_argument("--router-model")
    p.add_argument("--signal", choices=["query", "query_answer"])
    p.add_argument("--expected-config-sha256")
    p.add_argument("--metrics-out", type=Path)
    p.add_argument("--summary-out", type=Path)
    p.add_argument("--check-json", type=Path)
    p.add_argument("--compare-metrics", type=Path)
    p.add_argument("--compare-tolerance", type=float, default=1e-9)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv)

    pairs_path = args.pairs_file or (args.eval_bundle / "JOINED_USAGE_QUALITY_SCORES.jsonl")
    pairs = load_pairs(pairs_path)
    records = read_scores(args.scores)
    if not records:
        raise SystemExit(f"No score records in {args.scores}")
    ids, config_hashes, models, signals = validate_records(records, pairs, args.expected_config_sha256)

    signal = args.signal or (next(iter(signals)) if signals else "query")
    ns = argparse.Namespace(
        eval_bundle=args.eval_bundle,
        runs_root=args.runs_root,
        split_file=args.split_file,
        input_file=args.input_file,
        small_file=args.small_file,
        labels_file=args.labels_file,
        pairs_file=args.pairs_file,
        manifest_file=args.manifest_file,
        require_frozen_inputs=args.require_frozen_inputs,
        signal=signal,
    )
    rows, small_rows, labels, manifest_pairs, manifest, measured = load_inputs(ns)

    expected_ids = set(manifest_pairs)
    got_ids = set(ids)
    extra = got_ids - expected_ids
    missing = expected_ids - got_ids
    if extra:
        raise SystemExit(f"Score file has {len(extra)} ids outside the frozen set, e.g. {sorted(extra)[:5]}")
    if missing and not args.allow_partial:
        raise SystemExit(f"Score file incomplete: {len(got_ids)}/{len(expected_ids)} ids; no summary written")

    model = args.router_model or (next(iter(models)) if models else None)
    if args.router_model and models and args.router_model not in models:
        raise SystemExit(f"--router-model {args.router_model!r} conflicts with file metadata {sorted(models)}")
    if args.signal and signals and args.signal not in signals:
        raise SystemExit(f"--signal {args.signal!r} conflicts with file metadata {sorted(signals)}")

    metadata = {"router_model": model, "signal": signal}
    if args.allow_partial and missing:
        per_dataset = {}
        for sid in got_ids:
            ds = pairs[sid]["dataset"]
            per_dataset[ds] = per_dataset.get(ds, 0) + 1
        report = {
            "scores": str(args.scores),
            "scores_sha256": sha256_file(args.scores),
            "n_records": len(records),
            "expected_n": len(expected_ids),
            "missing": len(missing),
            "extra": 0,
            "complete": False,
            "per_dataset_present": per_dataset,
            "metadata": metadata,
            "input_files": measured,
            "note": "partial coverage check only; no metrics/summary produced",
        }
        out = args.check_json or args.scores.with_suffix(".partial.check.json")
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return report

    metrics_out = args.metrics_out or args.scores.with_suffix(".metrics.csv")
    summary_out = args.summary_out or args.scores.with_suffix(".summary.json")
    payload = evaluate_scores(
        args.scores,
        args.eval_bundle,
        labels,
        manifest_pairs,
        manifest,
        args.method_name,
        metadata=metadata,
        metrics_out=metrics_out,
        summary_out=summary_out,
    )

    check = {
        "scores": str(args.scores),
        "scores_sha256": sha256_file(args.scores),
        "n_records": len(records),
        "expected_n": len(expected_ids),
        "complete": True,
        "metadata": metadata,
        "config_sha256": sorted(config_hashes),
        "metrics_out": str(metrics_out),
        "summary_out": str(summary_out),
        "summary_sha256": sha256_file(summary_out),
        "metrics_sha256": sha256_file(metrics_out),
        "input_files": measured,
        "scoring_algorithm_sha256": payload["scoring_algorithm_sha256"],
    }
    if args.compare_metrics:
        cmp = compare_metrics(metrics_out, args.compare_metrics, args.compare_tolerance)
        check["comparison"] = cmp
        print(json.dumps({"comparison_passed": cmp["passed"], "n_cells": cmp["n_cells"], "max_abs_diff": cmp["max_abs_diff"]}), flush=True)
    out = args.check_json or args.scores.with_suffix(".replay.check.json")
    out.write_text(json.dumps(check, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(check, indent=2, ensure_ascii=False), flush=True)
    if args.compare_metrics and not check["comparison"]["passed"]:
        raise SystemExit(f"BASELINE REPLAY MISMATCH: max_abs_diff={check['comparison']['max_abs_diff']}")
    return payload


if __name__ == "__main__":
    main()
