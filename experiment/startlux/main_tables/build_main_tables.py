#!/usr/bin/env python3
"""Build strict 5,955 RouterXBench tables plus a separate paper reference.

Strict main tables
  * 9 historical methods from the 5,955 clean sources, level 2
    (MEAN_TABLE + JEV_MEAN_TABLE + FIXED_MEAN_TABLE)
  * 12 StartLux conditions from validated *.summary.json
  * BQ/TC normalized only by actual 5,955 clean Qcloud/full_cloud_cost

Paper reference
  * the three current Overleaf table fragments, copied exactly
  * explicitly labelled as a 5,954 / changed-answer / greedy+sampled protocol
    that is NOT strictly matched to the 5,955 tables

No inference or scoring is performed.  Missing StartLux conditions are written
as "pending", never as zero or as a borrowed value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

N_REQUESTS = 5955
EXPECTED_BASELINE_COUNT = 9
METRICS = (
    ("auroc", "AUROC ×100", "AUROC_pct", "higher"),
    ("bq", "BQ (%)", "BQ_pct", "higher"),
    ("tc", "TC (%)", "TC_pct", "lower"),
)
EXPECTED_DATASETS = (
    ("alpaca_5k_test", "Alpaca"),
    ("mmlu_test", "MMLU"),
    ("big_math_5k_test", "Big-Math"),
    ("magpie_5k_test", "Magpie"),
    ("mmlu_pro", "MMLU-Pro"),
    ("math", "MATH"),
)
DATASET_KEYS = tuple(key for key, _ in EXPECTED_DATASETS)
DATASET_LABELS = tuple(label for _, label in EXPECTED_DATASETS)
INSTANT_SIZES = ("0.8B", "2B", "4B", "9B", "27B")
AUTOTHINK_SIZES = ("9B", "27B")
SIZE_BY_FLOAT = {0.8: "0.8B", 2.0: "2B", 4.0: "4B", 9.0: "9B", 27.0: "27B"}
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CELL_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*\$\\pm\$\s*([0-9]+(?:\.[0-9]+)?)\s*$")
STARTLUX_METHOD_RE = re.compile(
    r"^StartLux-(?P<size>[0-9]+(?:\.[0-9]+)?)B-(?P<mode>Instant|AutoThink)"
    r"-(?P<signal>query|qa|query_answer)$"
)

STRICT_METHODS = (
    ("SelfAsk", "SelfAsk", "fixed", "3 fixed executions 42/43/44; sample SD across executions"),
    ("Entropy", "Entropy", "fixed", "3 fixed executions 42/43/44; sample SD across executions"),
    ("MaxLogits", "MaxLogits", "fixed", "3 fixed executions 42/43/44; sample SD across executions"),
    ("Conf. Margin", "ConfidenceMargin", "fixed", "3 fixed executions 42/43/44; sample SD across executions"),
    ("Emb. MLP", "EmbeddingMLP", "mean", "3 training seeds 42/43/44; sample SD across seeds"),
    ("HybridLLM", "HybridLLM", "mean", "3 training seeds 42/43/44; sample SD across seeds"),
    ("RouteLLM", "RouteLLM", "mean", "3 training seeds 42/43/44; sample SD across seeds"),
    ("Jev", "Jev", "jev", "3 actual API executions (original/43/44); sample SD across executions"),
    ("MeanProbe", "MeanProbe", "mean", "3 training seeds 42/43/44; sample SD across seeds"),
)

STRICT_CAPTIONS = {
    "auroc": (
        "Strict clean 5,955 comparison: AUROC ×100 (higher is better). Baseline: 9 previous "
        "methods from the 5,955 clean sources; SD is across the actual repeats named in the "
        "table notes (training seeds, API executions or fixed executions). StartLux: one "
        "exploratory execution per condition (no SD). pending = condition not complete at this "
        "snapshot, not zero. The current paper tables are a separate, protocol-mismatched "
        "reference; do not combine them with this table."
    ),
    "bq": (
        "Strict clean 5,955 comparison: BQ at the middle budget range, as a percentage of "
        "all-cloud quality (higher is better). BQ is normalized only by the actual 5,955 "
        "clean Qcloud. Baseline: 9 previous methods, SD across the actual repeats named in "
        "the table notes. StartLux: one exploratory execution per condition (no SD). pending "
        "= condition not complete at this snapshot, not zero. The current paper tables are a "
        "separate, protocol-mismatched reference; do not combine them with this table."
    ),
    "tc": (
        "Strict clean 5,955 comparison: TC at the middle quality-target range, as a percentage "
        "of all-cloud generation cost (lower is better). TC is normalized only by the actual "
        "5,955 clean full_cloud_cost. Baseline: 9 previous methods, SD across the actual "
        "repeats named in the table notes. StartLux: one exploratory execution per condition "
        "(no SD). pending = condition not complete at this snapshot, not zero. The current "
        "paper tables are a separate, protocol-mismatched reference; do not combine them with "
        "this table."
    ),
}
STRICT_CAPTION_TEX = {
    "auroc": (
        r"Strict clean 5,955 comparison: AUROC $\times100$ (higher is better). Baseline: 9 "
        r"previous methods from the 5,955 clean sources; SD is across the actual repeats named "
        r"in the table notes (training seeds, API executions or fixed executions). StartLux: one "
        r"exploratory execution per condition (no SD). \emph{pending} = condition not complete "
        r"at this snapshot, not zero. The current paper tables are a separate, "
        r"protocol-mismatched reference; do not combine them with this table."
    ),
    "bq": (
        r"Strict clean 5,955 comparison: BQ at the middle budget range, as a percentage of "
        r"all-cloud quality (higher is better). BQ is normalized only by the actual 5,955 clean "
        r"Qcloud. Baseline: 9 previous methods, SD across the actual repeats named in the table "
        r"notes. StartLux: one exploratory execution per condition (no SD). \emph{pending} = "
        r"condition not complete at this snapshot, not zero. The current paper tables are a "
        r"separate, protocol-mismatched reference; do not combine them with this table."
    ),
    "tc": (
        r"Strict clean 5,955 comparison: TC at the middle quality-target range, as a percentage "
        r"of all-cloud generation cost (lower is better). TC is normalized only by the actual "
        r"5,955 clean full\_cloud\_cost. Baseline: 9 previous methods, SD across the actual "
        r"repeats named in the table notes. StartLux: one exploratory execution per condition "
        r"(no SD). \emph{pending} = condition not complete at this snapshot, not zero. The "
        r"current paper tables are a separate, protocol-mismatched reference; do not combine "
        r"them with this table."
    ),
}
PAPER_CAPTIONS = {
    "auroc": (
        "Paper reference, unchanged: clean-test router ability (AUROC ×100). Entries report "
        "mean ± sample SD across three evaluation rounds. Protocol differs from the strict "
        "5,955 tables: 5,954 subset, historical greedy + sampled 42/43, changed local answers. "
        "Not strictly matched to StartLux 5,955; do not compare directly or combine ranks."
    ),
    "bq": (
        "Paper reference, unchanged: BQ at the middle budget range, as percentages of all-cloud "
        "quality; higher is better. Entries report mean ± sample SD across three evaluation "
        "rounds. Protocol differs from the strict 5,955 tables: 5,954 subset, historical greedy "
        "+ sampled 42/43, changed local answers. Not strictly matched to StartLux 5,955; do not "
        "compare directly or combine ranks."
    ),
    "tc": (
        "Paper reference, unchanged: TC at the middle quality-target range, as percentages of "
        "all-cloud generation cost; lower is better. Entries report mean ± sample SD across "
        "three evaluation rounds. Protocol differs from the strict 5,955 tables: 5,954 subset, "
        "historical greedy + sampled 42/43, changed local answers. Not strictly matched to "
        "StartLux 5,955; do not compare directly or combine ranks."
    ),
}
PAPER_CAPTION_TEX = {
    "auroc": (
        r"Paper reference, unchanged: clean-test router ability (AUROC $\times100$). Entries "
        r"report mean $\pm$ sample SD across three evaluation rounds. Protocol differs from the "
        r"strict 5,955 tables: 5,954 subset, historical greedy + sampled 42/43, changed local "
        r"answers. Not strictly matched to StartLux 5,955; do not compare directly or combine "
        r"ranks."
    ),
    "bq": (
        r"Paper reference, unchanged: BQ at the middle budget range, as percentages of all-cloud "
        r"quality; higher is better. Entries report mean $\pm$ sample SD across three evaluation "
        r"rounds. Protocol differs from the strict 5,955 tables: 5,954 subset, historical greedy "
        r"+ sampled 42/43, changed local answers. Not strictly matched to StartLux 5,955; do not "
        r"compare directly or combine ranks."
    ),
    "tc": (
        r"Paper reference, unchanged: TC at the middle quality-target range, as percentages of "
        r"all-cloud generation cost; lower is better. Entries report mean $\pm$ sample SD across "
        r"three evaluation rounds. Protocol differs from the strict 5,955 tables: 5,954 subset, "
        r"historical greedy + sampled 42/43, changed local answers. Not strictly matched to "
        r"StartLux 5,955; do not compare directly or combine ranks."
    ),
}
STRICT_TITLES = {
    "auroc": "AUROC ×100 (higher is better)",
    "bq": "BQ, % (higher is better)",
    "tc": "TC, % (lower is better)",
}
PAPER_TITLES = {
    "auroc": "Paper reference: AUROC ×100",
    "bq": "Paper reference: BQ, %",
    "tc": "Paper reference: TC, %",
}
PANEL_TEX = {
    "auroc": None,
    "bq": r"Budget-limited quality: BQ $\uparrow$",
    "tc": r"Quality-first expenditure: TC $\downarrow$",
}
STRICT_LABELS = {"auroc": "tab:strict_auroc", "bq": "tab:strict_bq", "tc": "tab:strict_tc"}
PAPER_LABELS = {"auroc": "tab:paperref_auroc", "bq": "tab:paperref_bq", "tc": "tab:paperref_tc"}
STRICT_REPEAT_SHORT = (
    "Repeat identity: SelfAsk/Entropy/MaxLogits/Conf. Margin = 3 fixed executions; "
    "Emb. MLP/HybridLLM/RouteLLM/MeanProbe = 3 training seeds; Jev = 3 actual API executions. "
    "A reported 0.00 is a measured sample SD for identical fixed executions, not a missing value."
)


class BuildError(RuntimeError):
    """Malformed or inconsistent input."""


class IncompleteError(BuildError):
    """--require-complete was requested before all conditions finished."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path, key: str | None = None, base: Path | None = None) -> dict:
    path = path.resolve()
    record = {
        "key": key or path.name,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if base is not None:
        try:
            record["relative_path"] = path.relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    return record


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot read valid JSON from {path}: {error}") from error


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def unwrap_method(method: str) -> str:
    match = re.fullmatch(r"\\(?:textbf|textit|emph)\{(.*)\}", method.strip())
    return match.group(1) if match else method.strip()


def extract_caption(text: str) -> str:
    match = re.search(r"\\caption\{(.*?)\}\s*\\label", text, re.DOTALL)
    return " ".join(match.group(1).split()) if match else ""


def split_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.endswith("\\\\"):
        return None
    body = stripped[:-2].strip()
    if not body or body.startswith("\\"):
        return None
    return [part.strip() for part in body.split("&")]


@dataclass
class BaselineRow:
    method: str
    cells: list[dict]
    separator_before: bool
    source_key: str
    repeat_identity: str = ""
    source_method: str = ""


@dataclass
class BaselineTable:
    key: str
    rows: list[BaselineRow]
    datasets: list[str]
    sources: list[dict]
    caption: str
    scope: str


def parse_cells(raw_cells: list[str], context: str) -> list[dict]:
    cells = []
    for raw_cell in raw_cells:
        match = CELL_RE.fullmatch(raw_cell)
        if not match:
            raise BuildError(f"{context}: cell {raw_cell!r} is not a mean$\\pm$SD pair")
        cells.append({"raw": raw_cell, "mean": match.group(1), "sd": match.group(2)})
    return cells


def check_rows(rows: list[BaselineRow], expected_count: int, context: str) -> None:
    if len(rows) != expected_count:
        raise BuildError(f"{context}: expected {expected_count} baseline rows, found {len(rows)}")
    methods = [row.method for row in rows]
    if len(set(methods)) != len(methods):
        raise BuildError(f"{context}: duplicate baseline method names: {methods}")


def parse_flat_baseline(path: Path, metric_key: str) -> BaselineTable:
    text = path.read_text(encoding="utf-8")
    rows: list[BaselineRow] = []
    datasets: list[str] | None = None
    in_rows = False
    pending_separator = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\bottomrule"):
            break
        if stripped.startswith("Method &"):
            parts = split_table_row(stripped)
            if parts is None or parts[0] != "Method":
                raise BuildError(f"{path}: malformed header row")
            datasets = parts[1:]
            if datasets != list(DATASET_LABELS):
                raise BuildError(
                    f"{path}: dataset order/labels {datasets!r} != {list(DATASET_LABELS)!r}"
                )
            in_rows = True
            continue
        if not in_rows:
            continue
        if stripped.startswith("\\midrule"):
            if rows:
                pending_separator = True
            continue
        parts = split_table_row(stripped)
        if parts is None:
            continue
        if datasets is None or len(parts) != 1 + len(datasets):
            raise BuildError(f"{path}: row has {len(parts) - 1} cells for {len(datasets or [])} datasets")
        method = parts[0]
        if not method:
            raise BuildError(f"{path}: empty method name")
        rows.append(
            BaselineRow(
                method=unwrap_method(method),
                source_method=method,
                cells=parse_cells(parts[1:], f"{path}: {method}"),
                separator_before=pending_separator,
                source_key=path.name,
                repeat_identity="3 evaluation rounds (paper reference; protocol differs)",
            )
        )
        pending_separator = False
    check_rows(rows, EXPECTED_BASELINE_COUNT, str(path))
    if datasets is None:
        raise BuildError(f"{path}: no Method header found")
    return BaselineTable(
        key=metric_key,
        rows=rows,
        datasets=list(datasets),
        sources=[source_record(path, key=path.name, base=path.parent)],
        caption=extract_caption(text),
        scope="paper_reference",
    )


def parse_auroc_baseline(path: Path) -> BaselineTable:
    text = path.read_text(encoding="utf-8")
    panels: dict[str, dict] = {}
    current: dict | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\bottomrule"):
            break
        panel_match = re.search(
            r"\\multicolumn\{4\}\{c\}\{(Same-source|Different-source)\}", stripped
        )
        if panel_match:
            name = panel_match.group(1)
            if name in panels:
                raise BuildError(f"{path}: duplicate panel {name}")
            current = {"name": name, "rows": [], "datasets": None, "pending_separator": False}
            panels[name] = current
            continue
        if current is None:
            continue
        if stripped.startswith("Method &"):
            parts = split_table_row(stripped)
            if parts is None or parts[0] != "Method":
                raise BuildError(f"{path}: malformed panel header row")
            current["datasets"] = parts[1:]
            continue
        if stripped.startswith("\\midrule"):
            if current["rows"]:
                current["pending_separator"] = True
            continue
        parts = split_table_row(stripped)
        if parts is None:
            continue
        if current["datasets"] is None:
            raise BuildError(f"{path}: row before panel header: {stripped!r}")
        if len(parts) != 1 + len(current["datasets"]):
            raise BuildError(f"{path}: malformed row {stripped!r}")
        method = parts[0]
        if not method:
            raise BuildError(f"{path}: empty method name")
        current["rows"].append(
            BaselineRow(
                method=unwrap_method(method),
                source_method=method,
                cells=parse_cells(parts[1:], f"{path}: {method}"),
                separator_before=current["pending_separator"],
                source_key=path.name,
                repeat_identity="3 evaluation rounds (paper reference; protocol differs)",
            )
        )
        current["pending_separator"] = False

    if set(panels) != {"Same-source", "Different-source"}:
        raise BuildError(f"{path}: expected both AUROC panels, found {sorted(panels)}")
    same = panels["Same-source"]
    different = panels["Different-source"]
    if same["datasets"] != list(DATASET_LABELS[:3]):
        raise BuildError(f"{path}: same-source labels {same['datasets']!r} are wrong")
    if different["datasets"] != list(DATASET_LABELS[3:]):
        raise BuildError(f"{path}: different-source labels {different['datasets']!r} are wrong")
    check_rows(same["rows"], EXPECTED_BASELINE_COUNT, f"{path}: Same-source")
    check_rows(different["rows"], EXPECTED_BASELINE_COUNT, f"{path}: Different-source")

    different_by_method = {row.method: row for row in different["rows"]}
    if set(different_by_method) != {row.method for row in same["rows"]}:
        raise BuildError(f"{path}: AUROC panels contain different method sets")
    rows = []
    for same_row in same["rows"]:
        different_row = different_by_method[same_row.method]
        if same_row.separator_before != different_row.separator_before:
            raise BuildError(f"{path}: midrule placement differs between panels for {same_row.method}")
        rows.append(
            BaselineRow(
                method=same_row.method,
                source_method=same_row.source_method,
                cells=same_row.cells + different_row.cells,
                separator_before=same_row.separator_before,
                source_key=path.name,
                repeat_identity="3 evaluation rounds (paper reference; protocol differs)",
            )
        )
    return BaselineTable(
        key="auroc",
        rows=rows,
        datasets=list(DATASET_LABELS),
        sources=[source_record(path, key=path.name, base=path.parent)],
        caption=extract_caption(text),
        scope="paper_reference",
    )


def load_paper_reference(args: argparse.Namespace) -> dict[str, BaselineTable]:
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    paths = {
        "auroc": baseline_dir / args.auroc_tex,
        "bq": baseline_dir / args.bq_tex,
        "tc": baseline_dir / args.tc_tex,
    }
    for metric_key, path in paths.items():
        if not path.is_file():
            raise BuildError(f"{metric_key}: paper reference table not found: {path}")
    tables = {
        "auroc": parse_auroc_baseline(paths["auroc"]),
        "bq": parse_flat_baseline(paths["bq"], "bq"),
        "tc": parse_flat_baseline(paths["tc"], "tc"),
    }
    method_sets = {key: {row.method for row in table.rows} for key, table in tables.items()}
    if len({frozenset(value) for value in method_sets.values()}) != 1:
        raise BuildError(f"paper reference method sets differ: {method_sets}")
    expected_order = [row.method for row in tables["auroc"].rows]
    for key, table in tables.items():
        if [row.method for row in table.rows] != expected_order:
            raise BuildError(f"{key}: paper reference row order differs from AUROC")
    for key, table in tables.items():
        for row in table.rows:
            if len(row.cells) != len(DATASET_KEYS):
                raise BuildError(f"{key}: {row.method} has {len(row.cells)} cells, expected 6")
    return tables


def load_freeze(args: argparse.Namespace) -> dict:
    path = Path(args.freeze_json).expanduser().resolve()
    if not path.is_file():
        raise BuildError(f"freeze file not found: {path}")
    freeze = read_json(path)
    if freeze.get("protocol") != "historical_absolute":
        raise BuildError(f"{path}: protocol is not historical_absolute")
    if freeze.get("budgets_and_targets_recomputed") is not False:
        raise BuildError(f"{path}: budgets_and_targets_recomputed is not false")
    if not isinstance(freeze.get("normalization_only"), dict):
        raise BuildError(f"{path}: normalization_only missing")
    source_sha = freeze.get("source_sha256")
    if not isinstance(source_sha, str) or not HEX64_RE.fullmatch(source_sha):
        raise BuildError(f"{path}: source_sha256 is not a SHA256")
    source_rel = freeze.get("source")
    verification = {"available": False, "verified": False, "path": None, "reason": None}
    if isinstance(source_rel, str) and source_rel:
        candidate = Path(args.freeze_source_base).expanduser() / source_rel
        verification["path"] = str(candidate.resolve())
        if candidate.is_file():
            actual = sha256_file(candidate)
            verification["available"] = True
            verification["verified"] = actual == source_sha.lower()
            verification["actual_sha256"] = actual
            if not verification["verified"]:
                raise BuildError(
                    f"{path}: frozen source hash mismatch for {candidate}: {actual} != {source_sha}"
                )
        else:
            verification["reason"] = "frozen source file not present locally"
    record = source_record(path, key=path.name)
    record.update(
        {
            "protocol": freeze.get("protocol"),
            "source": freeze.get("source"),
            "source_sha256": source_sha,
            "normalization_only": freeze.get("normalization_only"),
            "budgets_and_targets_recomputed": freeze.get("budgets_and_targets_recomputed"),
            "learned_methods": freeze.get("learned_methods"),
            "seeds": freeze.get("seeds"),
            "source_verification": verification,
        }
    )
    return record


def expected_conditions() -> list[dict]:
    specs = []
    order = 0
    for size in INSTANT_SIZES:
        for signal, suffix in (("query", "Q"), ("query_answer", "Q+A")):
            specs.append(
                {
                    "key": f"instant_{size.lower()}_{signal}",
                    "size_label": size,
                    "mode": "Instant",
                    "signal": signal,
                    "display": f"StartLux {size} Instant ({suffix})",
                    "order": order,
                }
            )
            order += 1
    for size in AUTOTHINK_SIZES:
        specs.append(
            {
                "key": f"autothink_{size.lower()}_query",
                "size_label": size,
                "mode": "AutoThink",
                "signal": "query",
                "display": f"StartLux {size} AutoThink (Q)",
                "order": order,
            }
        )
        order += 1
    if len(specs) != 12:
        raise AssertionError("expected 12 StartLux conditions")
    return specs


def check_sha(value, label: str, path: Path) -> str:
    if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
        raise BuildError(f"{path}: {label} is not a 64-character hex SHA256")
    return value.lower()


def validate_summary(path: Path, data: dict, spec: dict) -> dict:
    method = data.get("method")
    if not isinstance(method, str) or not method.strip():
        raise BuildError(f"{path}: method is missing")
    if data.get("signal") != spec["signal"]:
        raise BuildError(f"{path}: signal {data.get('signal')!r} != expected {spec['signal']!r}")
    n_requests = data.get("n_requests")
    if type(n_requests) is not int or n_requests != N_REQUESTS:
        raise BuildError(f"{path}: n_requests {n_requests!r} != {N_REQUESTS}")
    acceptance = data.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("full_frozen_id_set") is not True:
        raise BuildError(f"{path}: acceptance.full_frozen_id_set is not true")
    if "frozen_test_n" in data and data["frozen_test_n"] != N_REQUESTS:
        raise BuildError(f"{path}: frozen_test_n {data['frozen_test_n']!r} != {N_REQUESTS}")
    router_model = data.get("router_model")
    if not isinstance(router_model, str) or not router_model.strip():
        raise BuildError(f"{path}: router_model is missing")
    if spec["size_label"].lower() not in router_model.lower():
        raise BuildError(
            f"{path}: router_model {router_model!r} does not identify size {spec['size_label']}"
        )
    scores_sha = check_sha(data.get("scores_sha256"), "scores_sha256", path)
    config_sha = data.get("config_sha256")
    if config_sha is not None:
        config_sha = check_sha(config_sha, "config_sha256", path)
    scoring_sha = data.get("scoring_algorithm_sha256")
    if scoring_sha is not None:
        scoring_sha = check_sha(scoring_sha, "scoring_algorithm_sha256", path)
    level2 = data.get("level2_main_table")
    if not isinstance(level2, dict):
        raise BuildError(f"{path}: level2_main_table is missing")
    if set(level2) != set(DATASET_KEYS):
        raise BuildError(
            f"{path}: level2_main_table keys {sorted(level2)} != {list(DATASET_KEYS)}"
        )
    metrics = {}
    for dataset in DATASET_KEYS:
        values = level2[dataset]
        if not isinstance(values, dict):
            raise BuildError(f"{path}: level2_main_table[{dataset}] is not an object")
        metrics[dataset] = {}
        for metric_key, _, json_key, _ in METRICS:
            value = values.get(json_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BuildError(f"{path}: {dataset}.{json_key} is not numeric")
            number = float(value)
            if not math.isfinite(number) or not (0.0 <= number <= 100.0):
                raise BuildError(f"{path}: {dataset}.{json_key} is out of range: {value!r}")
            metrics[dataset][metric_key] = number
    return {
        "method": method,
        "router_model": router_model,
        "signal": data["signal"],
        "n_requests": n_requests,
        "acceptance": {"full_frozen_id_set": True},
        "frozen_test_n": data.get("frozen_test_n"),
        "scores_sha256": scores_sha,
        "config_sha256": config_sha,
        "scoring_algorithm_sha256": scoring_sha,
        "script_version": data.get("script_version"),
        "metrics": metrics,
        "level2_payload_sha256": hashlib.sha256(
            json.dumps(level2, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def collect_summaries(args: argparse.Namespace) -> tuple[dict[str, dict], list[dict]]:
    root = Path(args.results_root).expanduser().resolve()
    if not root.is_dir():
        raise BuildError(f"results root not found: {root}")
    host_dirs = sorted(
        path for path in root.glob(args.host_glob) if path.is_dir() and path.resolve() != root
    )
    if host_dirs:
        files = sorted({path.resolve() for host in host_dirs for path in host.rglob("*.summary.json")})
    else:
        files = sorted({path.resolve() for path in root.rglob("*.summary.json")})
    output_dir = Path(args.output_dir).expanduser().resolve()
    files = [path for path in files if output_dir not in path.parents]
    if not files:
        print(f"note: no *.summary.json found under {root} (all conditions will be pending)")

    specs = {spec["key"]: spec for spec in expected_conditions()}
    grouped: dict[str, list[dict]] = defaultdict(list)
    ignored = []
    for path in files:
        data = read_json(path)
        method = data.get("method")
        if not isinstance(method, str) or not method.strip():
            raise BuildError(f"{path}: summary has no method")
        if not method.startswith("StartLux-"):
            ignored.append(
                {
                    **source_record(path, key=path.name, base=root),
                    "reason": "non-StartLux summary ignored",
                }
            )
            continue
        match = STARTLUX_METHOD_RE.fullmatch(method)
        if not match:
            raise BuildError(f"{path}: unrecognized StartLux method name {method!r}")
        size = round(float(match.group("size")), 6)
        size_label = SIZE_BY_FLOAT.get(size)
        if size_label is None:
            raise BuildError(f"{path}: unplanned StartLux size {match.group('size')}B")
        signal = "query" if match.group("signal") == "query" else "query_answer"
        mode = match.group("mode")
        key = f"{mode.lower()}_{size_label.lower()}_{signal}"
        spec = specs.get(key)
        if spec is None:
            raise BuildError(
                f"{path}: condition {mode}-{size_label}-{signal} is not one of the 12 planned conditions"
            )
        if signal != spec["signal"] or mode != spec["mode"] or size_label != spec["size_label"]:
            raise BuildError(f"{path}: condition identity does not match planned {spec['key']}")
        normalized = validate_summary(path, data, spec)
        source = source_record(path, key=path.name, base=root)
        grouped[key].append(
            {
                **normalized,
                "condition_key": key,
                "display": spec["display"],
                "mode": mode,
                "size_label": size_label,
                "source_key": source.get("relative_path", str(path)),
                "source_path": str(path),
                "source_sha256": source["sha256"],
            }
        )

    resolved: dict[str, dict] = {}
    for key, records in grouped.items():
        records = sorted(records, key=lambda item: item["source_path"])
        hashes = {record["scores_sha256"] for record in records}
        if len(hashes) > 1:
            paths = [record["source_path"] for record in records]
            raise BuildError(
                f"duplicate conflict for {key}: scores_sha256 differ across {paths}: {sorted(hashes)}"
            )
        payloads = {record["level2_payload_sha256"] for record in records}
        if len(payloads) > 1:
            paths = [record["source_path"] for record in records]
            raise BuildError(
                f"duplicate conflict for {key}: metric payloads differ across {paths}"
            )
        canonical = records[0]
        canonical["source_paths"] = [record["source_path"] for record in records]
        canonical["source_keys"] = [record["source_key"] for record in records]
        canonical["source_sha256s"] = [record["source_sha256"] for record in records]
        canonical["duplicate_group_size"] = len(records)
        resolved[key] = canonical
    return resolved, ignored


def finite_number(value, context: str) -> float:
    if isinstance(value, bool) or value is None:
        raise BuildError(f"{context}: missing numeric value")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise BuildError(f"{context}: not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise BuildError(f"{context}: not finite: {value!r}")
    return number


def optional_number(value, context: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return finite_number(value, context)


def format_cell(mean: float, sd: float | None) -> dict:
    if sd is None:
        return {
            "mean": f"{mean:.2f}",
            "sd": "",
            "raw": f"{mean:.2f}",
            "mean_value": mean,
            "sd_value": None,
        }
    return {
        "mean": f"{mean:.2f}",
        "sd": f"{sd:.2f}",
        "raw": f"{mean:.2f}$\\pm${sd:.2f}",
        "mean_value": mean,
        "sd_value": sd,
    }


def load_strict_baseline(args: argparse.Namespace) -> tuple[dict[str, BaselineTable], dict]:
    mean_path = Path(args.mean_table).expanduser().resolve()
    main_path = Path(args.strict_main_table).expanduser().resolve()
    jev_path = Path(args.jev_mean_table).expanduser().resolve()
    fixed_path = Path(args.fixed_mean_table).expanduser().resolve()
    den_path = Path(args.denominator_manifest).expanduser().resolve()
    for path in (mean_path, main_path, jev_path, fixed_path, den_path):
        if not path.is_file():
            raise BuildError(f"strict 5,955 source not found: {path}")
    sources = {
        "mean_table": source_record(mean_path, key=mean_path.name),
        "strict_main_table": source_record(main_path, key=main_path.name),
        "jev_mean_table": source_record(jev_path, key=jev_path.name),
        "fixed_mean_table": source_record(fixed_path, key=fixed_path.name),
        "denominator_manifest": source_record(den_path, key=den_path.name),
    }

    mean_rows = {
        (row["task"], row["method"], int(row["level"])): row
        for row in csv.DictReader(mean_path.open(newline="", encoding="utf-8"))
    }
    fixed_rows = {
        (row["task"], row["method"], int(row["level"])): row
        for row in csv.DictReader(fixed_path.open(newline="", encoding="utf-8"))
    }
    jev_rows = {
        (row["task"], int(row["level"])): row
        for row in csv.DictReader(jev_path.open(newline="", encoding="utf-8"))
    }
    main_rows = list(csv.DictReader(main_path.open(newline="", encoding="utf-8")))
    q_small = {}
    n_by_task = defaultdict(set)
    for row in main_rows:
        task = row["task"]
        q_small[task] = finite_number(row["Q_small"], f"{main_path}: {task}.Q_small")
        n_by_task[task].add(int(row["n"]))
    for task in DATASET_KEYS:
        if task not in n_by_task:
            raise BuildError(f"{main_path}: task {task} missing")
        if len(n_by_task[task]) != 1:
            raise BuildError(f"{main_path}: task {task} has inconsistent n: {n_by_task[task]}")
    if sum(next(iter(n_by_task[task])) for task in DATASET_KEYS) != N_REQUESTS:
        raise BuildError(
            f"{main_path}: per-task n do not sum to {N_REQUESTS}: "
            f"{sum(next(iter(n_by_task[task])) for task in DATASET_KEYS)}"
        )

    den_doc = read_json(den_path)
    den = den_doc.get("conditions")
    if not isinstance(den, dict) or set(den) != set(DATASET_KEYS):
        raise BuildError(f"{den_path}: conditions do not cover the six expected tasks")
    if sum(int(value["n"]) for value in den.values()) != N_REQUESTS:
        raise BuildError(f"{den_path}: per-task n do not sum to {N_REQUESTS}")
    for task in DATASET_KEYS:
        value = den[task]
        if int(value["n"]) != next(iter(n_by_task[task])):
            raise BuildError(f"{den_path}: {task} n differs from {main_path}")
        qcloud = finite_number(value["Q_cloud"], f"{den_path}: {task}.Q_cloud")
        full_cost = finite_number(value["full_cloud_cost"], f"{den_path}: {task}.full_cloud_cost")
        q_edge = finite_number(value["Q_edge"], f"{den_path}: {task}.Q_edge")
        if qcloud <= 0 or full_cost <= 0:
            raise BuildError(f"{den_path}: {task} has non-positive denominator")
        if abs(q_edge - q_small[task]) > 1e-12:
            raise BuildError(
                f"{den_path}: {task} Q_edge {q_edge} != strict MAIN_TABLE Q_small {q_small[task]}"
            )

    for display, source_method, source_kind, _ in STRICT_METHODS:
        if source_kind == "mean":
            for task in DATASET_KEYS:
                row = mean_rows.get((task, source_method, 2))
                if row is None:
                    raise BuildError(f"{mean_path}: missing {task}/{source_method}/level2")
                if row.get("seeds") != "42/43/44":
                    raise BuildError(f"{mean_path}: {task}/{source_method} seeds {row.get('seeds')!r}")
        elif source_kind == "fixed":
            for task in DATASET_KEYS:
                row = fixed_rows.get((task, source_method, 2))
                if row is None:
                    raise BuildError(f"{fixed_path}: missing {task}/{source_method}/level2")
                if row.get("actual_executions") != "3":
                    raise BuildError(
                        f"{fixed_path}: {task}/{source_method} actual_executions "
                        f"{row.get('actual_executions')!r}"
                    )
        elif source_kind == "jev":
            for task in DATASET_KEYS:
                row = jev_rows.get((task, 2))
                if row is None:
                    raise BuildError(f"{jev_path}: missing {task}/Jev/level2")
                if row.get("actual_api_executions") != "3":
                    raise BuildError(
                        f"{jev_path}: {task}/Jev actual_api_executions "
                        f"{row.get('actual_api_executions')!r}"
                    )

    fixed_cross = {"max_abs_auroc": 0.0, "max_abs_bq": 0.0, "max_abs_tc": 0.0, "rows": 0}
    for display, source_method, source_kind, _ in STRICT_METHODS:
        if source_kind != "fixed":
            continue
        for task in DATASET_KEYS:
            fixed = fixed_rows[(task, source_method, 2)]
            mean = mean_rows[(task, source_method, 2)]
            fixed_cross["rows"] += 1
            fixed_cross["max_abs_auroc"] = max(
                fixed_cross["max_abs_auroc"],
                abs(float(fixed["AUROC_mean"]) - float(mean["AUROC_mean"])),
            )
            fixed_cross["max_abs_bq"] = max(
                fixed_cross["max_abs_bq"],
                abs(
                    float(fixed["BQ_mean"])
                    - (q_small[task] + float(mean["delta_BQ_over_small_mean"]))
                ),
            )
            fixed_cross["max_abs_tc"] = max(
                fixed_cross["max_abs_tc"],
                abs(float(fixed["TC_mean"]) - float(mean["TC_mean"])),
            )

    jev_cross = {"max_abs_auroc": 0.0, "max_abs_bq": 0.0, "max_abs_tc": 0.0, "rows": 0}
    for task in DATASET_KEYS:
        jev = jev_rows[(task, 2)]
        mean = mean_rows[(task, "Jev", 2)]
        jev_cross["rows"] += 1
        jev_cross["max_abs_auroc"] = max(
            jev_cross["max_abs_auroc"],
            abs(float(jev["AUROC_mean"]) - float(mean["AUROC_mean"])),
        )
        jev_cross["max_abs_bq"] = max(
            jev_cross["max_abs_bq"],
            abs(
                float(jev["BQ_mean"])
                - (q_small[task] + float(mean["delta_BQ_over_small_mean"]))
            ),
        )
        jev_cross["max_abs_tc"] = max(
            jev_cross["max_abs_tc"],
            abs(float(jev["TC_mean"]) - float(mean["TC_mean"])),
        )
    jev_cross["note"] = (
        "MEAN_TABLE Jev is the original single execution; JEV_MEAN_TABLE averages three "
        "actual API executions. Differences are a repeat-set difference, not a conflict."
    )

    tables = {key: {"key": key, "rows": [], "sources": [], "caption": ""} for key in ("auroc", "bq", "tc")}
    method_records = []
    for display, source_method, source_kind, repeat in STRICT_METHODS:
        cells_by_metric = {key: [] for key in ("auroc", "bq", "tc")}
        source_key = {
            "mean": mean_path.name,
            "fixed": fixed_path.name,
            "jev": jev_path.name,
        }[source_kind]
        source_sha = {
            "mean": sources["mean_table"]["sha256"],
            "fixed": sources["fixed_mean_table"]["sha256"],
            "jev": sources["jev_mean_table"]["sha256"],
        }[source_kind]
        for task in DATASET_KEYS:
            qcloud = float(den[task]["Q_cloud"])
            full_cost = float(den[task]["full_cloud_cost"])
            if source_kind == "mean":
                row = mean_rows[(task, source_method, 2)]
                auroc_mean = finite_number(row["AUROC_mean"], f"{mean_path}: {task}/{source_method}.AUROC")
                auroc_sd = optional_number(row["AUROC_sd"], f"{mean_path}: {task}/{source_method}.AUROC_sd")
                bq_abs = q_small[task] + finite_number(
                    row["delta_BQ_over_small_mean"],
                    f"{mean_path}: {task}/{source_method}.delta_BQ",
                )
                bq_sd_abs = optional_number(
                    row["delta_BQ_over_small_sd"],
                    f"{mean_path}: {task}/{source_method}.delta_BQ_sd",
                )
                tc = finite_number(row["TC_mean"], f"{mean_path}: {task}/{source_method}.TC")
                tc_sd = optional_number(row["TC_sd"], f"{mean_path}: {task}/{source_method}.TC_sd")
            elif source_kind == "fixed":
                row = fixed_rows[(task, source_method, 2)]
                auroc_mean = finite_number(row["AUROC_mean"], f"{fixed_path}: {task}/{source_method}.AUROC")
                auroc_sd = optional_number(row["AUROC_sampleSD"], f"{fixed_path}: {task}/{source_method}.AUROC_sd")
                bq_abs = finite_number(row["BQ_mean"], f"{fixed_path}: {task}/{source_method}.BQ")
                bq_sd_abs = optional_number(row["BQ_sampleSD"], f"{fixed_path}: {task}/{source_method}.BQ_sd")
                tc = finite_number(row["TC_mean"], f"{fixed_path}: {task}/{source_method}.TC")
                tc_sd = optional_number(row["TC_sampleSD"], f"{fixed_path}: {task}/{source_method}.TC_sd")
            else:
                row = jev_rows[(task, 2)]
                auroc_mean = finite_number(row["AUROC_mean"], f"{jev_path}: {task}/Jev.AUROC")
                auroc_sd = optional_number(row["AUROC_sampleSD"], f"{jev_path}: {task}/Jev.AUROC_sd")
                bq_abs = finite_number(row["BQ_mean"], f"{jev_path}: {task}/Jev.BQ")
                bq_sd_abs = optional_number(row["BQ_sampleSD"], f"{jev_path}: {task}/Jev.BQ_sd")
                tc = finite_number(row["TC_mean"], f"{jev_path}: {task}/Jev.TC")
                tc_sd = optional_number(row["TC_sampleSD"], f"{jev_path}: {task}/Jev.TC_sd")
            values = {
                "auroc": (auroc_mean * 100.0, None if auroc_sd is None else auroc_sd * 100.0),
                "bq": (
                    bq_abs * 100.0 / qcloud,
                    None if bq_sd_abs is None else bq_sd_abs * 100.0 / qcloud,
                ),
                "tc": (
                    tc * 100.0 / full_cost,
                    None if tc_sd is None else tc_sd * 100.0 / full_cost,
                ),
            }
            for metric_key, (mean, sd) in values.items():
                if not math.isfinite(mean) or not (0.0 <= mean <= 100.0):
                    raise BuildError(
                        f"{display}/{task}/{metric_key}: normalized value out of range: {mean}"
                    )
                if sd is not None and (not math.isfinite(sd) or sd < 0):
                    raise BuildError(f"{display}/{task}/{metric_key}: invalid SD: {sd}")
                cells_by_metric[metric_key].append(format_cell(mean, sd))
        method_records.append(
            {
                "method": display,
                "source_method": source_method,
                "source_kind": source_kind,
                "source_key": source_key,
                "source_sha256": source_sha,
                "repeat_identity": repeat,
                "cells": {
                    metric_key: {task: cell for task, cell in zip(DATASET_KEYS, cells_by_metric[metric_key])}
                    for metric_key in ("auroc", "bq", "tc")
                },
            }
        )
        for metric_key in ("auroc", "bq", "tc"):
            tables[metric_key]["rows"].append(
                BaselineRow(
                    method=display,
                    source_method=source_method,
                    cells=cells_by_metric[metric_key],
                    separator_before=(display == "MeanProbe"),
                    source_key=source_key,
                    repeat_identity=repeat,
                )
            )
    for metric_key, table in tables.items():
        table["datasets"] = list(DATASET_LABELS)
        table["sources"] = [sources["mean_table"], sources["fixed_mean_table"], sources["jev_mean_table"]]
        table["caption"] = ""
        tables[metric_key] = BaselineTable(
            key=metric_key,
            rows=table["rows"],
            datasets=table["datasets"],
            sources=table["sources"],
            caption=table["caption"],
            scope="strict_5955",
        )
    if [row.method for row in tables["auroc"].rows] != [spec[0] for spec in STRICT_METHODS]:
        raise BuildError("strict baseline method order was not preserved")

    independent = {"available": False, "verified": False, "max_abs_qcloud": None, "max_abs_cost": None}
    joined_path = Path(args.joined_scores).expanduser().resolve()
    split_path = Path(args.shared_split).expanduser().resolve()
    if joined_path.is_file() and split_path.is_file():
        split = read_json(split_path)
        test_ids = set(split.get("test_ids", []))
        if len(test_ids) != N_REQUESTS:
            raise BuildError(f"{split_path}: test_ids count {len(test_ids)} != {N_REQUESTS}")
        aggregate = defaultdict(lambda: {"n": 0, "q": 0.0, "c": 0.0})
        for line in joined_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_id") not in test_ids:
                continue
            task = "mmlu_pro" if str(row.get("dataset", "")).startswith("mmlu_pro_") else row.get("dataset")
            if task not in DATASET_KEYS:
                continue
            aggregate[task]["n"] += 1
            aggregate[task]["q"] += float(row["cloud_quality"])
            aggregate[task]["c"] += float(row["cost"])
        max_q = 0.0
        max_c = 0.0
        for task in DATASET_KEYS:
            actual = aggregate[task]
            expected = den[task]
            if actual["n"] != int(expected["n"]):
                raise BuildError(f"{joined_path}: {task} n {actual['n']} != {expected['n']}")
            max_q = max(max_q, abs(actual["q"] / actual["n"] - float(expected["Q_cloud"])))
            max_c = max(max_c, abs(actual["c"] - float(expected["full_cloud_cost"])))
        independent.update(
            {
                "available": True,
                "verified": max_q <= 1e-9 and max_c <= 1e-9,
                "max_abs_qcloud": max_q,
                "max_abs_cost": max_c,
                "test_ids": len(test_ids),
                "joined_scores": source_record(joined_path, key=joined_path.name),
                "shared_split": source_record(split_path, key=split_path.name),
            }
        )
        if not independent["verified"]:
            raise BuildError(
                f"{den_path}: independent 5,955 denominator check failed "
                f"(max qcloud {max_q}, max cost {max_c})"
            )
    else:
        independent["reason"] = "joined scores or shared split not present locally"

    details = {
        "sources": sources,
        "denominators": {
            task: {
                "n": int(den[task]["n"]),
                "Qcloud": float(den[task]["Q_cloud"]),
                "full_cloud_cost": float(den[task]["full_cloud_cost"]),
                "Q_small": q_small[task],
            }
            for task in DATASET_KEYS
        },
        "independent_denominator_check": independent,
        "cross_checks": {"fixed_vs_mean_nonlearned": fixed_cross, "jev_vs_mean_jev": jev_cross},
        "methods": method_records,
    }
    return tables, details


def blank_cell() -> dict:
    return {"raw": "", "mean": "", "sd": ""}


def new_value_text(value: float) -> str:
    return f"{value:.2f}"


def family_rows(
    metric_key: str,
    baseline: BaselineTable,
    conditions: list[dict],
    complete: dict[str, dict],
    scope: str,
) -> list[dict]:
    rows = []
    for baseline_row in baseline.rows:
        rows.append(
            {
                "kind": "baseline",
                "method": baseline_row.method,
                "display_method": baseline_row.method,
                "separator_before": baseline_row.separator_before,
                "cells": baseline_row.cells,
                "status": "complete",
                "repeat_identity": baseline_row.repeat_identity,
            }
        )
    if scope == "paper_reference":
        return rows
    for spec in conditions:
        record = complete.get(spec["key"])
        if record is None:
            cells = [blank_cell() for _ in DATASET_KEYS]
            status = "pending"
        else:
            cells = []
            for dataset in DATASET_KEYS:
                text = new_value_text(record["metrics"][dataset][metric_key])
                cells.append({"raw": text, "mean": text, "sd": ""})
            status = "complete"
        rows.append(
            {
                "kind": "startlux",
                "method": spec["display"],
                "display_method": spec["display"],
                "separator_before": False,
                "cells": cells,
                "status": status,
                "repeat_identity": "1 exploratory execution",
            }
        )
    return rows


def render_markdown(
    scope: str,
    baselines: dict[str, BaselineTable],
    conditions: list[dict],
    complete: dict[str, dict],
    pending: list[dict],
    provenance: dict,
    args: argparse.Namespace,
) -> str:
    strict = scope == "strict_5955"
    lines: list[str] = []
    if strict:
        lines.append("# StartLux + RouterXBench strict 5,955 main tables")
        lines.append("")
        lines.append(f"Snapshot: `{provenance['build']['generated_at_utc']}`  ")
        lines.append(
            f"StartLux status: **{len(complete)}/{len(conditions)} conditions complete**; "
            f"{len(pending)} pending. Strict baseline: {EXPECTED_BASELINE_COUNT} previous "
            "methods on the clean 5,955 split."
        )
        lines.append("")
        lines.append(
            "> **Scope.** This file is the strict 5,955 comparison only. The current paper "
            "tables use a different protocol (5,954 subset, historical greedy + sampled 42/43, "
            "changed local answers); they are kept unchanged in `PAPER_REFERENCE.md` and must "
            "not be combined with these tables or ranked together. `pending` marks a condition "
            "that has not produced a validated summary yet; it is not zero."
        )
    else:
        lines.append("# Paper reference tables (unchanged, protocol-mismatched)")
        lines.append("")
        lines.append(f"Snapshot: `{provenance['build']['generated_at_utc']}`  ")
        lines.append(
            "> **Not the strict main table.** These are the current paper values, copied "
            "exactly from the Overleaf table fragments. Their protocol is a 5,954 subset with "
            "historical greedy + sampled 42/43 rounds and changed local answers, not the clean "
            "5,955 frozen-answer protocol used by StartLux. They are shown separately for "
            "reference only. Do not compare cells directly with the strict tables or combine "
            "the two into a rank."
        )
    lines.append("")

    for metric_key, metric_label, _, direction in METRICS:
        number = {"auroc": 1, "bq": 2, "tc": 3}[metric_key]
        title = STRICT_TITLES[metric_key] if strict else PAPER_TITLES[metric_key]
        lines.append(f"## Table {number}: {title}")
        lines.append("")
        if strict:
            lines.append(
                f"*{metric_label}; {direction} is better. Baseline: strict 5,955 sources, "
                "mean ± sample SD across the actual repeats named below. StartLux: one "
                "exploratory execution, no SD.*"
            )
        else:
            lines.append(
                f"*{metric_label}; {direction} is better. Values copied exactly from the "
                "current paper table (mean ± sample SD, three evaluation rounds).*"
            )
        lines.append("")
        lines.append("| Method | " + " | ".join(DATASET_LABELS) + " |")
        lines.append("|---|" + "---:|" * len(DATASET_LABELS))
        rows = family_rows(metric_key, baselines[metric_key], conditions, complete, scope)
        wrote_baseline_header = False
        wrote_startlux_header = False
        for row in rows:
            if row["kind"] == "baseline" and not wrote_baseline_header:
                label = (
                    "**Strict 5,955 baseline — mean ± sample SD; repeat identity in notes**"
                    if strict
                    else "**Paper reference — mean ± sample SD, 3 rounds (protocol differs)**"
                )
                lines.append(f"| {label} | " + " | ".join([""] * len(DATASET_LABELS)) + " |")
                wrote_baseline_header = True
            if strict and row["kind"] == "startlux" and not wrote_startlux_header:
                lines.append(
                    "| **StartLux — one exploratory run, no SD** | "
                    + " | ".join([""] * len(DATASET_LABELS))
                    + " |"
                )
                wrote_startlux_header = True
            values = []
            for cell in row["cells"]:
                if row["status"] == "pending":
                    values.append("pending")
                elif row["kind"] == "baseline":
                    values.append(f"{cell['mean']} ± {cell['sd']}" if cell["sd"] else cell["mean"])
                else:
                    values.append(cell["mean"])
            lines.append(f"| {row['display_method']} | " + " | ".join(values) + " |")
        lines.append("")
        caption = (STRICT_CAPTIONS if strict else PAPER_CAPTIONS)[metric_key]
        lines.append(f"*{caption}*")
        lines.append("")
        if strict:
            lines.append(f"*{STRICT_REPEAT_SHORT}*")
            lines.append("")
        lines.append(
            "Notes: `MeanProbe` remains in its original position after the midrule. "
            "Baseline signals are historical and are not restated (no per-method signal "
            "mapping is supplied by the frozen contracts); they are not guessed."
        )
        lines.append("")

    if strict:
        lines.append("## Completion and provenance")
        lines.append("")
        lines.append("| Item | Count |")
        lines.append("|---|---:|")
        lines.append(f"| Strict baseline methods | {EXPECTED_BASELINE_COUNT} |")
        lines.append(f"| Strict baseline cells (3 metrics × 6 datasets) | {provenance['build']['strict_baseline_cells']} |")
        lines.append(f"| StartLux conditions complete | {len(complete)} / {len(conditions)} |")
        lines.append(f"| StartLux cells with values | {provenance['build']['strict_value_cells']} |")
        lines.append(f"| StartLux cells pending | {provenance['build']['strict_pending_cells']} |")
        lines.append("")
        lines.append("### Pending StartLux conditions")
        lines.append("")
        if pending:
            lines.append("| Condition | Mode | Signal | Size | Reason |")
            lines.append("|---|---|---|---:|---|")
            for spec in pending:
                lines.append(
                    f"| {spec['display']} | {spec['mode']} | `{spec['signal']}` | "
                    f"{spec['size_label']} | {spec['reason']} |"
                )
        else:
            lines.append("None: all 12 planned conditions have validated summaries.")
        lines.append("")
        lines.append("### Strict sources and normalization")
        lines.append("")
        strict_info = provenance["strict_main"]
        for name in ("mean_table", "fixed_mean_table", "jev_mean_table", "strict_main_table", "denominator_manifest"):
            src = strict_info["sources"][name]
            lines.append(f"- `{name}`: `{src['path']}`  \n  SHA256 `{src['sha256']}`")
        lines.append(
            "- Actual 5,955 denominators: `main_summary/MANIFEST.json`; independent "
            f"recomputation verified: {strict_info['independent_denominator_check']['verified']} "
            f"(max |Qcloud diff| {strict_info['independent_denominator_check']['max_abs_qcloud']}, "
            f"max |cost diff| {strict_info['independent_denominator_check']['max_abs_cost']})."
        )
        lines.append(
            "- Fixed-vs-MEAN nonlearned cross-check: max |AUROC| "
            f"{strict_info['cross_checks']['fixed_vs_mean_nonlearned']['max_abs_auroc']:.3g}, "
            f"max |BQ| {strict_info['cross_checks']['fixed_vs_mean_nonlearned']['max_abs_bq']:.3g}, "
            f"max |TC| {strict_info['cross_checks']['fixed_vs_mean_nonlearned']['max_abs_tc']:.3g}."
        )
        lines.append(
            "- JEV-vs-MEAN Jev difference (expected repeat-set difference, JEV selected): "
            f"max |AUROC| {strict_info['cross_checks']['jev_vs_mean_jev']['max_abs_auroc']:.4g}, "
            f"max |BQ| {strict_info['cross_checks']['jev_vs_mean_jev']['max_abs_bq']:.4g}, "
            f"max |TC| {strict_info['cross_checks']['jev_vs_mean_jev']['max_abs_tc']:.4g}."
        )
        lines.append("")
        lines.append("### StartLux source files")
        lines.append("")
        for key in sorted(complete):
            record = complete[key]
            lines.append(
                f"- `{record['condition_key']}`: `{record['source_key']}`  \n"
                f"  source SHA256 `{record['source_sha256']}`; scores SHA256 "
                f"`{record['scores_sha256']}`; n_requests {record['n_requests']}; "
                "full_frozen_id_set true."
            )
        lines.append("")
        lines.append("### Regeneration")
        lines.append("")
        lines.append("```bash")
        lines.append(
            f"python3 {Path(__file__).name} --results-root {args.results_root} "
            f"--output-dir {args.output_dir}"
        )
        lines.append(
            f"python3 {Path(__file__).name} --results-root {args.results_root} "
            f"--output-dir {args.output_dir} --require-complete  # only after all 12 arrive"
        )
        lines.append("```")
        lines.append("")
        lines.append(
            "Full source hashes and condition keys are in `provenance.json`; the tidy "
            "strict rows are in `main_tables.csv`; the editable LaTeX is under `tables/`. "
            "The paper reference is separate in `PAPER_REFERENCE.md` / `paper_reference.csv`."
        )
    else:
        lines.append("## Protocol and provenance")
        lines.append("")
        lines.append(
            "Source tables: `/Users/zhuhe/overleaf/router/table1_auroc.tex`, "
            "`table2_bq.tex`, `table3_tc.tex` (read-only, unchanged)."
        )
        lines.append(
            "Paper data source: `/Users/zhuhe/overleaf/router/data/main_three_round/results.csv` "
            "and `verification.json`; 5,954 subset, historical greedy + sampled 42/43, changed "
            "local answers. These values are not the strict 5,955 values."
        )
        for key in ("auroc", "bq", "tc"):
            src = baselines[key].sources[0]
            lines.append(
                f"- `{src['key']}`: `{src['path']}`; SHA256 `{src['sha256']}`; "
                f"{len(baselines[key].rows)} methods × {len(baselines[key].datasets)} datasets."
            )
        lines.append("")
        lines.append(
            "Do not combine this file with `MAIN_TABLES.md`; the two protocols are not "
            "matched. The strict main table is the comparison table."
        )
    lines.append("")
    return "\n".join(lines)


def render_latex(
    scope: str,
    metric_key: str,
    baselines: dict[str, BaselineTable],
    conditions: list[dict],
    complete: dict[str, dict],
) -> str:
    strict = scope == "strict_5955"
    baseline = baselines[metric_key]
    rows = family_rows(metric_key, baseline, conditions, complete, scope)
    font = (
        r"\fontsize{8}{10}\selectfont\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.16}"
        if metric_key == "auroc"
        else r"\fontsize{8.5}{11}\selectfont\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}"
    )
    lines = [
        "% Generated by build_main_tables.py. Do not edit cells by hand.",
        f"% Scope: {scope}",
        f"% Baseline source(s): {', '.join(src['key'] for src in baseline.sources)}",
        f"% StartLux snapshot: {len(complete)}/{len(conditions)} conditions complete; pending is explicit.",
        rf"\begin{{table*}}[t]\centering{font}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
        r"\toprule",
    ]
    if metric_key == "auroc":
        lines.append(
            r"\multicolumn{1}{c}{} & \multicolumn{3}{c}{Same-source} & "
            r"\multicolumn{3}{c}{Different-source} \\"
        )
        lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    else:
        lines.append(rf"\multicolumn{{7}}{{c}}{{{PANEL_TEX[metric_key]}}} \\")
    lines.append("Method & " + " & ".join(baseline.datasets) + r" \\")
    lines.append(r"\midrule")
    baseline_label = (
        r"\multicolumn{7}{l}{\emph{Strict 5,955 baseline: mean $\pm$ sample SD across actual repeats}} \\"
        if strict
        else r"\multicolumn{7}{l}{\emph{Paper reference: mean $\pm$ sample SD, three evaluation rounds (protocol differs)}} \\"
    )
    lines.append(baseline_label)
    wrote_startlux_header = False
    for row in rows:
        if strict and row["kind"] == "startlux" and not wrote_startlux_header:
            lines.append(r"\addlinespace[2pt]")
            lines.append(
                r"\multicolumn{7}{l}{\emph{StartLux: one exploratory execution per condition, point estimate (no SD)}} \\"
            )
            wrote_startlux_header = True
        if row["kind"] == "baseline" and row["separator_before"]:
            lines.append(r"\midrule")
        cells = []
        for cell in row["cells"]:
            if row["status"] == "pending":
                cells.append(r"\textit{pending}")
            elif row["kind"] == "baseline":
                cells.append(cell["raw"])
            else:
                cells.append(cell["mean"])
        method = tex_escape(row["method"])
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    caption = (STRICT_CAPTION_TEX if strict else PAPER_CAPTION_TEX)[metric_key]
    lines.append(rf"\caption{{{caption}}}")
    label = (STRICT_LABELS if strict else PAPER_LABELS)[metric_key]
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def render_standalone(scope: str) -> str:
    strict = scope == "strict_5955"
    title = "strict 5,955 main tables" if strict else "paper reference tables (protocol-mismatched)"
    return "\n".join(
        [
            f"% Standalone wrapper for the generated {title}.",
            "% Compile from this directory: pdflatex main_tables_standalone.tex",
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.8in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{amsmath}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\input{table_auroc.tex}",
            r"\clearpage",
            r"\input{table_bq.tex}",
            r"\clearpage",
            r"\input{table_tc.tex}",
            r"\end{document}",
            "",
        ]
    )


def csv_rows(
    scope: str,
    baselines: dict[str, BaselineTable],
    conditions: list[dict],
    complete: dict[str, dict],
) -> list[dict]:
    strict = scope == "strict_5955"
    rows = []
    for metric_key, metric_label, _, direction in METRICS:
        baseline = baselines[metric_key]
        for baseline_row in baseline.rows:
            for dataset, label, cell in zip(DATASET_KEYS, DATASET_LABELS, baseline_row.cells):
                rows.append(
                    {
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "direction": direction,
                        "method": baseline_row.method,
                        "group": "strict_5955_baseline" if strict else "paper_reference",
                        "model_size": "",
                        "mode": "historical_strict_5955" if strict else "paper_3_rounds",
                        "signal": "historical_not_restated",
                        "n_requests": str(N_REQUESTS) if strict else "",
                        "dataset": dataset,
                        "dataset_label": label,
                        "value_pct": cell["mean"],
                        "sd_pct": cell["sd"],
                        "status": "complete",
                        "separator_before": "true" if baseline_row.separator_before else "false",
                        "source_key": baseline_row.source_key,
                        "source_sha256": next(
                            (src["sha256"] for src in baseline.sources if src["key"] == baseline_row.source_key),
                            "",
                        ),
                        "scores_sha256": "",
                        "repeat_identity": baseline_row.repeat_identity,
                    }
                )
        if not strict:
            continue
        for spec in conditions:
            record = complete.get(spec["key"])
            for dataset, label in zip(DATASET_KEYS, DATASET_LABELS):
                if record is None:
                    value = "pending"
                    status = "pending"
                    source_key = f"expected:{spec['key']}"
                    source_sha = ""
                    scores_sha = ""
                    n_requests = ""
                    repeat = "pending"
                else:
                    value = new_value_text(record["metrics"][dataset][metric_key])
                    status = "complete"
                    source_key = record["source_key"]
                    source_sha = record["source_sha256"]
                    scores_sha = record["scores_sha256"]
                    n_requests = str(record["n_requests"])
                    repeat = "1 exploratory execution"
                rows.append(
                    {
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "direction": direction,
                        "method": spec["display"],
                        "group": "startlux_single_run",
                        "model_size": spec["size_label"],
                        "mode": spec["mode"],
                        "signal": spec["signal"],
                        "n_requests": n_requests,
                        "dataset": dataset,
                        "dataset_label": label,
                        "value_pct": value,
                        "sd_pct": "",
                        "status": status,
                        "separator_before": "false",
                        "source_key": source_key,
                        "source_sha256": source_sha,
                        "scores_sha256": scores_sha,
                        "repeat_identity": repeat,
                    }
                )
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "metric_label",
                "direction",
                "method",
                "group",
                "model_size",
                "mode",
                "signal",
                "n_requests",
                "dataset",
                "dataset_label",
                "value_pct",
                "sd_pct",
                "status",
                "separator_before",
                "source_key",
                "source_sha256",
                "scores_sha256",
                "repeat_identity",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    if output_dir == baseline_dir or baseline_dir in output_dir.parents:
        raise BuildError(f"refusing to write into the read-only baseline/Overleaf directory: {output_dir}")
    paper = load_paper_reference(args)
    strict, strict_info = load_strict_baseline(args)
    freeze = load_freeze(args)
    specs = expected_conditions()
    complete, ignored = collect_summaries(args)
    missing = [spec for spec in specs if spec["key"] not in complete]
    if args.require_complete and missing:
        names = ", ".join(spec["display"] for spec in missing)
        raise IncompleteError(
            f"--require-complete set but {len(missing)}/{len(specs)} StartLux conditions are missing: {names}"
        )
    pending = [
        {
            **spec,
            "reason": "no validated *.summary.json found",
        }
        for spec in missing
    ]
    conditions = [dict(spec) for spec in specs]
    for spec in conditions:
        if spec["key"] in complete:
            spec["source_key"] = complete[spec["key"]]["source_key"]
            spec["status"] = "complete"
        else:
            spec["source_key"] = f"expected:{spec['key']}"
            spec["status"] = "pending"

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    strict_baseline_cells = sum(len(table.rows) * len(table.datasets) for table in strict.values())
    strict_value_cells = len(complete) * len(DATASET_KEYS) * len(METRICS)
    strict_pending_cells = len(pending) * len(DATASET_KEYS) * len(METRICS)
    paper_cells = sum(len(table.rows) * len(table.datasets) for table in paper.values())

    paper_data_sources = []
    for path in (Path(args.paper_results_csv), Path(args.paper_verification_json)):
        if path.is_file():
            paper_data_sources.append(source_record(path, key=path.name))
        else:
            paper_data_sources.append({"key": path.name, "path": str(path), "available": False})

    provenance = {
        "schema": "routerxbench-startlux-main-tables/2",
        "build": {
            "generated_at_utc": generated_at,
            "results_root": str(Path(args.results_root).expanduser().resolve()),
            "output_dir": str(output_dir),
            "baseline_dir": str(baseline_dir),
            "require_complete": bool(args.require_complete),
            "status": "complete" if not pending else "allow_pending",
            "expected_conditions": len(specs),
            "complete_conditions": len(complete),
            "pending_conditions": len(pending),
            "strict_baseline_methods": EXPECTED_BASELINE_COUNT,
            "strict_baseline_cells": strict_baseline_cells,
            "strict_value_cells": strict_value_cells,
            "strict_pending_cells": strict_pending_cells,
            "paper_reference_methods": EXPECTED_BASELINE_COUNT,
            "paper_reference_cells": paper_cells,
        },
        "strict_main": {
            "scope": "strict_5955",
            "protocol": (
                "Clean 5,955 shared split. AUROC: native task labels, mean ×100, sample SD "
                "across the actual repeats in the source. BQ/TC: level 2 historical absolute "
                "bands; BQ normalized by actual 5,955 Qcloud; TC by actual 5,955 "
                "full_cloud_cost. StartLux summaries use the same actual 5,955 endpoints."
            ),
            "baseline_method_mapping": [
                {
                    "method": display,
                    "source_method": source_method,
                    "source_kind": source_kind,
                    "repeat_identity": repeat,
                }
                for display, source_method, source_kind, repeat in STRICT_METHODS
            ],
            "excluded_source_rows": [
                {"method": "Random", "reason": "not one of the nine paper methods"},
            ],
            "sources": strict_info["sources"],
            "denominators": strict_info["denominators"],
            "independent_denominator_check": strict_info["independent_denominator_check"],
            "cross_checks": strict_info["cross_checks"],
            "methods": strict_info["methods"],
        },
        "paper_reference": {
            "scope": "paper_reference",
            "protocol": (
                "Current paper tables: 5,954 subset, historical greedy + sampled 42/43 with "
                "changed local answers. NOT identical to the clean 5,955 frozen-answer "
                "protocol. Shown only as an unchanged reference; no combined rank."
            ),
            "sources": [
                {
                    **table.sources[0],
                    "metric": key,
                    "caption": table.caption,
                    "methods": [row.method for row in table.rows],
                    "datasets": table.datasets,
                    "cells": len(table.rows) * len(table.datasets),
                }
                for key, table in paper.items()
            ],
            "data_sources": paper_data_sources,
        },
        "startlux": {
            "expected_conditions": len(specs),
            "complete_conditions": len(complete),
            "pending_conditions": len(pending),
            "conditions": [
                {
                    "condition_key": key,
                    "display": record["display"],
                    "method": record["method"],
                    "router_model": record["router_model"],
                    "mode": record["mode"],
                    "size_label": record["size_label"],
                    "signal": record["signal"],
                    "n_requests": record["n_requests"],
                    "acceptance": record["acceptance"],
                    "frozen_test_n": record["frozen_test_n"],
                    "source_key": record["source_key"],
                    "source_path": record["source_path"],
                    "source_sha256": record["source_sha256"],
                    "source_keys": record["source_keys"],
                    "source_paths": record["source_paths"],
                    "source_sha256s": record["source_sha256s"],
                    "duplicate_group_size": record["duplicate_group_size"],
                    "scores_sha256": record["scores_sha256"],
                    "config_sha256": record["config_sha256"],
                    "scoring_algorithm_sha256": record["scoring_algorithm_sha256"],
                    "script_version": record["script_version"],
                    "metrics": record["metrics"],
                }
                for key, record in sorted(
                    complete.items(),
                    key=lambda item: next(s["order"] for s in specs if s["key"] == item[0]),
                )
            ],
            "pending": [
                {
                    "condition_key": spec["key"],
                    "display": spec["display"],
                    "mode": spec["mode"],
                    "size_label": spec["size_label"],
                    "signal": spec["signal"],
                    "status": "pending",
                    "reason": "no validated *.summary.json found",
                }
                for spec in pending
            ],
            "ignored_summaries": ignored,
        },
        "protocol_context": {
            "freeze": freeze,
            "note": (
                "The freeze remains the historical-absolute protocol context. Strict "
                "display normalization uses the actual 5,955 main_summary denominators, "
                "not the 5,958-era per-task n in the run manifest."
            ),
        },
        "display": {
            "strict": {
                "scope": "strict_5955",
                "table_titles": [STRICT_TITLES[key] for key, _, _, _ in METRICS],
                "captions": [STRICT_CAPTIONS[key] for key, _, _, _ in METRICS],
                "baseline_group_label": "Strict 5,955 baseline — mean ± sample SD; repeat identity in notes",
                "startlux_group_label": "StartLux — one exploratory run, no SD",
            },
            "paper_reference": {
                "scope": "paper_reference",
                "table_titles": [PAPER_TITLES[key] for key, _, _, _ in METRICS],
                "captions": [PAPER_CAPTIONS[key] for key, _, _, _ in METRICS],
                "baseline_group_label": "Paper reference — mean ± sample SD, 3 rounds (protocol differs)",
            },
            "dataset_labels": list(DATASET_LABELS),
        },
    }

    outputs = {}
    strict_md = render_markdown("strict_5955", strict, conditions, complete, pending, provenance, args)
    paper_md = render_markdown("paper_reference", paper, conditions, complete, pending, provenance, args)
    strict_rows = csv_rows("strict_5955", strict, conditions, complete)
    paper_rows = csv_rows("paper_reference", paper, conditions, complete)
    write_text(output_dir / "MAIN_TABLES.md", strict_md)
    write_text(output_dir / "PAPER_REFERENCE.md", paper_md)
    write_csv(output_dir / "main_tables.csv", strict_rows)
    write_csv(output_dir / "paper_reference.csv", paper_rows)
    for scope, tables, directory in (
        ("strict_5955", strict, output_dir / "tables"),
        ("paper_reference", paper, output_dir / "tables" / "paper_reference"),
    ):
        write_text(directory / "main_tables_standalone.tex", render_standalone(scope))
        for key in ("auroc", "bq", "tc"):
            write_text(
                directory / f"table_{key}.tex",
                render_latex(scope, key, tables, conditions, complete),
            )
        outputs[scope] = {
            "markdown": {
                "path": "MAIN_TABLES.md" if scope == "strict_5955" else "PAPER_REFERENCE.md",
                "sha256": sha256_file(output_dir / ("MAIN_TABLES.md" if scope == "strict_5955" else "PAPER_REFERENCE.md")),
            },
            "csv": {
                "path": "main_tables.csv" if scope == "strict_5955" else "paper_reference.csv",
                "sha256": sha256_file(output_dir / ("main_tables.csv" if scope == "strict_5955" else "paper_reference.csv")),
                "rows": len(strict_rows if scope == "strict_5955" else paper_rows),
            },
            "latex": {
                key: {
                    "path": str((directory / f"table_{key}.tex").relative_to(output_dir)),
                    "sha256": sha256_file(directory / f"table_{key}.tex"),
                }
                for key in ("auroc", "bq", "tc")
            },
            "standalone": {
                "path": str((directory / "main_tables_standalone.tex").relative_to(output_dir)),
                "sha256": sha256_file(directory / "main_tables_standalone.tex"),
            },
        }
    provenance["outputs"] = outputs
    write_text(output_dir / "provenance.json", json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")

    print(
        f"Built strict 5,955: {len(complete)}/{len(specs)} StartLux conditions, "
        f"{len(pending)} pending, {strict_baseline_cells} baseline cells, "
        f"{strict_value_cells} new value cells, {strict_pending_cells} pending cells."
    )
    print(f"Paper reference: {EXPECTED_BASELINE_COUNT} methods, {paper_cells} cells (separate).")
    print(f"Output: {output_dir}")
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="root containing dsh-host*/**/*.summary.json (default: parent of this script)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="directory for generated tables and provenance",
    )
    parser.add_argument(
        "--baseline-dir",
        default="/Users/zhuhe/overleaf/router",
        help="read-only directory containing the three paper reference table fragments",
    )
    parser.add_argument("--auroc-tex", default="table1_auroc.tex")
    parser.add_argument("--bq-tex", default="table2_bq.tex")
    parser.add_argument("--tc-tex", default="table3_tc.tex")
    parser.add_argument(
        "--mean-table",
        default="/Users/zhuhe/wwx/output/routerxbench-clean-main-3seed-reviewed-20261013/MEAN_TABLE.csv",
    )
    parser.add_argument(
        "--strict-main-table",
        default="/Users/zhuhe/wwx/output/routerxbench-clean-main-3seed-reviewed-20261013/MAIN_TABLE.csv",
    )
    parser.add_argument(
        "--jev-mean-table",
        default="/Users/zhuhe/wwx/output/routerxbench-free-real-repeats-20261013/JEV_MEAN_TABLE.csv",
    )
    parser.add_argument(
        "--fixed-mean-table",
        default="/Users/zhuhe/wwx/output/router-fixed-repeat-tables-20261013/FIXED_MEAN_TABLE.csv",
    )
    parser.add_argument(
        "--denominator-manifest",
        default="/Users/zhuhe/wwx/output/routerxbench-clean-multiseed-20261013/main_summary/MANIFEST.json",
    )
    parser.add_argument(
        "--joined-scores",
        default="/Users/zhuhe/wwx/output/routerxbench-actual-cost-replay-20261012/JOINED_USAGE_QUALITY_SCORES.jsonl",
    )
    parser.add_argument(
        "--shared-split",
        default="/Users/zhuhe/wwx/output/routerxbench-clean-multiseed-20261013/SHARED_SPLIT.json",
    )
    parser.add_argument(
        "--paper-results-csv",
        default="/Users/zhuhe/overleaf/router/data/main_three_round/results.csv",
    )
    parser.add_argument(
        "--paper-verification-json",
        default="/Users/zhuhe/overleaf/router/data/main_three_round/verification.json",
    )
    parser.add_argument(
        "--freeze-json",
        default="/Users/zhuhe/wwx/output/routerxbench-clean-multiseed-20261013/LATEST_MAIN_TABLE_FREEZE.json",
    )
    parser.add_argument(
        "--freeze-source-base",
        default="/Users/zhuhe/wwx",
        help="base used to resolve the freeze's relative source path",
    )
    parser.add_argument(
        "--host-glob",
        default="dsh-host*",
        help="host directory glob under --results-root; non-matching roots are scanned recursively",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail instead of writing pending rows unless all 12 StartLux conditions are complete",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build(args)
    except IncompleteError as error:
        print(f"INCOMPLETE: {error}", file=sys.stderr)
        return 2
    except BuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
