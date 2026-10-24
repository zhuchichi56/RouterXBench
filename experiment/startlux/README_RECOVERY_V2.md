# recovery_v2 — hardened RouterXBench / StartLux evaluation bundle

Published by **host1 (8h20-1)** after the original `eval_bundle/run_startlux_routerxbench.py`
(sha256 `33ccc76992ed24cf13c58ecc940bc9b31d790e13793dd2bee6e11f76d83c91e1`) was audited.

**Only run this bundle if `READY.json` exists here, has `"status": "READY"`, and the
sha256 of the file you execute matches the value listed in `READY.json["files"]`.**

## Files

| file | role |
|---|---|
| `run_startlux_routerxbench.py` | live scoring + validation + metrics (original metric math unchanged) |
| `replay_startlux_metrics.py` | independent replay/baseline entry point; never re-scores |
| `INPUT_SHA256.json` | frozen input file list, sha256, sizes, line counts |
| `SELFTEST_REPORT.json` | 8 CPU/fake-HTTP regression checks |
| `BASELINE_REPLAY_CHECK.json` | 54-cell replay of the frozen Jev baseline (`max_abs_diff = 0.0`) |
| `READY.json` | hashes, baseline reconciliation, check results, contract |
| `SHA256SUMS` | checksums of the files above |

The replay tool imports `evaluate_scores` from the run script, so **execute it from
this directory** (or put this directory on `PYTHONPATH`).

## Frozen protocol (do not change)

* 5,955 clean test ids from `clean-multiseed-20261013/SHARED_SPLIT.json`.
* `local=meta-llama/Llama-3.1-8B-Instruct`, absolute historical budget/target bands
  from `MANIFEST.json` (`historical_absolute` — see
  `routerxbench-clean-multiseed-20261013/LATEST_MAIN_TABLE_FREEZE.json`).
* `PROMPT_INSTRUCTIONS`, `CRITERIA`, `score_direction=higher_prefers_small`, and the
  `evaluate_scores` body are unchanged from the original script.
* Explicit per-request controls: `none_check=true`, `allow_none=false`; think controls
  are always part of the config hash. Instant → `--think-mode never`; AutoThink →
  `--think-mode auto --gate 0.9 --think-budget 1024`.

## Live scoring

```bash
python run_startlux_routerxbench.py \
  --server-url http://127.0.0.1:<port> \
  --expected-model <exact model name returned by the server> \
  --method-name <method> --signal query|query_answer \
  --concurrency <probed stable value> \
  --think-mode never|auto [--gate 0.9] [--think-budget 1024] [--return-details] \
  --output /tmp/<host>/results/<name>.jsonl \
  --model-path /tmp/.../models/<model dir> --model-archive-sha256 <zip sha256>
```

Behaviour:

* Every record stores the full raw server response, the request controls, the
  canonical config hash and the instruction hash. Resume refuses records from another
  model/signal/config or another instruction (and, for `query_answer`, another local
  response).
* Each chunk is flushed and `fsync`ed. A truncated trailing line is truncated on
  resume; successful records are never requested again.
* Per-question failures are appended to `<output>.failures.jsonl` with error type and
  are retried only in a bounded repair pass. Two consecutive fully-failed chunks abort
  the run instead of burning time.
* `--limit N` / `--probe-ids FILE` are probe modes: they never write metrics or
  summary. Probe records live in the formal output, so the full run skips them.
* **Metrics/summary are written only when the output contains exactly the 5,955 frozen
  ids, with no missing/extra/duplicate rows.** A run manifest
  (`<output>.run_manifest.json`) records config, input sha256, source sha256, model
  tree hash and counts.

## Replay / eval-only

```bash
# historical score file, metadata supplied from CLI (scores unchanged)
python replay_startlux_metrics.py --scores old.jsonl --method-name Jev \
  --router-model jev-1.13.0 --signal query

# baseline reconciliation against a reference metrics CSV (54 cells)
python replay_startlux_metrics.py --scores old.jsonl --method-name Jev \
  --compare-metrics reference.metrics.csv --check-json check.json
```

`--allow-partial` writes only a coverage check (`*.partial.check.json`) — never a full
`*.summary.json`.

## Integrity of the published copy

Verify with `sha256sum -c SHA256SUMS` before running. Files are read-only to the
consumers; publish any change under a new directory, never overwrite this one.
