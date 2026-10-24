# StartLux RouterXBench evaluation — 2026-10-25

Status: recovery and preparation; formal scoring completion is established only by unique persisted sample IDs.

## Frozen protocol

- Test split: 5,955 IDs from `output/routerxbench-clean-multiseed-20261013/SHARED_SPLIT.json` (outside this repository).
- Local model: `meta-llama/Llama-3.1-8B-Instruct`; reuse frozen generated answers and labels. No new external Judge calls.
- Signals: `query`, or `query_answer` with the actual frozen local answer.
- Score: probability of `local`, higher prefers the small model. `none_check=true`, `allow_none=false`; listed probabilities sum to one and `none_probability` is retained separately.
- Instant: `think_mode=never`. AutoThink: `think_mode=auto`, gate 0.9, budget 1,024. No quantization, prompt, model, or label changes.
- AUROC: native task ability labels; relative-label AUROC is separately identified.
- BQ/TC: historical absolute bands, with clean cloud quality/full cloud cost used only for display normalization. Authority: `output/routerxbench-clean-multiseed-20261013/LATEST_MAIN_TABLE_FREEZE.json`. Earlier clean-ratio summary is diagnostic and does not define this evaluation.
- One execution per condition is exploratory. Do not claim repeat stability from one execution.

## Queue

| ID | Host | Model | Signals | Next |
|---|---|---|---|---|
| A1 | 8h20-1 | 27B-Instant | query, query_answer | B1 |
| A2 | 8h20-2 | 9B-Instant | query, query_answer | B2 |
| A3 | 8h20-3 | 4B-Instant | query, query_answer | A4 |
| A4 | 8h20-3 | 2B-Instant | query, query_answer | A5 |
| A5 | 8h20-1 | 0.8B-Instant | query, query_answer | A1 |
| B1 | 8h20-1 | 27B-AutoThink | query | complete |
| B2 | 8h20-2 | 9B-AutoThink | query | complete |

Recovery queue update: run the already-downloaded A5 first on host 1 while A1 weights download; host 3 skips A5.

Use eight independent GPU replicas per host through `h20-shepherd run`. Model/environment preparation may overlap. Real end-to-end concurrency exploration is capped at about 30 minutes per host, using increasing levels and representative lengths; retain successful probe records for continuation. Select effective throughput, not worker count. Continue other samples after single-sample failures; retry only missing records with bounded backoff. Stop only an affected stage when a shared execution failure is established.

## Assets and evidence

Run root: `/Users/zhuhe/wwx/output/routerxbench-startlux-20261025`.
Persistent remote root: `/mnt/hdfs/user/hadoop-aipnlp/zhuhe18/routerxbench/runs/startlux-eval-20261025`.
Shared recovery scripts are published under `eval_bundle/recovery_v2/`, with a hashed `READY.json`. Existing inputs and old scripts are preserved.

DSH executor roots: `dsh-host1`, `dsh-host2`, `dsh-host3` below the local run root. Host 1 owns shared script recovery; all workers own only their assigned machine and work directory. Main agent owns verification and repository integration.

Planner baseline audit: `planner-audit/AUTHORITATIVE_BASELINE_CHECK.json`. All 54 original-Jev AUROC/BQ/TC cells match the frozen original-execution reference; maximum absolute discrepancy is 1.5631940186722204e-13. Current paper Jev entries aggregate three API executions, so do not compare one original execution directly with that aggregate.

## Acceptance

Each result has exactly the clean 5,955 unique IDs, matching query hashes, finite validated probabilities, expected model/signal/config identity, and separate failure/missing counts. Preserve complete responses, controls, source/config/input hashes, representative capacity evidence, and latency/token usage. Flush and persist incremental records; independently read back final HDFS artifacts before accepting completion. No inferred scores for missing samples.
