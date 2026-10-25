# StartLux RouterXBench results — 2026-10-25

A1–A5 / B1–B2 complete: **12/12 conditions, 71,460 records, zero failures**. Each condition covers exactly the same 5,955 clean IDs. These are single exploratory executions; no StartLux SD or stability claim is inferred.

[Full AUROC/BQ/TC comparison](startlux/main_tables/MAIN_TABLES.md) · [PDF](startlux/main_tables/preview/main_tables_preview.pdf) · [CSV](startlux/main_tables/main_tables.csv) · [Separate current-paper reference](startlux/main_tables/PAPER_REFERENCE.md)

## Configuration and comparison

Five Instant sizes (0.8B/2B/4B/9B/27B), query and query_answer; 9B/27B AutoThink query, gate=0.9, think_budget=1024. Frozen local model is Llama-3.1-8B-Instruct with historical answers/labels. none_check=true, allow_none=false; local probability is the score, higher prefers local. Instant think_mode=never; AutoThink=auto. Eight independent H20 replicas per host via shepherd, vLLM0.19.1. No new generation, training or Judge calls. W&B off.

Metrics retain historical absolute budget/target bands. BQ and TC are normalized by actual clean all-cloud quality/cost only. TC excludes router, local model and Judge costs. The strict main tables include SelfAsk, Entropy, MaxLogits, Confidence Margin, EmbeddingMLP, HybridLLM, RouteLLM, Jev and MeanProbe. Learned SD uses three training seeds42/43/44; Jev uses three API executions; fixed methods use three actual executions. StartLux has one execution. Current-paper 5,954-ID/three-answer-round values remain a separate reference because local answers and coverage differ.

## Verification

Planner independently checked all 71,460 raw records: exact unique IDs, frozen instruction/answer hashes, returned model/config/signal, explicit controls, normalized probabilities and original API responses. All 12 raw JSONL, summaries, metric CSVs and manifests were read back from HDFS and match local hashes. All 648 AUROC/BQ/TC cells across three ranges replay exactly (maximum difference0). Independently recomputed and checked all162 baseline main-table values/SDs and216 StartLux cells. Evidence: `startlux/FINAL_LOCAL_TABLE_CHECKS.json`, `FINAL_PERSISTENCE_REPLAY_CHECKS.json`. Earlier original-Jev regression covers54 cells within1.57e-13.

All service parents/backends exited after their queues. Shepherd recovery is recorded in executor NOTES. Synthetic keepalive is excluded from effective inference throughput.

## Measured execution

The table uses actual final manifests: newly scored records and wall time exclude retained probe records. Effective rps uses those new records. Each probe retained its successful semantic decisions; full runs skipped them.

| Condition | Concurrency | New records / wall time | New records/s | Total | Failed |
|---|---:|---:|---:|---:|---:|
| 0.8b_instant_qa | 64 | 5827/190.9s | 30.52 | 5955 | 0 |
| 0.8b_instant_query | 64 | 5827/189.5s | 30.76 | 5955 | 0 |
| 27b_autothink_query | 32 | 5827/1835.9s | 3.17 | 5955 | 0 |
| 27b_instant_qa | 64 | 5827/269.3s | 21.64 | 5955 | 0 |
| 27b_instant_query | 64 | 5827/193.8s | 30.06 | 5955 | 0 |
| 2b_instant_qa | 32 | 5699/183.1s | 31.13 | 5955 | 0 |
| 2b_instant_query | 32 | 5699/182.8s | 31.17 | 5955 | 0 |
| 4b_instant_qa | 16 | 5699/188.0s | 30.31 | 5955 | 0 |
| 4b_instant_query | 16 | 5699/184.7s | 30.85 | 5955 | 0 |
| 9b_autothink_query | 64 | 5563/1205.3s | 4.62 | 5955 | 0 |
| 9b_instant_qa | 32 | 5179/167.6s | 30.89 | 5955 | 0 |
| 9b_instant_query | 8 | 5179/168.7s | 30.70 | 5955 | 0 |

Probe ranges: 8→16→32→64. Full settings are above; per-tier completion/error/backend-counter/latency records are in each executor NOTES and persistent logs. 0.8B probe rates were about19–31rps; 4B/2B about9–15rps on the longer probe subsets; 9B Instant query about28–31rps and QA23–27rps; AutoThink9B1.24→3.31rps and27B0.87–2.66rps. Longer probe inputs make these rates distinct from full-run rates. Host1 27B-Instant probe levels used different length blocks; this confounds cross-tier speed comparisons. Its final manifests and launch script both specify64 for query and QA; an executor note saying QA8 is incorrect. This affects throughput interpretation, not result identity.

## Reproduction and storage

Live/resume: `experiment/run_startlux_routerxbench.py`; metrics-only: `experiment/replay_startlux_metrics.py`. Exact original arguments, controls, model/config/input/script hashes and software are in `startlux/results/*.run_manifest.json`; published runtime remains frozen under HDFS `eval_bundle/recovery_v2/`. On host1 the archive field contains the sha256sum filename suffix; the first field is the verified archive SHA256.

Rebuild tables from the existing raw result directory:

```sh
python3 experiment/startlux/main_tables/build_main_tables.py --results-root /Users/zhuhe/wwx/output/routerxbench-startlux-20261025 --output-dir experiment/startlux/main_tables --require-complete
/Users/zhuhe/wwx/neural-interfaces26/.venv/bin/python experiment/startlux/main_tables/render_preview.py --scope strict
```

Sources and hashes: `startlux/main_tables/provenance.json`. Persistent raw results: `/mnt/hdfs/user/hadoop-aipnlp/zhuhe18/routerxbench/runs/startlux-eval-20261025/results/host{1,2,3}/`. Local run root: `/Users/zhuhe/wwx/output/routerxbench-startlux-20261025`. Rendered PDF has3 pages, embedded fonts and zero raster image objects; SVG preserves editable text. Native LaTeX sources are included; PDF previews use matplotlib.

Implementation commits: `7fcdb2c`, `74d8ea3`, `4004ac0`; completed-results commit is recorded in `docs/dev.md` and Git history.
