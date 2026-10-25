# Data provenance

## StartLux evaluation (2026-10-25)

This evaluation reuses the existing RouterXBench clean split and scored local/cloud outputs. It performs no data generation, retraining, or new Judge labeling. All asset paths below are relative to `/Users/zhuhe/wwx/output/` and remain outside Git.

| Asset | Source | SHA256 |
|---|---|---|
| Test split | `routerxbench-clean-multiseed-20261013/SHARED_SPLIT.json` | `b3a62b617188ce0f31c3897aee0769d49195be85c3cdfd8a599766fa715e7501` |
| Native labels | `routerxbench-jev-gpt54-5958-20261011/all_labels.jsonl` | `772adbbe2870cd5f09bb54cc5b3a7ea18e88206e30918d2db355bf5c1b594001` |
| Joined quality/cost | `routerxbench-actual-cost-replay-20261012/JOINED_USAGE_QUALITY_SCORES.jsonl` | `c65f6bcdffd8f22c24691cc83ab928551947b5d182d4718361479ce24da1f8cb` |
| BQ scorer | `routerxbench-actual-cost-replay-20261012/budget_band.py` | `966536d89add4f21fbc9a22330af25197ea9c06bf30ebd9e64dded033a135801` |
| TC scorer | `routerxbench-tc-target-band-20261012/target_band.py` | `a6cf5b052a7214f32a2a383f800afa02e0a1b6b6e6351c794146eadd72fe3bed` |

Requests and local responses come from `routerxbench-jev-gpt54-5958-20261011/input.jsonl` and `small.jsonl`. Only IDs in the clean 5,955-item test split are evaluated. Historical 5,958-row score files are filtered to that split during replay. Preserve historical labels and model/Judge metadata as provenance; their presence does not authorize fresh calls to those models.

Use the existing split/prefix audits in `routerxbench-clean-multiseed-20261013/` as prior evidence; do not describe complete semantic disjointness as established. Runtime admission must verify file hashes, schema, unique IDs, and query/answer alignment. No test labels are used for StartLux training or prompt selection.

## Accepted StartLux output and comparison scopes

All 12 conditions contain the identical 5,955-ID set. Planner checks cover original instruction text/hash, frozen local-answer hash for query_answer, explicit controls, returned model, raw probabilities, summary hashes and separate HDFS readback. Each raw JSONL remains outside Git under persistent `runs/startlux-eval-20261025/results/host{1,2,3}/`. Git retains summaries/metrics/manifests and final verification JSON in `experiment/startlux/`.

The strict comparison uses the historical 5,955-answer protocol: learned baseline seeds from `routerxbench-clean-main-3seed-reviewed-20261013`, Jev API repetitions from `routerxbench-free-real-repeats-20261013`, and fixed-method executions from `router-fixed-repeat-tables-20261013`. Clean denominators are independently recomputed from joined quality/cost rows; historical absolute bands are unchanged. Current paper tables instead aggregate historical greedy plus sampled42/43 answers on 5,954 common IDs. Those unchanged numbers are supplied separately in `PAPER_REFERENCE.md`; they are not presented as matched StartLux comparisons.
