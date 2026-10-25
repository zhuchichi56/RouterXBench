# StartLux + RouterXBench strict 5,955 main tables

Snapshot: `2026-10-24T21:59:14Z`  
StartLux status: **12/12 conditions complete**; 0 pending. Strict baseline: 9 previous methods on the clean 5,955 split.

> **Scope.** This file is the strict 5,955 comparison only. The current paper tables use a different protocol (5,954 subset, historical greedy + sampled 42/43, changed local answers); they are kept unchanged in `PAPER_REFERENCE.md` and must not be combined with these tables or ranked together. `pending` marks a condition that has not produced a validated summary yet; it is not zero.

## Table 1: AUROC ×100 (higher is better)

*AUROC ×100; higher is better. Baseline: strict 5,955 sources, mean ± sample SD across the actual repeats named below. StartLux: one exploratory execution, no SD.*

| Method | Alpaca | MMLU | Big-Math | Magpie | MMLU-Pro | MATH |
|---|---:|---:|---:|---:|---:|---:|
| **Strict 5,955 baseline — mean ± sample SD; repeat identity in notes** |  |  |  |  |  |  |
| SelfAsk | 54.87 ± 0.00 | 54.14 ± 0.00 | 47.87 ± 0.00 | 44.08 ± 0.00 | 56.37 ± 0.00 | 44.14 ± 0.00 |
| Entropy | 49.78 ± 0.00 | 57.68 ± 0.00 | 48.61 ± 0.00 | 47.64 ± 0.00 | 57.11 ± 0.00 | 45.55 ± 0.00 |
| MaxLogits | 46.67 ± 0.00 | 56.07 ± 0.00 | 47.68 ± 0.00 | 40.54 ± 0.00 | 56.41 ± 0.00 | 43.04 ± 0.00 |
| Conf. Margin | 49.16 ± 0.00 | 53.75 ± 0.00 | 48.46 ± 0.00 | 47.00 ± 0.00 | 56.42 ± 0.00 | 41.97 ± 0.00 |
| Emb. MLP | 58.49 ± 0.66 | 63.69 ± 0.12 | 63.16 ± 0.60 | 72.19 ± 0.42 | 61.55 ± 0.30 | 65.82 ± 0.71 |
| HybridLLM | 63.70 ± 1.92 | 65.13 ± 1.29 | 64.21 ± 0.84 | 66.23 ± 8.05 | 60.56 ± 3.01 | 67.26 ± 2.39 |
| RouteLLM | 57.69 ± 0.43 | 58.47 ± 0.32 | 60.74 ± 0.06 | 74.40 ± 0.15 | 52.62 ± 0.44 | 61.74 ± 0.09 |
| Jev | 59.27 ± 0.08 | 69.77 ± 0.71 | 71.92 ± 0.57 | 70.59 ± 0.19 | 70.33 ± 0.22 | 80.47 ± 0.17 |
| MeanProbe | 65.71 ± 0.13 | 67.62 ± 0.11 | 67.76 ± 0.04 | 76.43 ± 0.11 | 64.66 ± 0.08 | 72.46 ± 0.18 |
| **StartLux — one exploratory run, no SD** |  |  |  |  |  |  |
| StartLux 0.8B Instant (Q) | 58.34 | 57.17 | 62.51 | 65.05 | 58.71 | 72.37 |
| StartLux 0.8B Instant (Q+A) | 55.19 | 66.82 | 81.77 | 58.66 | 71.32 | 87.25 |
| StartLux 2B Instant (Q) | 57.43 | 49.26 | 59.74 | 64.22 | 53.76 | 72.53 |
| StartLux 2B Instant (Q+A) | 62.49 | 74.93 | 87.62 | 73.34 | 79.50 | 91.36 |
| StartLux 4B Instant (Q) | 66.59 | 67.30 | 70.70 | 76.02 | 67.44 | 80.91 |
| StartLux 4B Instant (Q+A) | 68.92 | 88.17 | 92.51 | 78.09 | 87.47 | 94.72 |
| StartLux 9B Instant (Q) | 68.06 | 67.80 | 72.94 | 76.56 | 65.19 | 79.33 |
| StartLux 9B Instant (Q+A) | 74.09 | 90.71 | 93.29 | 79.28 | 89.39 | 95.64 |
| StartLux 27B Instant (Q) | 66.83 | 74.90 | 76.43 | 73.91 | 71.72 | 82.92 |
| StartLux 27B Instant (Q+A) | 73.90 | 95.15 | 96.49 | 81.92 | 93.51 | 98.11 |
| StartLux 9B AutoThink (Q) | 65.31 | 64.48 | 72.56 | 76.24 | 61.51 | 79.18 |
| StartLux 27B AutoThink (Q) | 68.10 | 74.49 | 75.37 | 80.53 | 71.93 | 82.85 |

*Strict clean 5,955 comparison: AUROC ×100 (higher is better). Baseline: 9 previous methods from the 5,955 clean sources; SD is across the actual repeats named in the table notes (training seeds, API executions or fixed executions). StartLux: one exploratory execution per condition (no SD). pending = condition not complete at this snapshot, not zero. The current paper tables are a separate, protocol-mismatched reference; do not combine them with this table.*

*Repeat identity: SelfAsk/Entropy/MaxLogits/Conf. Margin = 3 fixed executions; Emb. MLP/HybridLLM/RouteLLM/MeanProbe = 3 training seeds; Jev = 3 actual API executions. A reported 0.00 is a measured sample SD for identical fixed executions, not a missing value.*

Notes: `MeanProbe` remains in its original position after the midrule. Baseline signals are historical and are not restated (no per-method signal mapping is supplied by the frozen contracts); they are not guessed.

## Table 2: BQ, % (higher is better)

*BQ (%); higher is better. Baseline: strict 5,955 sources, mean ± sample SD across the actual repeats named below. StartLux: one exploratory execution, no SD.*

| Method | Alpaca | MMLU | Big-Math | Magpie | MMLU-Pro | MATH |
|---|---:|---:|---:|---:|---:|---:|
| **Strict 5,955 baseline — mean ± sample SD; repeat identity in notes** |  |  |  |  |  |  |
| SelfAsk | 92.21 ± 0.00 | 80.44 ± 0.00 | 81.11 ± 0.00 | 83.43 ± 0.00 | 67.25 ± 0.00 | 64.30 ± 0.00 |
| Entropy | 91.65 ± 0.00 | 79.37 ± 0.00 | 79.20 ± 0.00 | 81.03 ± 0.00 | 64.96 ± 0.00 | 65.54 ± 0.00 |
| MaxLogits | 92.56 ± 0.00 | 78.87 ± 0.00 | 78.74 ± 0.00 | 83.10 ± 0.00 | 64.29 ± 0.00 | 65.48 ± 0.00 |
| Conf. Margin | 91.61 ± 0.00 | 78.57 ± 0.00 | 80.06 ± 0.00 | 81.01 ± 0.00 | 65.18 ± 0.00 | 64.49 ± 0.00 |
| Emb. MLP | 91.27 ± 0.20 | 79.79 ± 0.16 | 78.92 ± 0.35 | 78.03 ± 0.14 | 65.37 ± 0.17 | 65.56 ± 0.13 |
| HybridLLM | 91.01 ± 0.07 | 80.19 ± 0.10 | 79.68 ± 0.46 | 78.09 ± 0.89 | 64.65 ± 1.11 | 64.95 ± 0.74 |
| RouteLLM | 90.96 ± 0.01 | 81.22 ± 0.05 | 79.73 ± 0.05 | 77.80 ± 0.07 | 69.30 ± 0.24 | 64.02 ± 0.12 |
| Jev | 92.59 ± 0.09 | 81.66 ± 0.05 | 79.62 ± 0.07 | 81.92 ± 0.06 | 61.62 ± 0.14 | 62.55 ± 0.08 |
| MeanProbe | 91.27 ± 0.03 | 80.75 ± 0.08 | 79.90 ± 0.04 | 78.01 ± 0.07 | 64.85 ± 0.14 | 65.81 ± 0.10 |
| **StartLux — one exploratory run, no SD** |  |  |  |  |  |  |
| StartLux 0.8B Instant (Q) | 92.34 | 81.06 | 79.87 | 78.87 | 63.39 | 64.78 |
| StartLux 0.8B Instant (Q+A) | 94.26 | 82.08 | 84.91 | 85.19 | 66.24 | 66.68 |
| StartLux 2B Instant (Q) | 92.46 | 82.61 | 79.62 | 79.61 | 65.61 | 66.15 |
| StartLux 2B Instant (Q+A) | 94.03 | 83.68 | 84.72 | 85.80 | 67.23 | 66.48 |
| StartLux 4B Instant (Q) | 92.30 | 80.92 | 79.65 | 81.84 | 61.61 | 63.05 |
| StartLux 4B Instant (Q+A) | 95.01 | 84.75 | 85.96 | 86.79 | 67.32 | 66.32 |
| StartLux 9B Instant (Q) | 91.96 | 82.03 | 78.92 | 82.33 | 61.93 | 63.59 |
| StartLux 9B Instant (Q+A) | 94.53 | 85.30 | 86.28 | 86.90 | 67.28 | 66.25 |
| StartLux 27B Instant (Q) | 92.68 | 80.79 | 79.33 | 83.49 | 62.41 | 62.49 |
| StartLux 27B Instant (Q+A) | 95.21 | 87.38 | 86.98 | 86.97 | 68.33 | 65.58 |
| StartLux 9B AutoThink (Q) | 92.12 | 80.64 | 78.87 | 82.10 | 63.96 | 62.94 |
| StartLux 27B AutoThink (Q) | 91.88 | 81.45 | 79.79 | 82.28 | 61.65 | 62.54 |

*Strict clean 5,955 comparison: BQ at the middle budget range, as a percentage of all-cloud quality (higher is better). BQ is normalized only by the actual 5,955 clean Qcloud. Baseline: 9 previous methods, SD across the actual repeats named in the table notes. StartLux: one exploratory execution per condition (no SD). pending = condition not complete at this snapshot, not zero. The current paper tables are a separate, protocol-mismatched reference; do not combine them with this table.*

*Repeat identity: SelfAsk/Entropy/MaxLogits/Conf. Margin = 3 fixed executions; Emb. MLP/HybridLLM/RouteLLM/MeanProbe = 3 training seeds; Jev = 3 actual API executions. A reported 0.00 is a measured sample SD for identical fixed executions, not a missing value.*

Notes: `MeanProbe` remains in its original position after the midrule. Baseline signals are historical and are not restated (no per-method signal mapping is supplied by the frozen contracts); they are not guessed.

## Table 3: TC, % (lower is better)

*TC (%); lower is better. Baseline: strict 5,955 sources, mean ± sample SD across the actual repeats named below. StartLux: one exploratory execution, no SD.*

| Method | Alpaca | MMLU | Big-Math | Magpie | MMLU-Pro | MATH |
|---|---:|---:|---:|---:|---:|---:|
| **Strict 5,955 baseline — mean ± sample SD; repeat identity in notes** |  |  |  |  |  |  |
| SelfAsk | 80.61 ± 0.00 | 76.25 ± 0.00 | 73.93 ± 0.00 | 75.54 ± 0.00 | 82.81 ± 0.00 | 80.30 ± 0.00 |
| Entropy | 76.83 ± 0.00 | 78.33 ± 0.00 | 79.47 ± 0.00 | 68.09 ± 0.00 | 80.59 ± 0.00 | 73.03 ± 0.00 |
| MaxLogits | 74.64 ± 0.00 | 78.46 ± 0.00 | 79.02 ± 0.00 | 67.88 ± 0.00 | 83.76 ± 0.00 | 73.69 ± 0.00 |
| Conf. Margin | 76.51 ± 0.00 | 78.90 ± 0.00 | 76.86 ± 0.00 | 68.37 ± 0.00 | 80.40 ± 0.00 | 72.93 ± 0.00 |
| Emb. MLP | 81.65 ± 0.24 | 77.73 ± 1.17 | 67.09 ± 1.29 | 87.88 ± 0.39 | 85.44 ± 1.14 | 76.51 ± 0.33 |
| HybridLLM | 85.07 ± 2.29 | 76.69 ± 1.13 | 66.60 ± 1.37 | 83.04 ± 9.94 | 87.33 ± 2.02 | 75.99 ± 3.31 |
| RouteLLM | 88.43 ± 0.28 | 67.11 ± 0.30 | 68.96 ± 0.24 | 91.28 ± 0.16 | 70.21 ± 0.62 | 76.96 ± 0.18 |
| Jev | 68.30 ± 0.58 | 68.20 ± 1.26 | 67.42 ± 0.44 | 64.84 ± 0.10 | 81.14 ± 0.28 | 74.79 ± 0.52 |
| MeanProbe | 81.12 ± 0.25 | 72.57 ± 0.15 | 68.34 ± 0.23 | 84.91 ± 0.35 | 85.67 ± 0.02 | 73.41 ± 0.16 |
| **StartLux — one exploratory run, no SD** |  |  |  |  |  |  |
| StartLux 0.8B Instant (Q) | 74.14 | 75.59 | 72.25 | 74.02 | 81.49 | 73.03 |
| StartLux 0.8B Instant (Q+A) | 52.29 | 63.80 | 56.98 | 53.35 | 75.93 | 64.37 |
| StartLux 2B Instant (Q) | 78.10 | 73.70 | 70.55 | 75.04 | 77.61 | 73.29 |
| StartLux 2B Instant (Q+A) | 56.38 | 59.21 | 46.23 | 52.44 | 67.68 | 62.34 |
| StartLux 4B Instant (Q) | 74.33 | 66.63 | 70.69 | 68.20 | 81.59 | 73.13 |
| StartLux 4B Instant (Q+A) | 50.05 | 40.02 | 37.13 | 48.07 | 56.90 | 57.35 |
| StartLux 9B Instant (Q) | 77.64 | 66.12 | 67.88 | 69.46 | 82.04 | 76.67 |
| StartLux 9B Instant (Q+A) | 56.68 | 39.23 | 34.09 | 51.06 | 58.47 | 57.46 |
| StartLux 27B Instant (Q) | 69.82 | 62.67 | 69.04 | 62.83 | 82.49 | 73.11 |
| StartLux 27B Instant (Q+A) | 47.15 | 31.52 | 31.47 | 46.49 | 47.58 | 54.35 |
| StartLux 9B AutoThink (Q) | 73.06 | 69.25 | 70.54 | 67.73 | 79.39 | 75.35 |
| StartLux 27B AutoThink (Q) | 78.87 | 60.51 | 71.51 | 72.89 | 81.46 | 73.11 |

*Strict clean 5,955 comparison: TC at the middle quality-target range, as a percentage of all-cloud generation cost (lower is better). TC is normalized only by the actual 5,955 clean full_cloud_cost. Baseline: 9 previous methods, SD across the actual repeats named in the table notes. StartLux: one exploratory execution per condition (no SD). pending = condition not complete at this snapshot, not zero. The current paper tables are a separate, protocol-mismatched reference; do not combine them with this table.*

*Repeat identity: SelfAsk/Entropy/MaxLogits/Conf. Margin = 3 fixed executions; Emb. MLP/HybridLLM/RouteLLM/MeanProbe = 3 training seeds; Jev = 3 actual API executions. A reported 0.00 is a measured sample SD for identical fixed executions, not a missing value.*

Notes: `MeanProbe` remains in its original position after the midrule. Baseline signals are historical and are not restated (no per-method signal mapping is supplied by the frozen contracts); they are not guessed.

## Completion and provenance

| Item | Count |
|---|---:|
| Strict baseline methods | 9 |
| Strict baseline cells (3 metrics × 6 datasets) | 162 |
| StartLux conditions complete | 12 / 12 |
| StartLux cells with values | 216 |
| StartLux cells pending | 0 |

### Pending StartLux conditions

None: all 12 planned conditions have validated summaries.

### Strict sources and normalization

- `mean_table`: `/Users/zhuhe/wwx/output/routerxbench-clean-main-3seed-reviewed-20261013/MEAN_TABLE.csv`  
  SHA256 `5abe83320ae121da3e4a66cc109191d3782f320f6aaff772bc63e37fdebcef24`
- `fixed_mean_table`: `/Users/zhuhe/wwx/output/router-fixed-repeat-tables-20261013/FIXED_MEAN_TABLE.csv`  
  SHA256 `dd53f94ce04317d73c542744db27db31d99a35c7d5866fb0c380ac3c8a3a31e0`
- `jev_mean_table`: `/Users/zhuhe/wwx/output/routerxbench-free-real-repeats-20261013/JEV_MEAN_TABLE.csv`  
  SHA256 `942be01ff0c126e5613789aaacb4a872924301538a14a858d377f78adff2aa24`
- `strict_main_table`: `/Users/zhuhe/wwx/output/routerxbench-clean-main-3seed-reviewed-20261013/MAIN_TABLE.csv`  
  SHA256 `89dcbe676278765b0914ba73e5a5a76bf99604e3d40fa279ed15ffafd0e7e625`
- `denominator_manifest`: `/Users/zhuhe/wwx/output/routerxbench-clean-multiseed-20261013/main_summary/MANIFEST.json`  
  SHA256 `5c3f57c5457329df40331bf9534959b5c08317463e1c6eee0f514e76f87d54c1`
- Actual 5,955 denominators: `main_summary/MANIFEST.json`; independent recomputation verified: True (max |Qcloud diff| 0.0, max |cost diff| 9.769962616701378e-15).
- Fixed-vs-MEAN nonlearned cross-check: max |AUROC| 5.55e-17, max |BQ| 8.88e-16, max |TC| 4.44e-16.
- JEV-vs-MEAN Jev difference (expected repeat-set difference, JEV selected): max |AUROC| 0.007182, max |BQ| 0.008362, max |TC| 0.0295.

### StartLux source files

- `autothink_27b_query`: `dsh-host1/27b_autothink_query.summary.json`  
  source SHA256 `a947a9942fec89311bff73c137481d3b49269a32fdf6ee45d79f06ee16eff17b`; scores SHA256 `65b358d81a5d46302d2e267cabd05d544b04a83333ba80507d841c6b6744f90d`; n_requests 5955; full_frozen_id_set true.
- `autothink_9b_query`: `dsh-host2/9b_autothink_query.summary.json`  
  source SHA256 `a4c5163af79200c9a70d728db3529a9b0f786fea958342033abc283985d307c8`; scores SHA256 `8f67e4f92aecd3c98e9a4d53071ba7ae89325846eae83937725ef628ec203510`; n_requests 5955; full_frozen_id_set true.
- `instant_0.8b_query`: `dsh-host1/0.8b/0.8b_instant_query.summary.json`  
  source SHA256 `46e8a42ff3731f72f28ec52b1cfea7dcb0ffd70baf0e979d869853c18948e66f`; scores SHA256 `a23887660f04ca3c9d7fa115464b88d07586b12e8ca6d6e56b3398d0137852a9`; n_requests 5955; full_frozen_id_set true.
- `instant_0.8b_query_answer`: `dsh-host1/0.8b/0.8b_instant_qa.summary.json`  
  source SHA256 `fe483e8c578c732e5df981d93db1e134d18512798b44b61a0be2117730bc37fb`; scores SHA256 `55b3768852227d76c352eb4401112069a70dff94059c6a4c8d6fdd6f485b6b6f`; n_requests 5955; full_frozen_id_set true.
- `instant_27b_query`: `dsh-host1/27b_instant_query.summary.json`  
  source SHA256 `64349ae39859a890409aff738d523815377cbee10f24d76f1cfc57a9085a38a2`; scores SHA256 `68bf48e735e97cef2c0cbb467d4e9b2ee183ae80875635968185641a2c880ac3`; n_requests 5955; full_frozen_id_set true.
- `instant_27b_query_answer`: `dsh-host1/27b_instant_qa.summary.json`  
  source SHA256 `c31a8ba34a9c9526ba967e201136a29b3dadadb777bdd4586fb6191ff1ee6af0`; scores SHA256 `20d4d20087226b7d5b15bd7108a4fa0047bbc379cb4e61e2584b59090c77fe39`; n_requests 5955; full_frozen_id_set true.
- `instant_2b_query`: `dsh-host3/2b_instant_query.summary.json`  
  source SHA256 `315a511df21b107dc9afff9aa606bde206734c854833abc403d4f27e33393779`; scores SHA256 `f85859a87e8e347850b8902bbb1815b0621801c97bf3e0f95d21339906156a0f`; n_requests 5955; full_frozen_id_set true.
- `instant_2b_query_answer`: `dsh-host3/2b_instant_qa.summary.json`  
  source SHA256 `c8bc7b5ca236d01aaf1b02620f9a8f28ca16befebddedee7844b37d296f6679d`; scores SHA256 `8f8d7851079d0de509d7a46ea2984e6f756c013b4498f954d87455400a32bd81`; n_requests 5955; full_frozen_id_set true.
- `instant_4b_query`: `dsh-host3/4b_instant_query.summary.json`  
  source SHA256 `5a2cefd51af4fc5d5c3b6134fc8bd962cdccefa1388e747f1af8c60c465144fe`; scores SHA256 `a8cec15a9b4b0125cd380903721adae304e7639f917521fc74e453837e982e7f`; n_requests 5955; full_frozen_id_set true.
- `instant_4b_query_answer`: `dsh-host3/4b_instant_qa.summary.json`  
  source SHA256 `a474b1a0c630bc6dcdfb7490847b8ec6e2a99680e44f7f86b39eee0fc667f60d`; scores SHA256 `4d68dd7ea5c7d9063fd85462f972984b91146a60d389d67fef44bc113bd186d7`; n_requests 5955; full_frozen_id_set true.
- `instant_9b_query`: `dsh-host2/9b_instant_query.summary.json`  
  source SHA256 `74b443d1556ef951707e951c52788470b2558ff0d8a95e2090ed0bca83550e7a`; scores SHA256 `f42fa38ee2424607e74be081f7d89ba488e31f69d901742b93ff91040c8eb9ad`; n_requests 5955; full_frozen_id_set true.
- `instant_9b_query_answer`: `dsh-host2/9b_instant_qa.summary.json`  
  source SHA256 `a0c46b17f446ab81a27b7e6fbd45de36eb62997e4488aa7fada743b35f62fbd4`; scores SHA256 `d5f71cf919dc0ae97e06a5d18358975c855e418101a6de44312b66687681b2bc`; n_requests 5955; full_frozen_id_set true.

### Regeneration

```bash
python3 build_main_tables.py --results-root /Users/zhuhe/wwx/output/routerxbench-startlux-20261025 --output-dir /Users/zhuhe/wwx/output/routerxbench-startlux-20261025/dsh-table
python3 build_main_tables.py --results-root /Users/zhuhe/wwx/output/routerxbench-startlux-20261025 --output-dir /Users/zhuhe/wwx/output/routerxbench-startlux-20261025/dsh-table --require-complete  # only after all 12 arrive
```

Full source hashes and condition keys are in `provenance.json`; the tidy strict rows are in `main_tables.csv`; the editable LaTeX is under `tables/`. The paper reference is separate in `PAPER_REFERENCE.md` / `paper_reference.csv`.
