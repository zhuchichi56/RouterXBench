# Paper reference tables (unchanged, protocol-mismatched)

Snapshot: `2026-10-24T21:59:14Z`  
> **Not the strict main table.** These are the current paper values, copied exactly from the Overleaf table fragments. Their protocol is a 5,954 subset with historical greedy + sampled 42/43 rounds and changed local answers, not the clean 5,955 frozen-answer protocol used by StartLux. They are shown separately for reference only. Do not compare cells directly with the strict tables or combine the two into a rank.

## Table 1: Paper reference: AUROC ×100

*AUROC ×100; higher is better. Values copied exactly from the current paper table (mean ± sample SD, three evaluation rounds).*

| Method | Alpaca | MMLU | Big-Math | Magpie | MMLU-Pro | MATH |
|---|---:|---:|---:|---:|---:|---:|
| **Paper reference — mean ± sample SD, 3 rounds (protocol differs)** |  |  |  |  |  |  |
| SelfAsk | 53.60 ± 1.15 | 54.70 ± 0.93 | 48.22 ± 0.41 | 43.96 ± 0.38 | 55.31 ± 1.15 | 43.94 ± 0.39 |
| Entropy | 49.28 ± 0.61 | 57.32 ± 0.74 | 52.05 ± 3.13 | 48.41 ± 1.96 | 56.51 ± 0.55 | 44.74 ± 1.16 |
| MaxLogits | 47.34 ± 0.69 | 56.27 ± 0.78 | 51.70 ± 3.51 | 41.13 ± 2.19 | 56.10 ± 0.28 | 42.44 ± 0.91 |
| Conf. Margin | 48.21 ± 1.29 | 53.25 ± 1.28 | 51.46 ± 2.79 | 47.73 ± 1.02 | 55.44 ± 1.31 | 42.75 ± 0.72 |
| Emb. MLP | 58.71 ± 0.27 | 63.71 ± 0.80 | 62.47 ± 0.74 | 71.30 ± 0.78 | 60.68 ± 1.02 | 66.25 ± 0.40 |
| HybridLLM | 63.20 ± 0.45 | 65.08 ± 0.08 | 62.90 ± 1.15 | 65.29 ± 0.83 | 60.16 ± 0.67 | 68.32 ± 0.96 |
| RouteLLM | 57.14 ± 0.73 | 58.72 ± 0.51 | 60.10 ± 0.85 | 72.86 ± 1.33 | 52.55 ± 0.49 | 62.34 ± 0.67 |
| Jev | 58.56 ± 0.99 | 69.59 ± 0.54 | 71.67 ± 0.28 | 70.46 ± 0.95 | 70.74 ± 0.53 | 81.93 ± 1.34 |
| MeanProbe | 66.16 ± 0.73 | 66.95 ± 0.95 | 68.26 ± 0.58 | 75.69 ± 1.00 | 64.60 ± 0.50 | 73.13 ± 0.59 |

*Paper reference, unchanged: clean-test router ability (AUROC ×100). Entries report mean ± sample SD across three evaluation rounds. Protocol differs from the strict 5,955 tables: 5,954 subset, historical greedy + sampled 42/43, changed local answers. Not strictly matched to StartLux 5,955; do not compare directly or combine ranks.*

Notes: `MeanProbe` remains in its original position after the midrule. Baseline signals are historical and are not restated (no per-method signal mapping is supplied by the frozen contracts); they are not guessed.

## Table 2: Paper reference: BQ, %

*BQ (%); higher is better. Values copied exactly from the current paper table (mean ± sample SD, three evaluation rounds).*

| Method | Alpaca | MMLU | Big-Math | Magpie | MMLU-Pro | MATH |
|---|---:|---:|---:|---:|---:|---:|
| **Paper reference — mean ± sample SD, 3 rounds (protocol differs)** |  |  |  |  |  |  |
| SelfAsk | 91.93 ± 0.24 | 81.09 ± 0.71 | 81.01 ± 1.02 | 83.14 ± 0.27 | 67.02 ± 0.31 | 62.94 ± 1.40 |
| Entropy | 91.44 ± 0.22 | 79.75 ± 0.70 | 79.62 ± 1.37 | 80.66 ± 0.38 | 65.02 ± 1.25 | 64.00 ± 1.58 |
| MaxLogits | 92.42 ± 0.14 | 79.45 ± 0.96 | 79.56 ± 1.22 | 82.77 ± 0.33 | 64.44 ± 1.00 | 63.64 ± 1.78 |
| Conf. Margin | 91.25 ± 0.34 | 79.30 ± 1.18 | 80.44 ± 1.49 | 80.56 ± 0.41 | 65.45 ± 1.24 | 63.04 ± 1.38 |
| Emb. MLP | 90.99 ± 0.28 | 80.59 ± 0.84 | 78.98 ± 1.26 | 77.42 ± 0.53 | 65.04 ± 1.43 | 63.54 ± 1.88 |
| HybridLLM | 90.73 ± 0.25 | 81.07 ± 0.98 | 79.70 ± 1.16 | 77.54 ± 0.48 | 64.86 ± 1.59 | 63.21 ± 1.65 |
| RouteLLM | 90.68 ± 0.27 | 81.83 ± 0.79 | 79.62 ± 1.24 | 77.19 ± 0.53 | 69.41 ± 1.06 | 62.21 ± 1.65 |
| Jev | 92.13 ± 0.40 | 81.97 ± 0.60 | 79.19 ± 1.27 | 81.39 ± 0.46 | 61.56 ± 1.24 | 60.57 ± 1.86 |
| MeanProbe | 90.99 ± 0.26 | 81.57 ± 0.82 | 79.95 ± 1.47 | 77.45 ± 0.49 | 64.45 ± 1.64 | 63.94 ± 1.75 |

*Paper reference, unchanged: BQ at the middle budget range, as percentages of all-cloud quality; higher is better. Entries report mean ± sample SD across three evaluation rounds. Protocol differs from the strict 5,955 tables: 5,954 subset, historical greedy + sampled 42/43, changed local answers. Not strictly matched to StartLux 5,955; do not compare directly or combine ranks.*

Notes: `MeanProbe` remains in its original position after the midrule. Baseline signals are historical and are not restated (no per-method signal mapping is supplied by the frozen contracts); they are not guessed.

## Table 3: Paper reference: TC, %

*TC (%); lower is better. Values copied exactly from the current paper table (mean ± sample SD, three evaluation rounds).*

| Method | Alpaca | MMLU | Big-Math | Magpie | MMLU-Pro | MATH |
|---|---:|---:|---:|---:|---:|---:|
| **Paper reference — mean ± sample SD, 3 rounds (protocol differs)** |  |  |  |  |  |  |
| SelfAsk | 81.33 ± 1.01 | 74.70 ± 1.39 | 73.91 ± 2.87 | 76.14 ± 0.75 | 84.05 ± 1.43 | 81.41 ± 1.06 |
| Entropy | 78.09 ± 1.90 | 77.79 ± 1.13 | 75.06 ± 4.48 | 69.30 ± 1.09 | 81.52 ± 1.78 | 74.00 ± 1.12 |
| MaxLogits | 74.47 ± 0.74 | 77.16 ± 1.92 | 74.67 ± 4.81 | 68.79 ± 0.96 | 84.29 ± 1.32 | 74.48 ± 0.69 |
| Conf. Margin | 78.47 ± 1.67 | 78.32 ± 2.23 | 72.86 ± 4.00 | 69.83 ± 1.33 | 81.67 ± 1.23 | 73.90 ± 0.96 |
| Emb. MLP | 82.10 ± 0.57 | 76.81 ± 0.86 | 68.62 ± 1.52 | 89.05 ± 1.14 | 85.68 ± 1.06 | 77.33 ± 0.79 |
| HybridLLM | 85.71 ± 0.55 | 76.42 ± 1.50 | 69.50 ± 2.90 | 84.08 ± 1.10 | 87.62 ± 1.01 | 76.60 ± 1.01 |
| RouteLLM | 88.80 ± 0.55 | 66.14 ± 0.93 | 70.77 ± 1.75 | 91.98 ± 0.61 | 70.50 ± 1.35 | 78.08 ± 1.09 |
| Jev | 69.40 ± 1.14 | 67.44 ± 2.33 | 68.86 ± 2.48 | 66.39 ± 1.35 | 80.23 ± 1.20 | 75.00 ± 0.34 |
| MeanProbe | 81.57 ± 0.35 | 73.44 ± 0.98 | 67.35 ± 1.46 | 86.65 ± 1.60 | 85.34 ± 0.88 | 74.77 ± 1.47 |

*Paper reference, unchanged: TC at the middle quality-target range, as percentages of all-cloud generation cost; lower is better. Entries report mean ± sample SD across three evaluation rounds. Protocol differs from the strict 5,955 tables: 5,954 subset, historical greedy + sampled 42/43, changed local answers. Not strictly matched to StartLux 5,955; do not compare directly or combine ranks.*

Notes: `MeanProbe` remains in its original position after the midrule. Baseline signals are historical and are not restated (no per-method signal mapping is supplied by the frozen contracts); they are not guessed.

## Protocol and provenance

Source tables: `/Users/zhuhe/overleaf/router/table1_auroc.tex`, `table2_bq.tex`, `table3_tc.tex` (read-only, unchanged).
Paper data source: `/Users/zhuhe/overleaf/router/data/main_three_round/results.csv` and `verification.json`; 5,954 subset, historical greedy + sampled 42/43, changed local answers. These values are not the strict 5,955 values.
- `table1_auroc.tex`: `/Users/zhuhe/overleaf/router/table1_auroc.tex`; SHA256 `cae5d6ce90b697a9f3c7ea69d0416d06d9eb41599018fb9f577913db4752b3c1`; 9 methods × 6 datasets.
- `table2_bq.tex`: `/Users/zhuhe/overleaf/router/table2_bq.tex`; SHA256 `8de4a4fc92911ef2db3f1ccec3702d6dee2ccd7579eb3d5ce92644920ea132d8`; 9 methods × 6 datasets.
- `table3_tc.tex`: `/Users/zhuhe/overleaf/router/table3_tc.tex`; SHA256 `95a64104d5463a7aa189082f1d377f4ffafff60d7fd047eadb2bfd10cbfae39d`; 9 methods × 6 datasets.

Do not combine this file with `MAIN_TABLES.md`; the two protocols are not matched. The strict main table is the comparison table.
