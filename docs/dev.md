# Development log

## 2026-10-25 — Resume interrupted StartLux evaluation

### Question
The previous session stopped before launching the three StartLux executor tasks. Reuse the existing RouterXBench clean inputs, labels, scorer and baseline; start real scoring without repeating old experiments.

### Analysis / Root Cause
Only task briefs existed at recovery. Remote GPUs were running shepherd synthetic load, not evaluation. Existing 0.8B service smoke had passed; larger StartLux weights were new downloads. An unnecessary shared-script readiness dependency and waiting for model bundles delayed the first scoring request. The old main_summary uses a superseded ratio protocol; the latest explicit freeze retains historical absolute bands. One original Jev execution is distinct from the paper's three-execution aggregate.

### Solution
Launch three DSH workers with exclusive host ownership. Freeze script/config identity, preserve successful records across probes/resume, record item failures separately and retain raw API responses. Reuse existing evaluation math. Run already-downloaded 0.8B on host 1 while 27B downloads, start downloaded 9B independently on host 2, and run 4B then 2B on host 3. Retain exact controls and original labels/bands. Use shepherd for GPU work and persist incremental output via each host launcher.

### Validation
Planner independently replayed all 54 original-Jev metric cells against the authoritative historical-absolute reference: maximum absolute difference 1.5631940186722204e-13. Executor regression report covers eight focused CPU/loopback HTTP checks, including per-item failures, no duplicate successful resume, config mismatch, truncated tails and withholding complete summaries for missing IDs. Shared runtime was independently read back from HDFS and its published hashes checked. GPU/model results are pending actual score coverage; preparation does not count as evaluation completion.

### Files Changed
`AGENTS.md`; `experiment/run_startlux_routerxbench.py`; `experiment/replay_startlux_metrics.py`; `experiment/INPUT_SHA256.json`; `experiment/startlux/`; `experiment/STARTLUX_PLAN.md`; `docs/data.md`; `docs/dev.md`; `experiment/LOG.md`.

### Commit Hash
Implementation: `7fcdb2c` — Add resumable StartLux evaluation and frozen metric replay. Integrated replay also reproduces all 54 reference cells with zero difference.


## Startup update

At 04:35–04:36 CST on 2026-10-25, actual StartLux parents and eight per-GPU vLLM backends were observed for 0.8B (host1), 9B (host2), and 4B (host3). They were loading/initializing at the last observation; this does not establish scored-item completion. Runtime entry scripts: host1 `/tmp/startlux-eval-20261025/run_0p8b_instant.sh`; host2 `.../bin/wave_instant.sh` (then `wave_autothink.sh`); host3 `.../run/start_4b_probe.sh`. The executor notes remain the current source for exact PIDs and probe/full-run transition.


## First scoring evidence (04:38 CST)

Host1 0.8B-Instant produced 128 unique query records and 128 unique query_answer records during incremental concurrency probes. Planner read the actual JSONL records and verified model identity, finite normalized local/cloud probabilities and explicit `none_check=true`, `allow_none=false`, `think_mode=never`. Failure files were empty. Incremental copies are visible under persistent `results/host1/`; a later final independent readback is still required. Probe records are retained for full-run continuation, not counted as completed full conditions.

Host2 startup exposed stale shared FlashInfer build metadata pointing to an old Python3.10 environment, while current service is Python3.12. Current environment contains the required header. The worker was instructed to use a task-specific `FLASHINFER_WORKSPACE_BASE` without deleting another project's cache; this is an environment repair and does not change model/evaluation settings. Host3 continued initializing independently.


## 2026-10-25 — Accept complete StartLux results and comparison tables

### Question
Deliver the full StartLux matrix together with the previous RouterXBench methods in paper-style main tables.

### Analysis / Root Cause
The current manuscript tables use5,954 common IDs and changed local answers across three evaluation rounds. Mixing those cells with the frozen historical5,955 StartLux protocol would misstate comparability. Earlier service/cache/FUSE issues are resolved; no additional model or Judge calls are needed for old baselines.

### Solution
Keep the frozen StartLux protocol; build strict tables from accepted historical5,955 old-method sources and independently recomputed clean display denominators. Preserve the current paper cells in a separate reference. Retain all12 conditions and their actual repeat identities, plus editable table scripts/CSV/LaTeX, vector PDF/SVG and source provenance. Raw scores remain in HDFS; summaries/manifests/verification accompany the report.

### Validation
All71,460 records pass exact-ID, question/local-answer hash, model/config/controls and raw-response checks. Independent HDFS hashes match all48 raw/summary/metrics/manifest files. All648 replay cells agree exactly; all378 displayed metric values and historical baseline SDs match their sources. Preview inspected at final size; embedded fonts, no raster PDF objects, editable SVG text. No Overleaf changes.

### Files Changed
`AGENTS.md`, `docs/data.md`, `docs/dev.md`, `experiment/STARTLUX_PLAN.md`, `LOG.md`, `MAIN_RESULTS.md`, `experiment/startlux/results/`, `experiment/startlux/main_tables/`, final verification JSON.

### Commit Hash
Completed results: see the Git commit titled `Record complete StartLux results and matched RouterXBench tables`; implementation `7fcdb2c`.


## 2026-10-25 — Final resume acceptance

### Question
Resume the StartLux evaluation and finish any outstanding delivery.

### Analysis / Root Cause
All 12 frozen conditions were already complete; the result integration remained staged and the repository ignore rules excluded the result summaries/manifests and CSV tables. No additional inference was needed.

### Solution
Accept the completed matrix and explicitly include its small result artifacts and table CSVs in the result commit, preserving unrelated working changes.

### Validation
Rechecked all 12 local raw SHA256 values and 71,460 records, with exactly 5,955 unique sample_id values per condition. All 36 archived summary/metric/manifest hashes match the persistent-readback evidence. Existing independent replay checks cover 648 cells with zero difference and zero failed records.

### Files Changed
Result artifacts under `experiment/startlux/results/`, table CSVs, and the already staged evaluation documentation and table sources.

### Commit Hash
See the commit titled `Record complete StartLux results and matched RouterXBench tables`.
