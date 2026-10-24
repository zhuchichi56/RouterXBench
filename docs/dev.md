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
