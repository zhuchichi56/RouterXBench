# RouterXBench project execution

Preserve unrelated uncommitted changes. Do not modify frozen historical assets under the sibling `output/` tree. Main-table StartLux evaluation uses historical absolute BQ/TC bands and clean endpoints only as display denominators; `LATEST_MAIN_TABLE_FREEZE.json` is authoritative over earlier clean-ratio summaries.

## StartLux evaluation entry points

- Live scoring / resume: `experiment/run_startlux_routerxbench.py` (resume is the same command and output path; successful IDs are skipped and condition mismatch is refused).
- Metrics-only replay: `experiment/replay_startlux_metrics.py`.
- Frozen input checksums: `experiment/INPUT_SHA256.json`.
- Queue, configuration and acceptance: `experiment/STARTLUX_PLAN.md`.
- Published remote runtime: `/mnt/hdfs/user/hadoop-aipnlp/zhuhe18/routerxbench/runs/startlux-eval-20261025/eval_bundle/recovery_v2/`; scripts are frozen after READY publication.
- Executor launch/control: `bash ~/claude-dotfiles-mt/skills/dsh-delegate/scripts/dsh_run.sh --context-file <TASK_CONTEXT.md> <BRIEF.md> <WORKDIR> 6`; monitor with `python3 -I ~/claude-dotfiles-mt/skills/dsh-delegate/scripts/dsh_status.py <WORKDIR> --tail 3`; continue/correct with `dsh_steer.sh <WORKDIR> <instruction>`.
- Work directories: `/Users/zhuhe/wwx/output/routerxbench-startlux-20261025/dsh-host{1,2,3}`. Actual per-host launch scripts live there and under remote `/tmp/startlux-eval-20261025/`; record the exact entry and PID in NOTES before launch. Never start a second copy while the first is active.
- GPU service/evaluation launch: `~/.local/h20-shepherd/h20-shepherd run -- bash <host-launch-script>`; never bypass task-mode or kill unknown processes. Stop only the exact recorded service/launcher PID and use its trap/finally path for descendants and shepherd recovery.
- Typical scoring command inside that launcher: `python <published-dir>/run_startlux_routerxbench.py --expected-model <model> --method-name <method> --signal <query|query_answer> --think-mode <never|auto> --concurrency <measured> --output <scores.jsonl> --model-path <weights> --require-frozen-inputs`. AutoThink adds `--gate 0.9 --think-budget 1024`; retain full controls in config identity.

Use verified throughput settings; do not rerun completed semantic requests for capacity exploration. A single failed item does not stop the batch. Accept results only after exact unique-ID coverage and independent persistent readback. Keep `docs/dev.md`, `docs/data.md` and experiment logs aligned with actual evidence. Do not push master or sync Overleaf as part of this evaluation.
