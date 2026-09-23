# AgentBench + Jev branch

This branch extends RouterXBench from request-level routing to selective control
inside an agent trajectory. It does not change the existing Knowledge or Math
branches.

## Frozen first slice

- Benchmark: AgentBench WebShop, upstream commit
  `d1e4a10db08c87075c78972e48ecc182be03e2d5`.
- Unit of routing: one environment step.
- Candidate set: the exact actions returned by `env.get_available_actions()`.
- Router: Jev choice probability from `typesafe-sdk`.
- Decision: execute Jev's action when confidence is at least `tau`; otherwise
  call the strong agent for that step.
- Seed: 42 for task selection and every supported model request.
- Primary curve: task success versus strong-agent call rate over `tau`.
- Required baselines: strong-only and a cost-matched cheap-generative cascade.
- Raw artifacts: task identity, observation, complete candidate set, selected
  action, probabilities, threshold, route, latency, tokens, reward and terminal
  status for every step.

The first formal comparison must wait for official strong-only AgentBench
alignment and a bound `TYPESAFE_API_KEY`. A replay-only adapter is available:

```bash
PYTHONPATH=src python src/agentbench/run_jev_replay.py \
  --input experiment/agentbench_jev/states.jsonl \
  --output experiment/agentbench_jev/jev-decisions.jsonl \
  --threshold 0.5
```

WebShop is first because it exposes a bounded action set natively. ALFWorld can
reuse the same router over `admissible_commands`; free-form OS/DB tasks require
a separate typed action schema and are outside this first slice.
