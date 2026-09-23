"""Run Jev routing on captured AgentBench states without executing environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentbench.jev_router import AgentStep, JevRouter


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args()

    router = JevRouter(confidence_threshold=args.threshold, model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for line_number, item in iter_jsonl(args.input):
            step = AgentStep(
                task_id=str(item.get("task_id", item.get("id", line_number))),
                step_id=int(item.get("step_id", 0)),
                instruction=item["instruction"],
                observation=item["observation"],
                available_actions=item["available_actions"],
                action_history=item.get("action_history", []),
            )
            record = dict(item)
            record["jev_decision"] = router.route(step).to_dict()
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
