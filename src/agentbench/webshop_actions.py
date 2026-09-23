"""Translate native AgentBench WebShop actions into Jev choices."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def bounded_click_actions(available_actions: Mapping[str, Any]) -> list[str]:
    """Return only directly executable bounded actions.

    WebShop search requires generating arbitrary keywords, so the literal
    ``search`` control is deliberately excluded and delegated to the strong
    agent. All remaining clickables are exact environment actions.
    """

    clickables: Sequence[str] = available_actions.get("clickables", []) or []
    return [f"click[{value}]" for value in clickables if value.lower() != "search"]


def action_to_tool_call(action: str, call_id: str) -> dict[str, Any]:
    if action.startswith("click[") and action.endswith("]"):
        name = "click_action"
        arguments = {"value": action[6:-1]}
    elif action.startswith("search[") and action.endswith("]"):
        name = "search_action"
        arguments = {"keywords": action[7:-1]}
    else:
        raise ValueError(f"Unsupported WebShop action: {action}")
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
