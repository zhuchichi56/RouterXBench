"""Jev-backed routing for bounded AgentBench action spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence


class SystemOneClient(Protocol):
    def system_one(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class AgentStep:
    task_id: str
    step_id: int
    instruction: str
    observation: str
    available_actions: Sequence[str]
    action_history: Sequence[str] = ()


@dataclass(frozen=True)
class JevDecision:
    action: str | None
    confidence: float
    route: str
    model: str | None
    probabilities: Mapping[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JevRouter:
    """Choose a bounded action with Jev or delegate to the strong agent."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model: str = "jev-latest",
        client: SystemOneClient | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.confidence_threshold = confidence_threshold
        self.model = model
        self._client = client

    def _get_client(self) -> SystemOneClient:
        if self._client is None:
            try:
                from typesafe_sdk import TypeSafeClient
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Jev routing requires typesafe-sdk. Install project requirements first."
                ) from exc
            self._client = TypeSafeClient()
        return self._client

    @staticmethod
    def _action_keys(actions: Sequence[str]) -> dict[str, str]:
        return {f"action_{index}": action for index, action in enumerate(actions)}

    def route(self, step: AgentStep) -> JevDecision:
        actions = list(dict.fromkeys(step.available_actions))
        if not actions:
            return JevDecision(None, 0.0, "strong", None, {}, "no_bounded_actions")
        if len(actions) > 255:
            return JevDecision(
                None, 0.0, "strong", None, {}, "action_space_exceeds_jev_limit"
            )

        key_to_action = self._action_keys(actions)
        response = self._get_client().system_one(
            state={
                "task_instruction": step.instruction,
                "current_observation": step.observation,
                "previous_actions": list(step.action_history),
            },
            questions={
                "next_action": {
                    "type": "choice",
                    "instructions": (
                        "Choose the single best valid next action for completing the task. "
                        "Use only the supplied alternatives."
                    ),
                    "criteria": key_to_action,
                }
            },
            model=self.model,
        )
        answer = response.choices["next_action"]
        selected_action = key_to_action[answer.choice]
        probabilities = {
            key_to_action[key]: float(value)
            for key, value in answer.probabilities.items()
            if key in key_to_action
        }
        confidence = float(answer.confidence)
        use_jev = confidence >= self.confidence_threshold
        return JevDecision(
            action=selected_action if use_jev else None,
            confidence=confidence,
            route="jev" if use_jev else "strong",
            model=getattr(response, "model", self.model),
            probabilities=probabilities,
            reason="confidence_pass" if use_jev else "confidence_below_threshold",
        )


class SelectiveAgentPolicy:
    """Execute Jev decisions and call a strong policy only on escalation."""

    def __init__(
        self,
        router: JevRouter,
        strong_policy: Callable[[AgentStep], str],
    ) -> None:
        self.router = router
        self.strong_policy = strong_policy

    def select_action(self, step: AgentStep) -> tuple[str, JevDecision]:
        decision = self.router.route(step)
        if decision.route == "jev" and decision.action is not None:
            return decision.action, decision
        return self.strong_policy(step), decision
