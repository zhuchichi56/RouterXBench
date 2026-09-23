import unittest
from types import SimpleNamespace

from src.agentbench.jev_router import AgentStep, JevRouter, SelectiveAgentPolicy
from src.agentbench.webshop_actions import action_to_tool_call, bounded_click_actions


class FakeClient:
    def __init__(self, choice="action_1", confidence=0.8):
        self.choice = choice
        self.confidence = confidence

    def system_one(self, state, questions, *, model=None):
        probabilities = {key: 0.1 for key in questions["next_action"]["criteria"]}
        probabilities[self.choice] = self.confidence
        answer = SimpleNamespace(
            choice=self.choice,
            confidence=self.confidence,
            probabilities=probabilities,
        )
        return SimpleNamespace(choices={"next_action": answer}, model=model)


def make_step():
    return AgentStep(
        task_id="webshop-0",
        step_id=0,
        instruction="Buy the requested product.",
        observation="Search page",
        available_actions=["search[blue shirt]", "click[back]"],
    )


class JevRouterTest(unittest.TestCase):
    def test_webshop_search_is_delegated_but_clicks_are_bounded(self):
        available = {"has_search_bar": True, "clickables": ["search", "B0123", "Next >"]}
        self.assertEqual(
            bounded_click_actions(available),
            ["click[B0123]", "click[Next >]"],
        )

    def test_jev_action_can_be_inserted_as_valid_tool_history(self):
        call = action_to_tool_call("click[buy now]", "jev-step-4")
        self.assertEqual(call["function"]["name"], "click_action")
        self.assertIn("buy now", call["function"]["arguments"])

    def test_confident_choice_executes_jev_action(self):
        decision = JevRouter(client=FakeClient(), confidence_threshold=0.5).route(make_step())
        self.assertEqual(decision.route, "jev")
        self.assertEqual(decision.action, "click[back]")

    def test_low_confidence_escalates_without_executing_choice(self):
        decision = JevRouter(
            client=FakeClient(confidence=0.4), confidence_threshold=0.5
        ).route(make_step())
        self.assertEqual(decision.route, "strong")
        self.assertIsNone(decision.action)
        self.assertEqual(decision.reason, "confidence_below_threshold")

    def test_empty_action_space_escalates_without_api_call(self):
        step = AgentStep("alfworld-0", 0, "task", "observation", [])
        decision = JevRouter(client=FakeClient()).route(step)
        self.assertEqual(decision.route, "strong")
        self.assertEqual(decision.reason, "no_bounded_actions")

    def test_selective_policy_calls_strong_agent_only_on_escalation(self):
        strong_calls = []

        def strong_policy(step):
            strong_calls.append(step.task_id)
            return "search[fallback]"

        policy = SelectiveAgentPolicy(
            JevRouter(client=FakeClient(confidence=0.4), confidence_threshold=0.5),
            strong_policy,
        )
        action, decision = policy.select_action(make_step())
        self.assertEqual(action, "search[fallback]")
        self.assertEqual(decision.route, "strong")
        self.assertEqual(strong_calls, ["webshop-0"])


if __name__ == "__main__":
    unittest.main()
