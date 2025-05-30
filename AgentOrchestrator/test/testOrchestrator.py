import unittest
from AgentFramework.core.AgentScheduler import AgentScheduler
from AgentOrchestrator.test.NumericOrchestrator import NumericRequest, NumericResult, NumericOrchestrator


class TestNumericOrchestrator(unittest.TestCase):
    """
    Unit tests for the `NumericOrchestrator`.

    This orchestrator demonstrates conditional branching: it first runs a `DoubleAgent` that doubles
    the incoming number. If the result is below a configurable threshold (default: 10), it dynamically
    triggers an additional `IncrementAgent` to increase the result by 1.

    The orchestrator is designed to block until the entire internal process is finished and return
    a clean result, making it behave like a normal `ConnectedAgent` in the larger system.

    Each test checks a specific control path:

    ┌──────────────────────────────┬─────────────────────────────────────────────────────────────┐
    │ Input                        │ Execution Path                                               │
    ├──────────────────────────────┼─────────────────────────────────────────────────────────────┤
    │ 6                            │ DoubleAgent: 6×2=12 → >= 10 → result returned directly       │
    │ 3                            │ DoubleAgent: 3×2=6 → < 10 → IncrementAgent: 6+1=7            │
    │ -1                           │ Not implemented here, but could be used for failure testing  │
    └──────────────────────────────┴─────────────────────────────────────────────────────────────┘

    To use:
        - Make sure `NumericOrchestrator` is correctly registered in your orchestrator module.
        - Run this file directly or as part of your test suite.

    Notes:
        - All tests create a fresh instance of `AgentScheduler` and `NumericOrchestrator`.
        - Results are retrieved using `get_one_output()` which simulates final delivery.
    """

    def _run(self, n: int) -> NumericResult:
        """
        Helper method to set up and execute the orchestrator for a given integer input.

        It:
          1. Instantiates the orchestrator
          2. Adds it to a new scheduler
          3. Feeds it the test input
          4. Runs the scheduler until all agents are done
          5. Returns the orchestrator’s final output

        Args:
            n (int): Input value to pass into the orchestrator

        Returns:
            NumericResult: Final output message containing the result, success flag, and any error
        """
        orch = NumericOrchestrator()
        top = AgentScheduler(uuid="test")
        top.add_agent(orch)
        orch.feed(NumericRequest(value=n))
        top.step_all()
        return orch.pop_one_output()

    def test_threshold_met_directly(self):
        """
        Test case where the doubled value is >= threshold.
        Should not invoke IncrementAgent.
        Input: 6 → 6×2 = 12 → result is 12
        """
        res = self._run(50)  # 50+100=150 *2 = 300  → meets threshold
        self.assertTrue(res.success)
        self.assertEqual(res.value, 300)

    def test_threshold_not_met(self):
        """
        Test case where the doubled value is < threshold.
        Should dynamically invoke IncrementAgent.
        Input: 3 → 3×2 = 6 → 6+1 = 7 → result is 7
        """
        res = self._run(20)  # 20+100=120 *2=240 <300 so +200+1 = 441
        self.assertTrue(res.success)
        self.assertEqual(res.value, 441)


