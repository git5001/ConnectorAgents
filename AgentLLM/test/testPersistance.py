# tests/test_scheduler_persistence.py
import unittest
from pathlib import Path
import tempfile

from AgentFramework.core.AgentScheduler import AgentScheduler

from AgentLLM.test.TestModels import EchoAgent, Msg  # Msg is used both for messages and global_state


def build_tree():
    a1 = EchoAgent(uuid="a1")
    a2 = EchoAgent(uuid="a2")
    a1.connectTo(a2)

    inner = AgentScheduler(uuid="inner")
    inner.agents.extend([a1, a2])

    outer = AgentScheduler(uuid="outer")
    outer.agents.append(inner)

    return outer, inner, a1, a2


class TestSchedulerPersistence(unittest.TestCase):

    def test_scheduler_roundtrip(self):
        print("Start test")

        # Use temp dir manually to work around PyCharm runner
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "run"

            # 1) build and mutate
            outer, inner, a1, a2 = build_tree()

            # --- set up global_state and is_active to test persistence ---
            outer._global_state = Msg(text="GLOBAL")    # test a BaseModel as global_state
            a2.is_active = False                        # test is_active persists

            # feed and run enough steps so both agents process
            a1.feed(Msg(text="hello"))
            outer.step()   # triggers a1
            outer.step()   # triggers a2

            # 2) snapshot and save
            snapshot_before = outer.save_state()
            outer.is_active = False
            outer.save_scheduler(save_dir)

            # 3) recreate and reload
            new_outer, new_inner, new_a1, new_a2 = build_tree()
            new_outer.load_scheduler(save_dir)
            snapshot_after = new_outer.save_state()

            # 4) verify raw snapshot keys match (except binary blobs)
            self.assertFalse(new_outer.is_active, "Scheduler is_active flag did not persist")
            self.assertSetEqual(set(snapshot_before.keys()), set(snapshot_after.keys()),  "Top-level keys changed after reload")

            # 5) semantic checks
            restored_inner = new_outer.agents[0]
            restored_a1 = restored_inner.agents[0]
            restored_a2 = restored_inner.agents[1]

            # --- scheduler global_state ----------------------------------
            self.assertTrue(isinstance(new_outer._global_state, Msg))
            self.assertEqual(outer._global_state, new_outer._global_state,
                             "Global state did not persist")

            # --- is_active flags -----------------------------------------
            self.assertEqual(a2.is_active, restored_a2.is_active,
                             "Agent.is_active flag did not persist")
            # a1 was left True by default
            self.assertTrue(restored_a1.is_active)

            # --- scheduler counters --------------------------------------
            self.assertEqual(outer.state.agent_idx, new_outer.state.agent_idx)
            self.assertEqual(inner.state.step_counter, restored_inner.state.step_counter)

            # --- agent runtime state (CounterState) ----------------------
            # a1 ran once, so it must have state
            self.assertIsNotNone(a1._state)
            self.assertIsNotNone(restored_a1._state)
            self.assertEqual(a1._state.count, restored_a1._state.count)

            # a2 was marked inactive before running, so its state should stay None
            self.assertIsNone(a2._state)
            self.assertIsNone(restored_a2._state)

            # --- queue contents (deep equality) --------------------------
            # Compare each queue entry’s (parents, message) pair
            for (p1, m1), (p2, m2) in zip(a1.input_port.queue, restored_a1.input_port.queue):
                self.assertEqual(p1, p2)
                self.assertEqual(m1, m2)
            for (p1, m1), (p2, m2) in zip(a2.input_port.queue, restored_a2.input_port.queue):
                self.assertEqual(p1, p2)
                self.assertEqual(m1, m2)

            # --- final outputs consistency -------------------------------
            self.assertEqual(a1.get_final_outputs(), restored_a1.get_final_outputs())
            self.assertEqual(a2.get_final_outputs(), restored_a2.get_final_outputs())


if __name__ == "__main__":
    unittest.main()
