import unittest

import torch

import phl_dam_pressure_task as task
import phl_dam_ntm_pressure as pressure
from phl_dam_004b_lease import EVAL_SEED_OFFSET, pack_batch
from phl_dam_ntm_baseline import active_parameter_count


class NTMPressureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_scale = task.SCALE
        task.set_scale("pressure")

    @classmethod
    def tearDownClass(cls) -> None:
        task.set_scale(cls.previous_scale)

    def test_both_arms_are_parameter_matched_to_phl_dam(self) -> None:
        for arm in pressure.CONTROLLER_WIDTH:
            count = active_parameter_count(pressure.build(arm))
            self.assertLess(
                abs(count - pressure.PHL_DAM_PRESSURE_PARAMETERS)
                / pressure.PHL_DAM_PRESSURE_PARAMETERS, 0.01, arm,
            )

    def test_model_speaks_the_pressure_vocabulary(self) -> None:
        model = pressure.build("ntm_dnc_factorized")
        episode = task.generate_episode(500_000, 0, 16, "canonical")
        batch = pack_batch([episode], torch.device("cpu"))
        with torch.no_grad():
            logits = model(batch.tokens)
        self.assertEqual(logits.shape, (1, task.SEQUENCE_LENGTH, task.VOCAB_SIZE))

    def test_evaluation_uses_the_004d_held_out_stream(self) -> None:
        """Same seeds as evaluate_arm, so recall is scored on identical episodes."""
        a = task.generate_episode(3 + EVAL_SEED_OFFSET, 5, 8, "canonical")
        b = task.generate_episode(3 + EVAL_SEED_OFFSET, 5, 8, "canonical")
        self.assertTrue(torch.equal(a.tokens, b.tokens))
        self.assertEqual(EVAL_SEED_OFFSET, 20_000)

    def test_a_short_run_is_finite_and_reports_004d_fields(self) -> None:
        summary = pressure.run("ntm_dnc_factorized", writes=8, seed=0, steps=2,
                               batch_size=2, eval_episodes=2, learning_rate=2e-3)
        self.assertTrue(summary["finite"])
        for field in ("breakthrough_step", "final_recall_ce", "training_history"):
            self.assertIn(field, summary)
        self.assertEqual(summary["configuration"]["sequence_length"], 456)
        self.assertGreater(summary["metrics"]["queries"], 0)
        self.assertLessEqual(summary["metrics"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
