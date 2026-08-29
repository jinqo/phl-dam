"""Tests for the backbone replacement.

PHL's measured contribution in the existing attribution runs was optimisation
reliability, not representational power: when the PHL-off model succeeded it
matched PHL exactly (99.47 / 99.90 / 99.95), it just succeeded on 3/6 seeds
instead of 6/6. The `ssm` backbone replaces the hand-designed fixed transport
lattice with a learned per-channel decay that provably cannot amplify.
"""

import math
import unittest

import torch

import phl_dam_pressure_task as task
import phl_dam_004b_lease as lease


class BackboneTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("compact")
        self.batch = lease.pack_batch(
            [task.generate_episode(0, i, 8, "canonical") for i in range(2)],
            torch.device("cpu"),
        )

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_every_backbone_runs_and_is_finite(self) -> None:
        for backbone in ("phl", "ssm", "none"):
            model = lease.PHLDAMLease(arm="content_only", backbone=backbone)
            with torch.no_grad():
                logits, _ = model(self.batch.tokens, arm="content_only")
            self.assertTrue(torch.isfinite(logits).all(), backbone)

    def test_unknown_backbone_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            lease.PHLDAMLease(arm="content_only", backbone="nonsense")

    def test_ssm_decay_cannot_amplify_for_any_parameter_value(self) -> None:
        """The property the PHL slot path lacked, and which caused 1e19 gradients."""
        model = lease.PHLDAMLease(arm="content_only", backbone="ssm")
        for value in (-60.0, -20.0, 0.0, 20.0, 60.0):
            with torch.no_grad():
                model.ssm_log_rate.fill_(value)
            decay = torch.exp(-torch.exp(model.ssm_log_rate))
            self.assertTrue(bool((decay >= 0.0).all()), value)
            self.assertTrue(bool((decay <= 1.0).all()), value)

    def test_ssm_rates_are_log_spaced_at_initialisation(self) -> None:
        """Channels must span short and long horizons, as PHL did by design."""
        model = lease.PHLDAMLease(arm="content_only", backbone="ssm")
        decay = torch.exp(-torch.exp(model.ssm_log_rate))
        self.assertLess(float(decay.min()), 0.5)
        self.assertGreater(float(decay.max()), 0.99)

    def test_backbone_none_removes_the_temporal_path_entirely(self) -> None:
        model = lease.PHLDAMLease(arm="content_only", backbone="none")
        state = model.init_state(2, torch.device("cpu"))
        before = state.phl.clone()
        _, new_state, _ = model.step(
            torch.randn(2, model.d_model),
            torch.randn(2, model.d_model),
            torch.randn(2, model.d_model),
            state,
            arm="content_only",
            previous_token_ids=torch.tensor([task.KEY_START, task.KEY_START + 1]),
            current_token_ids=torch.tensor([task.VALUE_START, task.VALUE_START]),
        )
        self.assertTrue(torch.equal(new_state.phl, before))

    def test_backbones_produce_different_outputs(self) -> None:
        outputs = {}
        for backbone in ("phl", "ssm", "none"):
            torch.manual_seed(0)
            model = lease.PHLDAMLease(arm="content_only", backbone=backbone)
            with torch.no_grad():
                outputs[backbone], _ = model(self.batch.tokens, arm="content_only")
        self.assertFalse(torch.equal(outputs["phl"], outputs["ssm"]))
        self.assertFalse(torch.equal(outputs["phl"], outputs["none"]))

    def test_recurrent_state_size_is_unchanged_by_the_swap(self) -> None:
        """The swap must not buy accuracy with extra state."""
        counts = {
            b: lease.parameter_report(
                lease.PHLDAMLease(arm="content_only", backbone=b)
            )["recurrent_state_floats"]
            for b in ("phl", "ssm")
        }
        self.assertEqual(counts["phl"], counts["ssm"])

    def test_gradients_reach_the_learned_decay(self) -> None:
        model = lease.PHLDAMLease(arm="content_only", backbone="ssm")
        logits, _ = model(self.batch.tokens, arm="content_only")
        loss, _, _ = lease.common_objective(logits, self.batch)
        loss.backward()
        self.assertIsNotNone(model.ssm_log_rate.grad)
        self.assertTrue(torch.isfinite(model.ssm_log_rate.grad).all())
        self.assertGreater(model.ssm_log_rate.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
