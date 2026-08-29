"""Tests for the read-back consistency signal.

Motivation, from the PHL-DAM-004G ladder: the fix rescued W=16 (1/5 -> 3/5) but
W=20 stayed at 0/5, and the failing runs all show NEGATIVE write selectivity -
the gate ends up firing more off-binding than on-binding. The write path's only
gradient arrives 32-256 tokens later and must survive eviction first, so under
high write load nothing survives early enough to teach the gate anything.

Read-back gives that path an immediate, self-supervised signal. These tests pin
the properties that make it legitimate: it uses no future information, it only
asks for consistency where the model chose to write, and it actually reaches the
write and addressing parameters.
"""

import unittest

import torch

import phl_dam_pressure_task as task
import phl_dam_004b_lease as lease


class ReadbackTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("compact")
        lease.seed_everything(0)
        self.model = lease.PHLDAMLease(arm="content_only")
        self.batch = lease.pack_batch(
            [task.generate_episode(0, i, 12, "canonical") for i in range(4)],
            torch.device("cpu"),
        )

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_readback_is_finite_and_non_negative(self) -> None:
        self.model(self.batch.tokens, arm="content_only")
        error = lease.readback_objective(self.model)
        self.assertTrue(torch.isfinite(error))
        self.assertGreaterEqual(float(error), 0.0)

    def test_it_requires_a_forward_pass_first(self) -> None:
        fresh = lease.PHLDAMLease(arm="content_only")
        with self.assertRaises(RuntimeError):
            lease.readback_objective(fresh)

    def test_gradient_reaches_the_write_and_addressing_paths(self) -> None:
        """The whole point: give the write gate a signal that skips the delay."""
        self.model(self.batch.tokens, arm="content_only")
        lease.readback_objective(self.model).backward()
        for name in ("write_gate", "key_projection", "value_projection"):
            parameter = dict(self.model.named_parameters())[f"{name}.weight"]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_it_uses_no_labels_and_no_future_information(self) -> None:
        """Overwriting every target and future-use table must not change it."""
        self.model(self.batch.tokens, arm="content_only")
        before = float(lease.readback_objective(self.model).detach())
        self.batch.query_target_tokens.fill_(task.VALUE_START)
        self.batch.key_use_positions.fill_(lease.INFINITY)
        self.batch.timing_label.fill_(lease.TIMING_IGNORE_INDEX)
        self.model(self.batch.tokens, arm="content_only")
        after = float(lease.readback_objective(self.model).detach())
        self.assertEqual(before, after)

    def test_a_closed_write_gate_produces_no_readback_demand(self) -> None:
        """Weighted by the gate: it only asks for consistency where we wrote."""
        quiet = lease.PHLDAMLease(arm="content_only")
        with torch.no_grad():
            quiet.write_gate.bias.fill_(-30.0)
            quiet.write_gate.weight.zero_()
        quiet(self.batch.tokens, arm="content_only")
        self.assertLess(float(lease.readback_objective(quiet).detach()), 1e-6)

    def test_an_open_gate_produces_a_larger_demand_than_a_closed_one(self) -> None:
        errors = {}
        for label, bias in (("closed", -30.0), ("open", 5.0)):
            model = lease.PHLDAMLease(arm="content_only")
            with torch.no_grad():
                model.write_gate.bias.fill_(bias)
                model.write_gate.weight.zero_()
            model(self.batch.tokens, arm="content_only")
            errors[label] = float(lease.readback_objective(model).detach())
        self.assertGreater(errors["open"], errors["closed"])

    def test_readback_does_not_change_the_forward_output(self) -> None:
        """It is a loss term only; logits must be untouched."""
        lease.seed_everything(3)
        model = lease.PHLDAMLease(arm="content_only")
        with torch.no_grad():
            first, _ = model(self.batch.tokens, arm="content_only")
            _ = lease.readback_objective(model)
            second, _ = model(self.batch.tokens, arm="content_only")
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
