import unittest

import torch

from phl_dam_stage_b import SEQUENCE_LENGTH, VOCAB_SIZE, common_objective, make_batch
import phl_dam_ntm_baseline as ntm


class NTMBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ntm.NTMBaseline()
        self.generator = torch.Generator().manual_seed(11)

    def test_parameters_are_matched_to_the_other_models(self) -> None:
        count = ntm.active_parameter_count(self.model)
        self.assertLess(abs(count - 33_034) / 33_034, 0.01)

    def test_it_is_not_handicapped_on_state(self) -> None:
        """A baseline must not lose by being given less memory than PHL-DAM."""
        self.assertLessEqual(ntm.recurrent_state_floats(self.model), 456)
        self.assertGreater(ntm.recurrent_state_floats(self.model), 300)

    def test_forward_is_causal(self) -> None:
        batch = make_batch(self.generator, 2)
        cut = 100
        mutated = batch.tokens.clone()
        mutated[:, cut:] = 4
        with torch.no_grad():
            original = self.model(batch.tokens)
            changed = self.model(mutated)
        self.assertTrue(torch.equal(original[:, :cut], changed[:, :cut]))

    def test_forward_shape_and_finiteness(self) -> None:
        batch = make_batch(self.generator, 2)
        logits = self.model(batch.tokens)
        self.assertEqual(logits.shape, (2, SEQUENCE_LENGTH, VOCAB_SIZE))
        self.assertTrue(torch.isfinite(logits).all())

    def test_content_weighting_is_a_distribution_over_slots(self) -> None:
        memory = torch.randn(3, self.model.slots, self.model.width)
        key = torch.randn(3, self.model.width)
        strength = torch.full((3, 1), 2.0)
        weighting = self.model.memory.content_weighting(memory, key, strength)
        self.assertEqual(weighting.shape, (3, self.model.slots))
        self.assertTrue(torch.allclose(weighting.sum(-1), torch.ones(3), atol=1e-5))

    def test_allocation_targets_the_least_used_slot(self) -> None:
        """The DNC's usage-based allocation, which PHL-DAM's occupancy term mimics."""
        usage = torch.tensor([[0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
        allocation = self.model.memory.allocation_weighting(usage)
        self.assertEqual(int(allocation.argmax()), 1, "slot 1 is least used")
        self.assertTrue(bool((allocation >= 0).all()))
        self.assertLessEqual(float(allocation.sum()), 1.0 + 1e-5)

    def test_allocation_is_differentiable_in_usage(self) -> None:
        """A hard one-hot here starves the usage path and the free gate."""
        usage = torch.tensor([[0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]],
                             requires_grad=True)
        self.model.memory.allocation_weighting(usage).sum().backward()
        self.assertIsNotNone(usage.grad)
        self.assertGreater(usage.grad.abs().sum().item(), 0.0)

    def test_an_empty_slot_receives_the_most_allocation(self) -> None:
        usage = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        allocation = self.model.memory.allocation_weighting(usage)
        self.assertAlmostEqual(float(allocation[0, 2]), 1.0, places=5)

    def test_memory_ablation_changes_the_output(self) -> None:
        batch = make_batch(self.generator, 2)
        with torch.no_grad():
            full = self.model(batch.tokens)
            ablated = self.model(batch.tokens, disable_memory=True)
        self.assertFalse(torch.equal(full, ablated))

    def test_gradients_reach_every_memory_interface(self) -> None:
        batch = make_batch(self.generator, 2)
        loss, _, _ = common_objective(self.model(batch.tokens), batch)
        loss.backward()
        for name in ("write_key", "erase_vector", "add_vector", "read_key",
                     "allocation_gate", "write_gate", "free_gate"):
            parameter = getattr(self.model.memory, name).weight
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_no_phl_or_lease_machinery(self) -> None:
        names = " ".join(n.lower() for n, _ in self.model.named_parameters())
        for forbidden in ("phl", "lease", "horizon"):
            self.assertNotIn(forbidden, names)


class FactorizedNTMTests(unittest.TestCase):
    """The control arm that gives the DNC PHL-DAM's key/value role wiring."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ntm.build_model(factorized=True)
        self.generator = torch.Generator().manual_seed(11)

    def test_parameters_are_matched_to_phl_dam(self) -> None:
        count = ntm.active_parameter_count(self.model)
        self.assertLess(abs(count - 33_034) / 33_034, 0.01)

    def test_state_matches_phl_dam_banks(self) -> None:
        # 8 slots x (24 key + 24 value) + 8 usage = PHL-DAM's 8 x (24+24+1).
        self.assertEqual(ntm.recurrent_state_floats(self.model), 392)
        self.assertLessEqual(ntm.recurrent_state_floats(self.model), 456)

    def test_forward_is_causal(self) -> None:
        batch = make_batch(self.generator, 2)
        mutated = batch.tokens.clone()
        mutated[:, 100:] = 4
        with torch.no_grad():
            original = self.model(batch.tokens)
            changed = self.model(mutated)
        self.assertTrue(torch.equal(original[:, :100], changed[:, :100]))

    def test_stored_key_depends_only_on_previous_token(self) -> None:
        memory = self.model.memory
        d = self.model.d_model
        controller = torch.randn(1, self.model.controller_width)
        previous = torch.randn(1, d)
        state = torch.zeros(1, self.model.slots, self.model.row_width)
        usage = torch.zeros(1, self.model.slots)
        with torch.no_grad():
            a, _, _ = memory(controller, state, usage, previous, torch.randn(1, d))
            b, _, _ = memory(controller, state, usage, previous, torch.randn(1, d))
        key_width = memory.key_width
        self.assertTrue(torch.allclose(a[..., :key_width], b[..., :key_width]))
        self.assertFalse(torch.allclose(a[..., key_width:], b[..., key_width:]))

    def test_read_returns_the_value_bank_addressed_by_the_key_bank(self) -> None:
        memory = self.model.memory
        key_width = memory.key_width
        state = torch.zeros(1, self.model.slots, self.model.row_width)
        state[0, 3, :key_width] = 5.0 * memory.read_key.bias.detach()
        state[0, 3, key_width:] = 7.0
        controller = torch.zeros(1, self.model.controller_width)
        with torch.no_grad():
            memory.read_strength.bias.fill_(50.0)
            memory.write_gate.bias.fill_(-50.0)
            _, _, read = memory(
                controller, state, torch.zeros(1, self.model.slots),
                torch.zeros(1, self.model.d_model), torch.zeros(1, self.model.d_model),
            )
        self.assertEqual(read.shape, (1, memory.value_width))
        self.assertTrue(torch.allclose(read, torch.full_like(read, 7.0), atol=1e-3))

    def test_gradients_reach_every_memory_interface(self) -> None:
        batch = make_batch(self.generator, 2)
        loss, _, _ = common_objective(self.model(batch.tokens), batch)
        loss.backward()
        for name in ("write_key", "erase_vector", "add_vector", "read_key",
                     "allocation_gate", "write_gate", "free_gate"):
            parameter = getattr(self.model.memory, name).weight
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_memory_ablation_changes_the_output(self) -> None:
        batch = make_batch(self.generator, 2)
        with torch.no_grad():
            self.assertFalse(torch.equal(
                self.model(batch.tokens), self.model(batch.tokens, disable_memory=True)
            ))


if __name__ == "__main__":
    unittest.main()
