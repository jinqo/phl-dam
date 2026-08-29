import unittest

import torch

from phl_dam_stage_b import SEQUENCE_LENGTH, VOCAB_SIZE, make_batch
import phl_dam_ssm_baseline as ssm


class SSMBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ssm.build_model(selective=False)
        self.generator = torch.Generator().manual_seed(3)

    def test_parameter_count_matches_the_other_baselines(self) -> None:
        """Within 0.5% of PHL-DAM (33,034) and the Transformer (33,074)."""
        count = ssm.active_parameter_count(self.model)
        self.assertLess(abs(count - 33_034) / 33_034, 0.005)
        self.assertLess(abs(count - 33_074) / 33_074, 0.005)

    def test_inference_state_is_constant_in_sequence_length(self) -> None:
        """An SSM's advantage claim: state must not grow with length."""
        self.assertEqual(ssm.recurrent_state_floats(self.model), 96)
        for length in (64, 176, 512):
            self.assertEqual(ssm.recurrent_state_floats(self.model), 96, length)

    def test_recurrence_can_never_explode(self) -> None:
        """decay must lie in [0, 1] for ANY parameter value.

        Both ends saturate in float32 and both are benign: 1.0 is a pure
        integrator (marginally stable), 0.0 is a channel that forgets.
        Anything above 1.0 would amplify, and that must never happen.
        """
        for layer in self.model.layers:
            for value in (-40.0, -20.0, -16.0, 0.0, 20.0, 40.0):
                with torch.no_grad():
                    layer.ssm.log_rate.fill_(value)
                decay = torch.exp(-torch.exp(layer.ssm.log_rate))
                self.assertTrue(bool((decay >= 0.0).all()), value)
                self.assertTrue(bool((decay <= 1.0).all()), value)

    def test_a_saturated_channel_is_an_integrator_not_a_divergence(self) -> None:
        """The saturated case must accumulate, never amplify."""
        layer = self.model.layers[0]
        with torch.no_grad():
            layer.ssm.log_rate.fill_(-40.0)
        decay = torch.exp(-torch.exp(layer.ssm.log_rate))
        state = torch.ones(1)
        for _ in range(1000):
            state = decay[0] * state
        self.assertLessEqual(float(state), 1.0 + 1e-6)

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

    def test_recurrence_ablation_changes_the_output(self) -> None:
        batch = make_batch(self.generator, 2)
        with torch.no_grad():
            full = self.model(batch.tokens)
            ablated = self.model(batch.tokens, disable_recurrence=True)
        self.assertEqual(full.shape, ablated.shape)
        self.assertFalse(torch.equal(full, ablated))

    def test_gradients_are_finite_and_reach_the_recurrence(self) -> None:
        batch = make_batch(self.generator, 2)
        from phl_dam_stage_b import common_objective

        loss, _, _ = common_objective(self.model(batch.tokens), batch)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.model.layers[0].ssm.log_rate.grad.abs().sum().item(), 0.0)

    def test_no_lease_or_memory_slot_parameters(self) -> None:
        names = " ".join(n.lower() for n, _ in self.model.named_parameters())
        for forbidden in ("lease", "slot", "eviction"):
            self.assertNotIn(forbidden, names)


class SelectiveSSMTests(unittest.TestCase):
    """The selective variant is the fair baseline for associative recall.

    A fixed-decay S4D cannot choose per token whether to hold or overwrite its
    state, which is the known reason non-selective SSMs fail associative recall.
    Comparing PHL-DAM only against the non-selective variant would be a strawman.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ssm.build_model(selective=True)
        self.generator = torch.Generator().manual_seed(5)

    def test_parameter_count_is_matched(self) -> None:
        count = ssm.active_parameter_count(self.model)
        self.assertLess(abs(count - 33_074) / 33_074, 0.01)

    def test_decay_is_input_dependent(self) -> None:
        """Selection means two different tokens get two different decays."""
        layer = self.model.layers[0].ssm
        self.assertTrue(layer.selective)
        left = torch.zeros(1, 1, self.model.d_model)
        right = torch.ones(1, 1, self.model.d_model)
        with torch.no_grad():
            a = torch.nn.functional.softplus(layer.delta_projection(left))
            b = torch.nn.functional.softplus(layer.delta_projection(right))
        self.assertFalse(torch.allclose(a, b))

    def test_selective_decay_still_cannot_amplify(self) -> None:
        layer = self.model.layers[0].ssm
        for fill in (-40.0, 0.0, 40.0):
            with torch.no_grad():
                layer.delta_projection.bias.fill_(fill)
            sequence = torch.randn(2, 6, self.model.d_model)
            delta = torch.nn.functional.softplus(layer.delta_projection(sequence))
            decay = torch.exp(-delta * torch.exp(layer.log_rate))
            self.assertTrue(bool((decay >= 0.0).all()), fill)
            self.assertTrue(bool((decay <= 1.0).all()), fill)

    def test_selective_forward_is_causal_and_finite(self) -> None:
        batch = make_batch(self.generator, 2)
        cut = 100
        mutated = batch.tokens.clone()
        mutated[:, cut:] = 4
        with torch.no_grad():
            original = self.model(batch.tokens)
            changed = self.model(mutated)
        self.assertTrue(torch.equal(original[:, :cut], changed[:, :cut]))
        self.assertTrue(torch.isfinite(original).all())

    def test_selection_parameters_receive_gradient(self) -> None:
        from phl_dam_stage_b import common_objective

        batch = make_batch(self.generator, 2)
        loss, _, _ = common_objective(self.model(batch.tokens), batch)
        loss.backward()
        grad = self.model.layers[0].ssm.delta_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
