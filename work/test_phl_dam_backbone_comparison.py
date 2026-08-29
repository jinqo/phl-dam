import unittest

import torch

from phl_dam_backbone_comparison import (
    FastWeightDelta,
    ParameterMatchedDAM,
    active_parameter_count,
    build_model,
    recurrent_state_floats,
)
from phl_dam_stage_b import common_objective, make_batch


class BackboneComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(21)
        self.generator = torch.Generator().manual_seed(22)

    def test_dam_only_has_no_phl_parameters_or_state(self) -> None:
        model = build_model("dam_only")
        names = " ".join(name.lower() for name, _ in model.named_parameters())
        self.assertNotIn("phl_", names)
        state = model.init_state(2, torch.device("cpu"))
        self.assertEqual(state.phl.numel(), 0)
        self.assertEqual(recurrent_state_floats("dam_only", model), 392)

    def test_fast_weight_uses_delta_matrix_and_no_slots_in_state(self) -> None:
        model = FastWeightDelta()
        state = model.init_state(2, torch.device("cpu"))
        self.assertEqual(state.matrix.shape, (2, 24, 24))
        self.assertEqual(recurrent_state_floats("fast_weight", model), 576)
        self.assertFalse(hasattr(state, "keys"))

    def test_variants_are_lease_free_and_finite(self) -> None:
        batch = make_batch(self.generator, 2)
        for kind in ("dam_only", "fast_weight"):
            model = build_model(kind)
            names = " ".join(name.lower() for name, _ in model.named_parameters())
            self.assertNotIn("lease", names)
            logits, diagnostics = model(batch.tokens, return_diagnostics=True)
            self.assertTrue(torch.isfinite(logits).all(), kind)
            self.assertIsNotNone(diagnostics)

    def test_common_objective_reaches_each_memory_path(self) -> None:
        batch = make_batch(self.generator, 4)
        for kind in ("dam_only", "fast_weight"):
            model = build_model(kind)
            logits, diagnostics = model(batch.tokens, return_diagnostics=True)
            loss, _, _ = common_objective(logits, batch)
            budget = (((diagnostics.write_gates.sum(1) - 3.0) / 3.0) ** 2).mean()
            (loss + 0.05 * budget).backward()
            parameters = dict(model.named_parameters())
            for name in (
                "key_projection.weight",
                "value_projection.weight",
                "query_projection.weight",
                "write_gate.weight",
                "read_gate.weight",
                "memory_projection.weight",
            ):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient, f"{kind}:{name}")
                self.assertTrue(torch.isfinite(gradient).all(), f"{kind}:{name}")
                self.assertGreater(gradient.norm().item(), 0.0, f"{kind}:{name}")

    def test_parameter_and_state_accounting(self) -> None:
        phl_dam = build_model("dam_only")
        fast_weight = build_model("fast_weight")
        self.assertGreater(active_parameter_count(phl_dam), 0)
        self.assertGreater(active_parameter_count(fast_weight), 0)
        self.assertNotEqual(
            recurrent_state_floats("dam_only", phl_dam),
            recurrent_state_floats("fast_weight", fast_weight),
        )

    def test_parameter_matched_dam_meets_attribution_contract(self) -> None:
        target = 33034
        base = build_model("dam_only")
        matched = ParameterMatchedDAM()
        matched_count = active_parameter_count(matched)
        self.assertLessEqual(abs(matched_count - target) / target, 0.01)
        self.assertEqual(matched_count, 33098)
        self.assertEqual(recurrent_state_floats("dam_only_matched", matched), 392)
        self.assertEqual(matched.init_state(2, torch.device("cpu")).phl.numel(), 0)
        self.assertFalse(matched.use_phl)
        self.assertEqual(matched.num_slots, 8)
        self.assertEqual((matched.d_key, matched.d_value), (24, 24))
        self.assertEqual(base.write_gate.in_features, matched.write_gate.in_features)
        self.assertEqual(base.read_gate.in_features, matched.read_gate.in_features)
        names = " ".join(name.lower() for name, _ in matched.named_parameters())
        self.assertNotIn("phl_", names)
        self.assertNotIn("lease", names)

    def test_parameter_matched_ffn_receives_gradient(self) -> None:
        model = ParameterMatchedDAM()
        batch = make_batch(self.generator, 4)
        logits, diagnostics = model(batch.tokens, return_diagnostics=True)
        loss, _, _ = common_objective(logits, batch)
        budget = (((diagnostics.write_gates.sum(1) - 3.0) / 3.0) ** 2).mean()
        (loss + 0.05 * budget).backward()
        for name, parameter in model.named_parameters():
            if name.startswith("generic_ffn") and name.endswith("weight"):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(parameter.grad.norm().item(), 0.0, name)


if __name__ == "__main__":
    unittest.main()
