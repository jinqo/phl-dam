import unittest

import torch

from phl_dam_stage_b import common_objective, make_batch
from phl_dam_transformer_rematch import (
    CausalTransformer,
    active_parameter_count,
    kv_cache_floats,
)


class TransformerRematchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.generator = torch.Generator().manual_seed(32)
        self.model = CausalTransformer()

    def test_parameter_match_and_state_accounting(self) -> None:
        target = 33034
        count = active_parameter_count(self.model)
        self.assertEqual(count, 33074)
        self.assertLessEqual(abs(count - target) / target, 0.01)
        self.assertEqual(kv_cache_floats(self.model, 176), 16896)

    def test_causal_mask_blocks_future_tokens(self) -> None:
        self.model.eval()
        batch = make_batch(self.generator, 1)
        original = self.model(batch.tokens)
        changed = batch.tokens.clone()
        changed[:, 100] = (changed[:, 100] + 7) % original.shape[-1]
        mutated = self.model(changed)
        self.assertTrue(torch.allclose(original[:, :100], mutated[:, :100], atol=1e-6))

    def test_attention_ablation_changes_output(self) -> None:
        batch = make_batch(self.generator, 2)
        normal = self.model(batch.tokens)
        disabled = self.model(batch.tokens, disable_attention=True)
        self.assertEqual(normal.shape, disabled.shape)
        self.assertFalse(torch.equal(normal, disabled))

    def test_predictive_objective_reaches_attention_and_ffn(self) -> None:
        batch = make_batch(self.generator, 4)
        logits = self.model(batch.tokens)
        loss, all_ce, recall_ce = common_objective(logits, batch)
        self.assertTrue(torch.isfinite(torch.stack([loss, all_ce, recall_ce])).all())
        loss.backward()
        parameters = dict(self.model.named_parameters())
        for name in (
            "attention.in_proj_weight",
            "attention.out_proj.weight",
            "ffn.0.weight",
            "ffn.2.weight",
            "output.weight",
        ):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0.0, name)

    def test_no_lease_or_promotion_parameters(self) -> None:
        names = " ".join(name.lower() for name, _ in self.model.named_parameters())
        self.assertNotIn("lease", names)
        self.assertNotIn("promotion", names)


if __name__ == "__main__":
    unittest.main()
