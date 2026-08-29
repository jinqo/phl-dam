import unittest

import torch

from phl_dam_stage_b import (
    NUM_VALUES,
    PHLDAM,
    QUERY,
    VALUE_START,
    WRITE,
    common_objective,
    compose_training_loss,
    make_batch,
)


class StageBTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.generator = torch.Generator().manual_seed(10)
        self.model = PHLDAM()

    def test_protocol_has_three_writes_queries_and_random_values(self) -> None:
        batch = make_batch(self.generator, 64)
        self.assertTrue((batch.tokens == WRITE).sum(dim=1).eq(3).all())
        self.assertTrue((batch.tokens == QUERY).sum(dim=1).eq(3).all())
        self.assertTrue(batch.recall_mask.sum(dim=1).eq(3).all())
        targets = batch.tokens[batch.recall_mask]
        self.assertTrue((targets >= VALUE_START).all())
        self.assertTrue((targets < VALUE_START + NUM_VALUES).all())
        self.assertGreater(torch.unique(targets).numel(), 5)

    def test_all_delays_are_in_preregistered_bins(self) -> None:
        batch = make_batch(self.generator, 256)
        covered = (
            ((batch.delays >= 29) & (batch.delays <= 63))
            | ((batch.delays >= 64) & (batch.delays <= 95))
            | ((batch.delays >= 96) & (batch.delays <= 169))
        )
        self.assertTrue(covered.all())
        for low, high in ((29, 63), (64, 95), (96, 169)):
            self.assertTrue(((batch.delays >= low) & (batch.delays <= high)).any())

    def test_forward_receives_tokens_only_and_is_finite(self) -> None:
        batch = make_batch(self.generator, 2)
        logits, diagnostics = self.model(batch.tokens, return_diagnostics=True)
        self.assertEqual(logits.shape, (*batch.tokens.shape, logits.shape[-1]))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNotNone(diagnostics)
        self.assertTrue(torch.isfinite(diagnostics.final_occupancy).all())

    def test_no_temporal_lease_parameters_or_state(self) -> None:
        names = " ".join(name.lower() for name, _ in self.model.named_parameters())
        self.assertNotIn("lease", names)
        state = self.model.init_state(2, torch.device("cpu"))
        self.assertFalse(hasattr(state.dam, "leases"))

    def test_common_objective_and_controller_gradients(self) -> None:
        batch = make_batch(self.generator, 4)
        logits, _ = self.model(batch.tokens)
        loss, all_ce, recall_ce = common_objective(logits, batch)
        self.assertTrue(torch.isfinite(torch.stack([loss, all_ce, recall_ce])).all())
        loss.backward()
        parameters = dict(self.model.named_parameters())
        for name in (
            "key_projection.weight",
            "value_projection.weight",
            "query_projection.weight",
            "write_gate.weight",
            "read_gate.weight",
            "memory_projection.weight",
        ):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0.0, name)

    def test_zero_write_budget_weight_removes_penalty(self) -> None:
        common_loss = torch.tensor(2.5, requires_grad=True)
        write_budget = torch.tensor(7.0, requires_grad=True)
        loss = compose_training_loss(common_loss, write_budget, 0.0)
        self.assertEqual(loss.item(), common_loss.item())
        loss.backward()
        self.assertEqual(common_loss.grad.item(), 1.0)
        self.assertEqual(write_budget.grad.item(), 0.0)

    def test_retrieval_ablation_preserves_causal_shape(self) -> None:
        batch = make_batch(self.generator, 2)
        normal, _ = self.model(batch.tokens)
        disabled, _ = self.model(batch.tokens, disable_retrieval=True)
        self.assertEqual(normal.shape, disabled.shape)
        self.assertFalse(torch.equal(normal, disabled))


if __name__ == "__main__":
    unittest.main()
