import unittest

import torch

from phl_dam_stage_b import PHLDAM, SEQUENCE_LENGTH, VOCAB_SIZE, common_objective, make_batch
from phl_dam_v2 import PHLDAMv2, active_parameter_count
import phl_dam_learning_curve as curve


class PHLDAMv2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = make_batch(torch.Generator().manual_seed(1), 4)

    def pair(self, **options):
        torch.manual_seed(0)
        reference = PHLDAM()
        torch.manual_seed(0)
        return reference, PHLDAMv2(**options)

    def test_default_switches_are_the_stage_b_model(self) -> None:
        reference, model = self.pair()
        for a, b in zip(reference.parameters(), model.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(reference(self.batch.tokens)[0],
                                    model(self.batch.tokens)[0]))

    def test_fast_path_computes_the_same_function(self) -> None:
        reference, model = self.pair(fast=True)
        for disable in (False, True):
            a = reference(self.batch.tokens, disable_retrieval=disable)[0]
            b = model(self.batch.tokens, disable_retrieval=disable)[0]
            self.assertLess((a - b).abs().max().item(), 1e-5)

    def test_fast_path_gradients_match(self) -> None:
        reference, model = self.pair(fast=True)
        common_objective(reference(self.batch.tokens)[0], self.batch)[0].backward()
        common_objective(model(self.batch.tokens)[0], self.batch)[0].backward()
        for (name, a), b in zip(reference.named_parameters(), model.parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-5, rtol=1e-3), name)

    def test_fast_path_is_causal(self) -> None:
        model = PHLDAMv2(fast=True, tie_query_key=True)
        mutated = self.batch.tokens.clone()
        mutated[:, 100:] = 4
        with torch.no_grad():
            a = model(self.batch.tokens)[0]
            b = model(mutated)[0]
        self.assertTrue(torch.equal(a[:, :100], b[:, :100]))

    def test_tied_query_reads_with_the_write_projection(self) -> None:
        model = PHLDAMv2(fast=True, tie_query_key=True)
        self.assertFalse(hasattr(model, "query_projection"))
        token = torch.randn(3, model.d_model)
        expected = torch.nn.functional.normalize(model.key_projection(token), dim=-1)
        self.assertTrue(torch.equal(model._query(token), expected))

    def test_write_gate_initialisation_is_configurable(self) -> None:
        model = PHLDAMv2(write_gate_bias=-1.0)
        self.assertEqual(float(model.write_gate.bias), -1.0)

    def test_dropping_phl_gives_the_dnc_state_size(self) -> None:
        self.assertEqual(PHLDAMv2().state_floats(), 456)
        self.assertEqual(PHLDAMv2(fast=True, use_phl=False).state_floats(), 392)

    def test_every_variant_runs_finite_with_the_right_shape(self) -> None:
        for options in ({"fast": True}, {"fast": True, "use_phl": False},
                        {"fast": True, "learned_temperature": True},
                        {"fast": True, "tie_query_key": True, "write_gate_bias": -1.0}):
            logits = PHLDAMv2(**options)(self.batch.tokens)[0]
            self.assertEqual(logits.shape, (4, SEQUENCE_LENGTH, VOCAB_SIZE), options)
            self.assertTrue(torch.isfinite(logits).all(), options)

    def test_memory_ablation_changes_the_output(self) -> None:
        model = PHLDAMv2(fast=True)
        with torch.no_grad():
            self.assertFalse(torch.equal(
                model(self.batch.tokens)[0],
                model(self.batch.tokens, disable_retrieval=True)[0],
            ))


class LearningCurveTests(unittest.TestCase):
    def test_every_arm_builds_and_reports(self) -> None:
        for arm in ("phl_dam", "phl_dam_v2", "ntm_dnc", "ntm_dnc_factorized",
                    "transformer", "ssm_selective", "ssm_diagonal"):
            options = {"fast": True} if arm == "phl_dam_v2" else {}
            result = curve.run(arm, options, seed=0, steps=2, probe_every=1,
                               target=0.9, eval_episodes=4)
            self.assertTrue(result["finite"], arm)
            self.assertEqual(len(result["curve"]), 2, arm)
            self.assertLessEqual(result["final_recall"], 1.0)

    def test_training_stream_matches_the_published_scripts(self) -> None:
        """The probe must not perturb the seed + 10_000 training stream."""
        torch.manual_seed(0)
        model = curve.build("transformer", {})
        before = [p.clone() for p in model.parameters()]
        curve.recall_on(model, 30_000, 4)
        for a, b in zip(before, model.parameters()):
            self.assertTrue(torch.equal(a, b))
        a = make_batch(torch.Generator().manual_seed(10_000), 2).tokens
        b = make_batch(torch.Generator().manual_seed(10_000), 2).tokens
        self.assertTrue(torch.equal(a, b))

    def test_parameter_counts_are_matched_across_arms(self) -> None:
        for arm in ("phl_dam", "ntm_dnc", "ntm_dnc_factorized", "transformer",
                    "ssm_selective", "ssm_diagonal"):
            count = active_parameter_count(curve.build(arm, {}))
            self.assertLess(abs(count - 33_034) / 33_034, 0.01, arm)


if __name__ == "__main__":
    unittest.main()
