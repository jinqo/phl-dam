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


class CopyReadoutTests(unittest.TestCase):
    OPTIONS = {"fast": True, "normalized_values": True, "use_phl": False,
               "d_model": 76, "copy_readout": True}

    def test_a_clean_retrieval_decodes_to_its_token_at_initialisation(self) -> None:
        """The point of the copy readout: no decoder has to be learned first."""
        torch.manual_seed(0)
        model = PHLDAMv2(**self.OPTIONS)
        vocabulary = model.value_projection(model.token_embedding.weight)
        with torch.no_grad():
            scores = model.copy_scale * vocabulary @ vocabulary.T
        self.assertGreater(scores.argmax(-1).eq(torch.arange(VOCAB_SIZE)).float().mean(), 0.95)

    def test_copy_term_vanishes_when_retrieval_is_disabled(self) -> None:
        torch.manual_seed(0)
        model = PHLDAMv2(**self.OPTIONS)
        plain = PHLDAMv2(**{**self.OPTIONS, "copy_readout": False})
        plain.load_state_dict({k: v for k, v in model.state_dict().items()
                               if k != "copy_scale"})
        tokens = make_batch(torch.Generator().manual_seed(2), 2).tokens
        with torch.no_grad():
            self.assertTrue(torch.allclose(
                model(tokens, disable_retrieval=True)[0],
                plain(tokens, disable_retrieval=True)[0]))

    def test_adds_one_parameter_and_is_causal(self) -> None:
        with_copy = PHLDAMv2(**self.OPTIONS)
        without = PHLDAMv2(**{**self.OPTIONS, "copy_readout": False})
        self.assertEqual(active_parameter_count(with_copy),
                         active_parameter_count(without) + 1)
        tokens = make_batch(torch.Generator().manual_seed(3), 2).tokens
        mutated = tokens.clone()
        mutated[:, 90:] = 4
        with torch.no_grad():
            self.assertTrue(torch.equal(with_copy(tokens)[0][:, :90],
                                        with_copy(mutated)[0][:, :90]))

    def test_vocabulary_can_be_resized_for_the_pressure_task(self) -> None:
        model = PHLDAMv2(**{**self.OPTIONS, "vocab_size": 87})
        tokens = torch.randint(0, 87, (2, 50))
        self.assertEqual(model(tokens)[0].shape, (2, 50, 87))

    def test_occupancy_decay_keeps_reads_bounded_over_long_sequences(self) -> None:
        """Dividing by a decayed occupancy must not blow the read up.

        Values decay with occupancy, so each slot still reads as a weighted
        average of candidate values; logits over a 456-token sequence with a
        strong decay stay finite and of the same order as without decay.
        """
        torch.manual_seed(0)
        plain = PHLDAMv2(**{**self.OPTIONS, "vocab_size": 87})
        decayed = PHLDAMv2(**{**self.OPTIONS, "vocab_size": 87, "occupancy_decay": 0.05})
        decayed.load_state_dict(plain.state_dict())
        tokens = torch.randint(0, 87, (2, 456), generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            a, b = plain(tokens)[0], decayed(tokens)[0]
        self.assertTrue(torch.isfinite(b).all())
        self.assertLess(b.abs().max().item(), 3.0 * a.abs().max().item() + 1.0)
        self.assertFalse(torch.equal(a, b))

    def test_dnc_control_arm_gets_the_same_readout(self) -> None:
        model = curve.build("ntm_dnc_factorized_copy", {})
        self.assertTrue(model.copy_readout)
        self.assertLess(abs(active_parameter_count(model) - 33_034) / 33_034, 0.01)


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
