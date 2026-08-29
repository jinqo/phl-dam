import unittest

import torch

from phl_dam_lease_pressure import (
    NUM_SLOTS,
    NUM_WRITES,
    LeasePredictor,
    generate_episode,
    sample_cues,
    simulate_policy,
    transported_lease_priority,
)


class LeasePressureTests(unittest.TestCase):
    def test_protocol_creates_memory_pressure_and_causal_queries(self) -> None:
        generator = torch.Generator().manual_seed(100)
        bindings = generate_episode(generator)
        self.assertEqual(len(bindings), NUM_WRITES)
        self.assertGreater(NUM_WRITES, NUM_SLOTS)
        queried = [binding for binding in bindings if binding.query_time is not None]
        self.assertGreater(len(queried), NUM_SLOTS)
        self.assertTrue(
            all(binding.query_time > binding.write_time for binding in queried)
        )

    def test_transport_changes_priority_and_expires_mass(self) -> None:
        near = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.assertGreater(
            transported_lease_priority(near, 5),
            transported_lease_priority(near, 0),
        )
        self.assertEqual(transported_lease_priority(near, 10), 0.0)

    def test_predictor_receives_gradient(self) -> None:
        generator = torch.Generator().manual_seed(101)
        cues, labels = sample_cues(generator, 64)
        model = LeasePredictor()
        loss = torch.nn.functional.cross_entropy(model(cues), labels)
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_all_policies_share_queries(self) -> None:
        generator = torch.Generator().manual_seed(102)
        bindings = generate_episode(generator)
        model = LeasePredictor()
        with torch.no_grad():
            probabilities = model(
                torch.stack([binding.cue for binding in bindings])
            ).softmax(dim=-1)
        query_counts = []
        for policy in (
            "random",
            "fifo",
            "static_learned",
            "randomized_lease",
            "phl_transported_lease",
            "oracle_next_use",
        ):
            _, queries = simulate_policy(bindings, probabilities, policy, seed=103)
            query_counts.append(queries)
        self.assertEqual(len(set(query_counts)), 1)


if __name__ == "__main__":
    unittest.main()
