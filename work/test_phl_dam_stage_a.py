import io
import unittest

import torch

from phl_dam_stage_a import OracleDAM, episode_loss, make_episodes


class OracleDAMTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = OracleDAM()

    def test_empty_memory_read_is_finite_and_zero(self) -> None:
        state = self.model.init_state(2, torch.device("cpu"))
        logits, attention, retrieved = self.model.read(state, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(attention, torch.zeros_like(attention)))
        self.assertTrue(torch.equal(retrieved, torch.zeros_like(retrieved)))

    def test_single_write_occupies_only_selected_slot(self) -> None:
        state = self.model.init_state(2, torch.device("cpu"))
        written = self.model.write_slot(
            state, 4, torch.tensor([1, 2]), torch.tensor([3, 4])
        )
        self.assertTrue(written.occupancy[:, 4].all())
        self.assertEqual(written.occupancy.sum().item(), 2)
        self.assertFalse(state.occupancy.any())

    def test_multiple_bindings_and_overwrite(self) -> None:
        keys = torch.tensor([[1, 2, 3], [4, 5, 6]])
        values = torch.tensor([[2, 4, 6], [1, 3, 5]])
        state = self.model.write_episode(keys, values)
        self.assertTrue(state.occupancy[:, :3].all())
        old_value = state.values[:, 1].clone()
        overwritten = self.model.write_slot(
            state, 1, torch.tensor([7, 8]), torch.tensor([8, 9])
        )
        self.assertFalse(torch.equal(old_value, overwritten.values[:, 1]))
        self.assertTrue(torch.equal(state.values[:, 1], old_value))

    def test_state_clone_has_no_aliasing(self) -> None:
        state = self.model.init_state(1, torch.device("cpu"))
        copied = state.clone()
        copied.keys[0, 0, 0] = 12.0
        copied.occupancy[0, 0] = True
        self.assertEqual(state.keys[0, 0, 0].item(), 0.0)
        self.assertFalse(state.occupancy[0, 0].item())

    def test_future_write_does_not_change_past_read(self) -> None:
        state = self.model.init_state(1, torch.device("cpu"))
        state = self.model.write_slot(state, 0, torch.tensor([2]), torch.tensor([3]))
        past_logits = self.model.read(state, torch.tensor([2]))[0].clone()
        _future_state = self.model.write_slot(
            state, 1, torch.tensor([4]), torch.tensor([5])
        )
        replayed_past_logits = self.model.read(state, torch.tensor([2]))[0]
        self.assertTrue(torch.equal(past_logits, replayed_past_logits))

    def test_gradients_reach_learned_memory_path(self) -> None:
        generator = torch.Generator().manual_seed(11)
        keys, values, order, _ = make_episodes(generator, 64, 32, 10)
        loss = episode_loss(self.model, keys, values, order)
        loss.backward()
        expected = (
            "key_embedding.weight",
            "value_embedding.weight",
            "binding_encoder.0.weight",
            "key_projection.weight",
            "value_projection.weight",
            "query_projection.weight",
            "output.weight",
        )
        parameters = dict(self.model.named_parameters())
        for name in expected:
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0.0, name)

    def test_checkpoint_round_trip(self) -> None:
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = OracleDAM()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        for original, loaded in zip(self.model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(original, loaded))

    def test_episode_constraints(self) -> None:
        keys = torch.zeros(1, 9, dtype=torch.long)
        values = torch.zeros(1, 9, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model.write_episode(keys, values)


if __name__ == "__main__":
    unittest.main()
