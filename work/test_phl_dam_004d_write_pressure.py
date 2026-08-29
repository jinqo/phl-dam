"""Invariants for the PHL-DAM-004D write-pressure ladder.

The point of 004D is that write count is the ONLY thing that varies across
conditions. These tests are what makes that claim checkable.
"""

import math
import unittest

import torch

import phl_dam_pressure_task as task
import phl_dam_004d_write_pressure as pressure
from phl_dam_004b_lease import PHLDAMLease


class PressureProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("pressure")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_only_write_count_varies_across_levels(self) -> None:
        shapes = {}
        for writes in task.PRESSURE_LEVELS:
            episodes = task.generate_episodes(0, 400, writes, "canonical")
            delays = [q.delay for e in episodes for q in e.queries]
            shapes[writes] = {
                "length": {int(e.tokens.shape[0]) for e in episodes},
                "queries": sum(len(e.queries) for e in episodes) / len(episodes),
                "live": sum(
                    sum(1 for i in e.items if not i.is_never) for e in episodes
                ) / len(episodes),
                "peak": sum(e.max_concurrent_live for e in episodes) / len(episodes),
                "delay_lo": min(delays),
                "delay_hi": max(delays),
                "dead": sum(
                    sum(1 for i in e.items if i.is_never) for e in episodes
                ) / len(episodes),
            }
        # Pinned across the ladder.
        self.assertEqual(
            {frozenset(v["length"]) for v in shapes.values()},
            {frozenset({task.SEQUENCE_LENGTH})},
        )
        self.assertLess(
            max(v["queries"] for v in shapes.values())
            - min(v["queries"] for v in shapes.values()),
            1.0,
        )
        self.assertLess(
            max(v["live"] for v in shapes.values())
            - min(v["live"] for v in shapes.values()),
            1.0,
        )
        self.assertLess(
            max(v["peak"] for v in shapes.values())
            - min(v["peak"] for v in shapes.values()),
            1.0,
        )
        self.assertEqual({v["delay_lo"] for v in shapes.values()}, {task.MIN_DELAY})
        self.assertEqual({v["delay_hi"] for v in shapes.values()}, {task.MAX_DELAY})
        # The one thing that does vary, and by a lot.
        dead = [shapes[w]["dead"] for w in sorted(shapes)]
        self.assertEqual(dead, sorted(dead))
        self.assertGreater(dead[-1] / max(dead[0], 1e-9), 5.0)

    def test_live_items_stay_below_slot_capacity(self) -> None:
        """Pressure must come from distractor writes, not live over-subscription."""
        for writes in task.PRESSURE_LEVELS:
            episodes = task.generate_episodes(1, 300, writes, "canonical")
            peaks = [e.max_concurrent_live for e in episodes]
            self.assertLess(sum(peaks) / len(peaks), task.NUM_SLOTS, writes)

    def test_write_events_all_fit_before_the_delay_window(self) -> None:
        for writes in task.PRESSURE_LEVELS:
            for episode in task.generate_episodes(2, 60, writes, "canonical"):
                self.assertEqual(len(episode.items), writes)
                last = max(i.write_value_position for i in episode.items)
                self.assertLessEqual(last, task.WRITE_REGION_END + 3)


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("pressure")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_controller_telemetry_is_complete_and_finite(self) -> None:
        model = PHLDAMLease(arm="content_only")
        batch, binding, episodes = pressure.probe_batch(0, 16, torch.device("cpu"))
        row = pressure.controller_telemetry(model, batch, binding, episodes)
        required = {
            "write_gate_at_binding",
            "write_gate_elsewhere",
            "write_selectivity",
            "write_commitments",
            "commitments_off_binding_fraction",
            "allocation_entropy_at_binding",
            "allocation_entropy_elsewhere",
            "allocation_margin_at_binding",
            "mean_occupied_slots",
            "slot_replacement_rate",
            "residency_estimate",
        }
        self.assertTrue(required <= set(row))
        for name, value in row.items():
            self.assertTrue(math.isfinite(value), name)
        self.assertLessEqual(row["mean_occupied_slots"], model.num_slots)
        self.assertLessEqual(
            row["allocation_entropy_at_binding"], math.log(model.num_slots) + 1e-6
        )
        self.assertAlmostEqual(
            row["write_selectivity"],
            row["write_gate_at_binding"] - row["write_gate_elsewhere"],
            places=6,
        )

    def test_probe_batch_is_disjoint_from_training_episodes(self) -> None:
        """Telemetry must never be measured on the batches being trained on."""
        pressure.probe_batch(0, 16, torch.device("cpu"))
        training = [
            task.generate_episode(0 + 500_000, 1 * 16 + i, 16, "canonical")
            for i in range(16)
        ]
        probe_seeds = {
            task.episode_seed(0 + 900_000, i) for i in range(pressure.PROBE_EPISODES)
        }
        self.assertFalse(probe_seeds & {e.seed for e in training})


class DivergenceProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("pressure")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_a_healthy_short_run_reports_finite_and_records_gradients(self) -> None:
        _, history, divergence = pressure.train_with_telemetry(
            writes=8,
            seed=0,
            steps=6,
            batch_size=4,
            learning_rate=2e-3,
            device=torch.device("cpu"),
        )
        self.assertTrue(divergence["finite"])
        self.assertIsNone(divergence["first_nonfinite_step"])
        self.assertGreater(divergence["max_gradient_norm"], 0.0)
        self.assertTrue(all(math.isfinite(r["gradient_norm"]) for r in history))

    def test_divergence_is_recorded_rather_than_discarded(self) -> None:
        """A run that goes non-finite must be marked, not dropped."""
        original = pressure.common_objective

        def exploding(logits, batch):
            loss, all_ce, recall_ce = original(logits, batch)
            return loss * float("inf"), all_ce, recall_ce

        pressure.common_objective = exploding
        try:
            _, history, divergence = pressure.train_with_telemetry(
                writes=8,
                seed=0,
                steps=3,
                batch_size=4,
                learning_rate=2e-3,
                device=torch.device("cpu"),
            )
        finally:
            pressure.common_objective = original
        self.assertFalse(divergence["finite"])
        self.assertIsNotNone(divergence["first_nonfinite_step"])
        self.assertIsNotNone(divergence["failure_reason"])
        self.assertTrue(history, "history must be preserved for a diverged run")

    def test_long_horizon_training_stays_finite(self) -> None:
        """The one-step finiteness tests cannot see delayed instability.

        This trains far enough to catch a divergence that develops over many
        updates, which is how the PHL-DAM-004C NaN escaped the suite.
        """
        task.set_scale("compact")
        model, history, divergence = pressure.train_with_telemetry(
            writes=12,
            seed=0,
            steps=120,
            batch_size=8,
            learning_rate=2e-3,
            device=torch.device("cpu"),
        )
        self.assertTrue(divergence["finite"], divergence["failure_reason"])
        for record in history:
            self.assertTrue(math.isfinite(record["loss"]), record["step"])
            self.assertTrue(math.isfinite(record["gradient_norm"]), record["step"])
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all(), name)


if __name__ == "__main__":
    unittest.main()
