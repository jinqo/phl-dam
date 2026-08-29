"""Invariants for the PHL-DAM continual-learning protocol.

The experiment's whole force depends on tasks being genuinely disjoint and
equally hard, and on `memory` mode genuinely not training. These tests are what
make those claims checkable rather than asserted.
"""

import itertools
import unittest

import torch

import phl_dam_pressure_task as task
import phl_dam_continual as cont
from phl_dam_004b_lease import PHLDAMLease


class TaskConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("compact")
        self.specs = cont.build_tasks(4, seed=0)

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_tasks_partition_the_key_vocabulary_without_overlap(self) -> None:
        keys = {}
        for spec in self.specs:
            observed = set()
            for index in range(120):
                observed |= {
                    item.key_token for item in cont.task_episode(spec, index, 8).items
                }
            keys[spec.index] = observed
        for left, right in itertools.combinations(keys, 2):
            self.assertEqual(
                len(keys[left] & keys[right]), 0, f"tasks {left} and {right} share keys"
            )
        for spec in self.specs:
            low = task.KEY_START + spec.key_low
            high = task.KEY_START + spec.key_high
            self.assertTrue(all(low <= k < high for k in keys[spec.index]), spec.index)

    def test_tasks_are_equally_difficult(self) -> None:
        """Retention differences must come from interference, not difficulty."""
        shapes = []
        for spec in self.specs:
            episodes = [cont.task_episode(spec, i, 8) for i in range(200)]
            delays = [q.delay for e in episodes for q in e.queries]
            shapes.append(
                {
                    "queries": sum(len(e.queries) for e in episodes) / len(episodes),
                    "live": sum(
                        sum(1 for i in e.items if not i.is_never) for e in episodes
                    ) / len(episodes),
                    "delay_lo": min(delays),
                    "delay_hi": max(delays),
                }
            )
        self.assertLess(
            max(s["queries"] for s in shapes) - min(s["queries"] for s in shapes), 1.0
        )
        self.assertLess(
            max(s["live"] for s in shapes) - min(s["live"] for s in shapes), 1.0
        )
        self.assertEqual({s["delay_lo"] for s in shapes}, {task.MIN_DELAY})
        self.assertEqual({s["delay_hi"] for s in shapes}, {task.MAX_DELAY})

    def test_relabelling_preserves_the_token_stream_structure(self) -> None:
        """Only key identity changes; positions and values are untouched."""
        base = task.generate_episode(self.specs[2].seed, 5, 8, "canonical")
        remapped = cont.task_episode(self.specs[2], 5, 8)
        self.assertEqual(
            [i.write_start for i in base.items], [i.write_start for i in remapped.items]
        )
        self.assertEqual(
            [i.value_token for i in base.items], [i.value_token for i in remapped.items]
        )
        self.assertEqual(
            [q.key_position for q in base.queries],
            [q.key_position for q in remapped.queries],
        )
        self.assertEqual(int((remapped.tokens == task.QUERY).sum()), len(remapped.queries))

    def test_tokens_agree_with_the_relabelled_items(self) -> None:
        spec = self.specs[1]
        episode = cont.task_episode(spec, 9, 8)
        for item in episode.items:
            self.assertEqual(int(episode.tokens[item.write_start + 2]), item.key_token)
        for query in episode.queries:
            item = episode.items[query.item_index]
            self.assertEqual(int(episode.tokens[query.key_position]), item.key_token)

    def test_too_many_tasks_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cont.build_tasks(64, seed=0)


class ConflictingRegimeTests(unittest.TestCase):
    """The disjoint regime proved too easy - `weights` forgot only 0.029.

    The conflicting regime makes every task contradict its predecessors on the
    same keys, which is where catastrophic forgetting actually bites.
    """

    def setUp(self) -> None:
        task.set_scale("compact")
        self.specs = cont.build_tasks(4, seed=0, regime="conflicting")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_all_tasks_share_the_same_keys(self) -> None:
        keys = []
        for spec in self.specs:
            observed = set()
            for index in range(120):
                observed |= {
                    item.key_token for item in cont.task_episode(spec, index, 8).items
                }
            keys.append(observed)
        for other in keys[1:]:
            self.assertEqual(keys[0], other, "conflicting tasks must share keys")

    def test_the_same_key_means_a_different_value_in_each_task(self) -> None:
        mappings = []
        for spec in self.specs:
            mapping = {}
            for index in range(120):
                for item in cont.task_episode(spec, index, 8).items:
                    mapping.setdefault(item.key_token, set()).add(item.value_token)
            mappings.append(mapping)
        shared = set.intersection(*(set(m) for m in mappings))
        self.assertGreater(len(shared), 8)
        for key in shared:
            values = [tuple(sorted(m[key])) for m in mappings]
            self.assertEqual(
                len(set(values)), len(values), f"key {key} does not conflict across tasks"
            )

    def test_each_task_remains_solvable_from_context_alone(self) -> None:
        """The binding is always written before it is queried, in every task.

        This is what makes the regime a fair test: a model reading memory can be
        right on every task, while one answering from memorised weights cannot.
        """
        for spec in self.specs:
            for index in range(40):
                episode = cont.task_episode(spec, index, 8)
                for query in episode.queries:
                    item = episode.items[query.item_index]
                    self.assertLess(item.write_value_position, query.key_position)
                    self.assertEqual(
                        int(episode.tokens[query.target_position]), item.value_token
                    )

    def test_too_many_conflicting_tasks_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cont.build_tasks(task.NUM_VALUES + 1, seed=0, regime="conflicting")

    def test_unknown_regime_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cont.build_tasks(4, seed=0, regime="nonsense")


class MemorySnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("compact")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_detached_memory_is_an_independent_copy(self) -> None:
        model = PHLDAMLease(arm="content_only")
        state = model.init_state(2, torch.device("cpu"))
        state.dam.keys.normal_()
        snapshot = cont.detach_memory(state)
        self.assertTrue(torch.equal(snapshot.keys, state.dam.keys))
        state.dam.keys.zero_()
        self.assertFalse(torch.equal(snapshot.keys, state.dam.keys))
        self.assertFalse(snapshot.keys.requires_grad)


class PersistenceTests(unittest.TestCase):
    """Regression tests for a real bug.

    `both` mode was originally described as carrying memory across episodes but
    the wiring was never added, so it produced results bit-identical to
    `weights`. These tests fail if that regresses.
    """

    def setUp(self) -> None:
        task.set_scale("compact")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_supplying_a_memory_bank_changes_the_forward_pass(self) -> None:
        model = PHLDAMLease(arm="content_only")
        batch = cont.make_batch(
            cont.build_tasks(2, 0)[0], 0, 2, 8, torch.device("cpu")
        )
        with torch.no_grad():
            empty, _ = model(batch.tokens, arm="content_only", collect=False)
            state = model.init_state(2, torch.device("cpu"))
            state.dam.keys.normal_()
            state.dam.values.normal_()
            state.dam.occupancy.fill_(1.0)
            carried, _ = model(
                batch.tokens, arm="content_only", collect=False,
                initial_dam=cont.detach_memory(state),
            )
        self.assertFalse(
            torch.equal(empty, carried),
            "a non-empty starting memory must change the output",
        )

    def test_slice_memory_matches_the_requested_batch(self) -> None:
        model = PHLDAMLease(arm="content_only")
        state = model.init_state(1, torch.device("cpu"))
        state.dam.keys.normal_()
        bank = cont.detach_memory(state)
        sliced = cont.slice_memory(bank, 5)
        self.assertEqual(sliced.keys.shape[0], 5)
        self.assertTrue(torch.equal(sliced.keys[0], bank.keys[0]))
        self.assertTrue(torch.equal(sliced.keys[4], bank.keys[0]))
        self.assertIsNone(cont.slice_memory(None, 5))

    def test_modes_declare_their_behaviour_distinctly(self) -> None:
        expected = {
            "weights":    (True,  False),
            "memory":     (False, False),
            "both":       (True,  True),
            "persistent": (False, True),
        }
        for mode, (trains, carries) in expected.items():
            summary = cont.run(
                mode=mode, seed=0, tasks=2, steps_per_task=1, batch_size=2,
                eval_episodes=2, writes=8, learning_rate=2e-3,
                device=torch.device("cpu"),
            )
            config = summary["configuration"]
            self.assertEqual(config["trains_after_first_task"], trains, mode)
            self.assertEqual(config["carries_memory_across_episodes"], carries, mode)


class ModeTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("compact")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_memory_mode_freezes_weights_after_the_first_task(self) -> None:
        """The structural claim requires that no gradient step follows task 0."""
        seen = []
        original = torch.optim.AdamW.step

        def spy(self, *args, **kwargs):
            seen.append(1)
            return original(self, *args, **kwargs)

        torch.optim.AdamW.step = spy
        try:
            cont.run(
                mode="memory", seed=0, tasks=2, steps_per_task=2, batch_size=2,
                eval_episodes=2, writes=8, learning_rate=2e-3,
                device=torch.device("cpu"),
            )
            memory_steps = len(seen)
            seen.clear()
            cont.run(
                mode="weights", seed=0, tasks=2, steps_per_task=2, batch_size=2,
                eval_episodes=2, writes=8, learning_rate=2e-3,
                device=torch.device("cpu"),
            )
            weights_steps = len(seen)
        finally:
            torch.optim.AdamW.step = original
        self.assertEqual(memory_steps, 2, "memory mode must train on task 0 only")
        self.assertEqual(weights_steps, 4, "weights mode must train on every task")

    def test_run_reports_every_continual_quantity(self) -> None:
        summary = cont.run(
            mode="weights", seed=0, tasks=3, steps_per_task=2, batch_size=2,
            eval_episodes=2, writes=8, learning_rate=2e-3, device=torch.device("cpu"),
        )
        self.assertEqual(len(summary["accuracy_after_each_task"]), 3)
        self.assertEqual(len(summary["final_accuracy_per_task"]), 3)
        self.assertEqual(len(summary["forgetting_per_task"]), 2)
        for index, row in enumerate(summary["accuracy_after_each_task"]):
            self.assertEqual(len(row), index + 1, "must evaluate every task seen so far")
        self.assertTrue(summary["finite"])
        self.assertIsNotNone(summary["retention_old_tasks"])

    def test_forgetting_is_peak_minus_final(self) -> None:
        summary = cont.run(
            mode="weights", seed=1, tasks=3, steps_per_task=2, batch_size=2,
            eval_episodes=2, writes=8, learning_rate=2e-3, device=torch.device("cpu"),
        )
        peaks = summary["peak_accuracy_per_task"]
        final = summary["final_accuracy_per_task"]
        for index, value in enumerate(summary["forgetting_per_task"]):
            self.assertAlmostEqual(value, peaks[index] - final[index], places=10)
            self.assertGreaterEqual(peaks[index] + 1e-12, final[index])


if __name__ == "__main__":
    unittest.main()
