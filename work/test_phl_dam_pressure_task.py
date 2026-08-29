import unittest
from collections import Counter

import torch

import phl_dam_pressure_task as task


class PressureTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        task.set_scale("full")
        self.episodes = task.generate_episodes(0, 300, 24, "canonical")

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_writes_outnumber_slots_and_keys_are_distinct(self) -> None:
        for episode in self.episodes[:50]:
            self.assertGreater(episode.writes, task.NUM_SLOTS)
            self.assertEqual(len(episode.items), episode.writes)
            keys = [item.key_token for item in episode.items]
            self.assertEqual(len(set(keys)), len(keys))

    def test_only_a_subset_of_writes_is_ever_queried(self) -> None:
        live = sum(
            1 for episode in self.episodes for item in episode.items if not item.is_never
        )
        total = sum(len(episode.items) for episode in self.episodes)
        self.assertLess(live, total)
        self.assertGreater(live / total, 0.10)

    def test_every_query_is_causal_and_inside_the_delay_range(self) -> None:
        for episode in self.episodes:
            for query in episode.queries:
                item = episode.items[query.item_index]
                self.assertGreater(query.key_position, item.write_value_position)
                self.assertGreaterEqual(query.delay, task.MIN_DELAY)
                self.assertLessEqual(query.delay, task.MAX_DELAY)
                self.assertEqual(query.delay, query.key_position - item.write_value_position)

    def test_delay_bins_cover_every_realised_delay(self) -> None:
        delays = [query.delay for episode in self.episodes for query in episode.queries]
        for delay in delays:
            self.assertNotEqual(task.delay_bin(delay), "out-of-range")
        covered = {task.delay_bin(delay) for delay in delays}
        self.assertEqual(covered, {name for name, _, _ in task.DELAY_BINS})

    def test_repeat_queries_exist_and_stay_ordered(self) -> None:
        repeats = 0
        for episode in self.episodes:
            for item in episode.items:
                positions = item.query_key_positions
                self.assertEqual(positions, sorted(positions))
                self.assertEqual(len(set(positions)), len(positions))
                if len(positions) > 1:
                    repeats += 1
        self.assertGreater(repeats, 0)

    def test_events_never_overlap_in_the_token_stream(self) -> None:
        for episode in self.episodes[:50]:
            claimed: set[int] = set()
            for item in episode.items:
                window = set(range(item.write_start, item.write_start + 4))
                self.assertFalse(window & claimed)
                claimed |= window
            for query in episode.queries:
                window = set(range(query.marker_position, query.marker_position + 3))
                self.assertFalse(window & claimed)
                claimed |= window
            self.assertNotIn(0, claimed)

    def test_tokens_encode_the_schedule_exactly(self) -> None:
        episode = self.episodes[0]
        tokens = episode.tokens
        self.assertEqual(tokens.shape, (task.SEQUENCE_LENGTH,))
        self.assertEqual(int(tokens[0]), task.BOS)
        self.assertEqual(int((tokens == task.WRITE).sum()), episode.writes)
        self.assertEqual(int((tokens == task.QUERY).sum()), len(episode.queries))
        for item in episode.items:
            start = item.write_start
            self.assertEqual(int(tokens[start]), task.WRITE)
            self.assertEqual(int(tokens[start + 1]), item.tag_token)
            self.assertEqual(int(tokens[start + 2]), item.key_token)
            self.assertEqual(int(tokens[start + 3]), item.value_token)
        for query in episode.queries:
            item = episode.items[query.item_index]
            self.assertEqual(int(tokens[query.marker_position]), task.QUERY)
            self.assertEqual(int(tokens[query.key_position]), item.key_token)
            self.assertEqual(int(tokens[query.target_position]), item.value_token)

    @staticmethod
    def _tag_drift(episodes) -> float:
        """Largest early-half vs late-half gap in the tag marginal."""
        early: Counter = Counter()
        late: Counter = Counter()
        for episode in episodes:
            midpoint = len(episode.items) // 2
            for item in episode.items[:midpoint]:
                early[item.tag_token] += 1
            for item in episode.items[midpoint:]:
                late[item.tag_token] += 1
        early_total = sum(early.values())
        late_total = sum(late.values())
        return max(
            abs(early[tag] / early_total - late[tag] / late_total)
            for tag in range(task.TAG_START, task.TAG_START + task.NUM_TAGS)
        )

    def test_tag_is_independent_of_write_order_in_randomised_episodes(self) -> None:
        """A recency policy must not read the lease cue off position.

        Exact independence is asserted on the randomised bulk. The controlled
        contrast subset deliberately forces dead writes after its anchor, so
        the full mixture carries a small, bounded and intentional dependence.
        """
        bulk = [episode for episode in self.episodes if not episode.is_contrast]
        self.assertLess(self._tag_drift(bulk), 0.02)

    def test_full_mixture_tag_drift_stays_bounded(self) -> None:
        self.assertLess(self._tag_drift(self.episodes), 0.05)

    def test_use_class_is_independent_of_write_position(self) -> None:
        positions_by_class: dict[int, list[int]] = {}
        for episode in self.episodes:
            for item in episode.items:
                key = task.NEVER if item.is_never else item.use_class
                positions_by_class.setdefault(key, []).append(item.write_value_position)
        means = {
            name: sum(values) / len(values)
            for name, values in positions_by_class.items()
            if len(values) > 200
        }
        self.assertGreater(len(means), 3)
        self.assertLess(max(means.values()) - min(means.values()), 12.0)

    def test_tag_signal_is_informative_but_far_from_perfect(self) -> None:
        counts = torch.zeros(task.NUM_CLASSES)
        for episode in self.episodes:
            for item in episode.items:
                counts[task.NEVER if item.is_never else item.use_class] += 1
        prior = counts / counts.sum()
        accuracy = task.bayes_tag_accuracy(prior)
        auroc = task.bayes_live_auroc(prior)
        self.assertGreater(accuracy, 0.55)
        self.assertLess(accuracy, 0.80)
        self.assertGreater(auroc, 0.70)
        self.assertLess(auroc, 0.95)

    def test_canonical_condition_over_subscribes_the_slots_under_pressure(self) -> None:
        for writes, minimum_above_eight in ((24, 0.50), (32, 0.85)):
            episodes = task.generate_episodes(1, 500, writes, "canonical")
            peaks = [episode.max_concurrent_live for episode in episodes]
            above = sum(peak > task.NUM_SLOTS for peak in peaks) / len(peaks)
            self.assertGreater(above, minimum_above_eight, writes)

    def test_generation_is_deterministic_and_seed_dependent(self) -> None:
        first = task.generate_episode(3, 11, 24)
        again = task.generate_episode(3, 11, 24)
        self.assertTrue(torch.equal(first.tokens, again.tokens))
        self.assertEqual(
            [q.key_position for q in first.queries],
            [q.key_position for q in again.queries],
        )
        other = task.generate_episode(4, 11, 24)
        self.assertFalse(torch.equal(first.tokens, other.tokens))

    def test_generation_is_independent_of_torch_rng_state(self) -> None:
        torch.manual_seed(1234)
        first = task.generate_episode(5, 2, 32)
        torch.manual_seed(9999)
        _ = torch.randn(64)
        again = task.generate_episode(5, 2, 32)
        self.assertTrue(torch.equal(first.tokens, again.tokens))

    def test_positions_are_randomised_across_episodes(self) -> None:
        first_write = {episode.items[0].write_start for episode in self.episodes}
        self.assertGreater(len(first_write), 10)
        first_query = {
            episode.queries[0].marker_position
            for episode in self.episodes
            if episode.queries
        }
        self.assertGreater(len(first_query), 20)

    def test_contrast_episodes_are_a_minority_with_an_old_live_anchor(self) -> None:
        contrast = [episode for episode in self.episodes if episode.is_contrast]
        self.assertGreater(len(contrast), 10)
        self.assertLess(len(contrast) / len(self.episodes), 0.30)
        anchored = 0
        for episode in contrast:
            live = [item for item in episode.items if not item.is_never]
            if live and min(item.index for item in live) <= 1:
                anchored += 1
        self.assertGreater(anchored / len(contrast), 0.80)


class ScaleProfileTests(unittest.TestCase):
    """The compact profile must be the same task, only shorter."""

    def tearDown(self) -> None:
        task.set_scale("full")

    def test_profiles_set_every_scale_dependent_global(self) -> None:
        task.set_scale("compact")
        self.assertEqual(task.SCALE, "compact")
        self.assertEqual(task.SEQUENCE_LENGTH, 176)
        self.assertEqual(task.PRESSURE_LEVELS, (8, 12, 16))
        task.set_scale("full")
        self.assertEqual(task.SEQUENCE_LENGTH, 456)
        self.assertEqual(task.PRESSURE_LEVELS, (16, 24, 32))

    def test_compact_episodes_respect_their_own_delay_bounds(self) -> None:
        task.set_scale("compact")
        episodes = task.generate_episodes(0, 300, 16, "canonical")
        for episode in episodes:
            self.assertEqual(episode.tokens.shape, (176,))
            for query in episode.queries:
                item = episode.items[query.item_index]
                self.assertGreater(query.key_position, item.write_value_position)
                self.assertGreaterEqual(query.delay, task.MIN_DELAY)
                self.assertLessEqual(query.delay, task.MAX_DELAY)
                self.assertNotEqual(task.delay_bin(query.delay), "out-of-range")

    def test_compact_canonical_still_over_subscribes_the_slots(self) -> None:
        task.set_scale("compact")
        episodes = task.generate_episodes(1, 500, 16, "canonical")
        peaks = [episode.max_concurrent_live for episode in episodes]
        self.assertGreater(sum(p > task.NUM_SLOTS for p in peaks) / len(peaks), 0.60)

    def test_compact_keeps_the_same_cue_strength(self) -> None:
        task.set_scale("compact")
        counts = torch.zeros(task.NUM_CLASSES)
        for episode in task.generate_episodes(0, 300, 16, "canonical"):
            for item in episode.items:
                counts[task.NEVER if item.is_never else item.use_class] += 1
        prior = counts / counts.sum()
        self.assertGreater(task.bayes_tag_accuracy(prior), 0.55)
        self.assertLess(task.bayes_tag_accuracy(prior), 0.80)

    def test_dilation_stretches_time_without_changing_the_task(self) -> None:
        """Temporal dilation must be a single factor, not a difficulty change."""
        shapes = {}
        for name in ("dilate10", "dilate15", "dilate20", "dilate26"):
            task.set_scale(name)
            episodes = task.generate_episodes(0, 300, 12, "canonical")
            peaks = [e.max_concurrent_live for e in episodes]
            delays = [q.delay for e in episodes for q in e.queries]
            shapes[name] = {
                "length": task.SEQUENCE_LENGTH,
                "peak": sum(peaks) / len(peaks),
                "queries": sum(len(e.queries) for e in episodes) / len(episodes),
                "delay_span": max(delays) - min(delays),
            }
            for delay in delays:
                self.assertGreaterEqual(delay, task.MIN_DELAY)
                self.assertLessEqual(delay, task.MAX_DELAY)
                self.assertNotEqual(task.delay_bin(delay), "out-of-range")
        lengths = [row["length"] for row in shapes.values()]
        self.assertEqual(lengths, sorted(lengths))
        self.assertGreater(lengths[-1] / lengths[0], 2.0)
        peaks = [row["peak"] for row in shapes.values()]
        queries = [row["queries"] for row in shapes.values()]
        self.assertLess(max(peaks) - min(peaks), 0.8)
        self.assertLess(max(queries) - min(queries), 0.8)

    def test_unknown_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            task.set_scale("enormous")


if __name__ == "__main__":
    unittest.main()
