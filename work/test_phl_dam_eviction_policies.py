import random
import unittest

import phl_dam_pressure_task as task
from phl_dam_eviction_policies import (
    BeladyPolicy,
    EvictionContext,
    FIFOPolicy,
    LRUPolicy,
    OracleFutureRelevancePolicy,
    RandomPolicy,
    SlotRecord,
    build_policies,
    simulate,
)


def _item(index: int, query_positions: list[int]) -> task.Item:
    return task.Item(
        index=index,
        key_token=task.KEY_START + index,
        value_token=task.VALUE_START,
        tag_token=task.TAG_START,
        use_class=task.NEVER if not query_positions else task.MEDIUM,
        write_start=index * 4 + 1,
        write_value_position=index * 4 + 4,
        query_key_positions=list(query_positions),
    )


def _context(records: dict[int, SlotRecord], now: int) -> EvictionContext:
    next_use = {}
    for slot, record in records.items():
        upcoming = [p for p in record.item.query_key_positions if p > now]
        next_use[slot] = upcoming[0] if upcoming else None
    return EvictionContext(
        now=now, occupants=records, rng=random.Random(0), next_use=next_use
    )


class PolicyUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        # Slot 0: oldest insertion, accessed most recently, needed soon.
        # Slot 1: middle insertion, stale access, needed far in the future.
        # Slot 2: newest insertion, stale access, never needed again.
        self.records = {
            0: SlotRecord(item=_item(0, [310]), inserted_at=10, last_access=290),
            1: SlotRecord(item=_item(1, [400]), inserted_at=100, last_access=100),
            2: SlotRecord(item=_item(2, []), inserted_at=200, last_access=200),
        }
        self.context = _context(self.records, now=300)

    def test_fifo_evicts_the_oldest_insertion(self) -> None:
        self.assertEqual(FIFOPolicy().choose_victim(self.context), 0)

    def test_lru_evicts_the_least_recently_accessed(self) -> None:
        self.assertEqual(LRUPolicy().choose_victim(self.context), 1)

    def test_oracle_evicts_the_never_needed_item_when_one_exists(self) -> None:
        self.assertEqual(OracleFutureRelevancePolicy().choose_victim(self.context), 2)

    def test_oracle_evicts_farthest_next_use_when_all_are_needed(self) -> None:
        records = {
            0: SlotRecord(item=_item(0, [310]), inserted_at=10, last_access=290),
            1: SlotRecord(item=_item(1, [440]), inserted_at=100, last_access=100),
            2: SlotRecord(item=_item(2, [330]), inserted_at=200, last_access=200),
        }
        context = _context(records, now=300)
        self.assertEqual(OracleFutureRelevancePolicy().choose_victim(context), 1)
        self.assertEqual(BeladyPolicy().choose_victim(context), 1)

    def test_random_stays_within_the_occupied_slots(self) -> None:
        policy = RandomPolicy()
        for _ in range(50):
            self.assertIn(policy.choose_victim(self.context), self.records)

    def test_ties_break_on_the_lowest_slot_index(self) -> None:
        records = {
            3: SlotRecord(item=_item(3, []), inserted_at=50, last_access=50),
            1: SlotRecord(item=_item(1, []), inserted_at=50, last_access=50),
        }
        context = _context(records, now=300)
        self.assertEqual(FIFOPolicy().choose_victim(context), 1)
        self.assertEqual(LRUPolicy().choose_victim(context), 1)


class SimulatorInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = task.generate_episodes(2, 120, 32, "canonical")
        self.policies = build_policies()

    def test_occupancy_never_exceeds_the_slot_count(self) -> None:
        for name, policy in self.policies.items():
            for index, episode in enumerate(self.episodes[:30]):
                occupancy = _replay_occupancy(episode, policy, index)
                self.assertLessEqual(max(occupancy), task.NUM_SLOTS, name)

    def test_writes_fill_empty_slots_before_any_eviction(self) -> None:
        policy = self.policies["fifo"]
        episode = self.episodes[0]
        trace = simulate(episode, policy, task.NUM_SLOTS, random.Random(0))
        self.assertEqual(trace.evictions, episode.writes - task.NUM_SLOTS)

    def test_eviction_removes_exactly_the_selected_item(self) -> None:
        episode = self.episodes[1]
        trace = simulate(episode, self.policies["lru"], task.NUM_SLOTS, random.Random(0))
        self.assertEqual(
            trace.evictions, trace.future_needed_evicted + trace.dead_evicted
        )
        self.assertEqual(
            trace.dead_retained_at_end + trace.live_retained_at_end, task.NUM_SLOTS
        )

    def test_every_policy_sees_the_same_queries(self) -> None:
        counts = set()
        for policy in self.policies.values():
            total = sum(
                len(simulate(episode, policy, task.NUM_SLOTS, random.Random(0)).outcomes)
                for episode in self.episodes
            )
            counts.add(total)
        self.assertEqual(len(counts), 1)

    def test_belady_is_the_same_policy_as_oracle_future_relevance(self) -> None:
        """Documented, not hidden: under this generator the two coincide."""
        oracle = self.policies["oracle_future_relevance"]
        belady = self.policies["belady"]
        for episode in self.episodes:
            left = simulate(episode, oracle, task.NUM_SLOTS, random.Random(0))
            right = simulate(episode, belady, task.NUM_SLOTS, random.Random(0))
            self.assertEqual(
                [o.hit for o in left.outcomes], [o.hit for o in right.outcomes]
            )

    def test_oracle_is_never_beaten_by_a_non_oracle_policy(self) -> None:
        rates = {}
        for name, policy in self.policies.items():
            hits = 0
            total = 0
            for index, episode in enumerate(self.episodes):
                trace = simulate(episode, policy, task.NUM_SLOTS, random.Random(index))
                hits += sum(outcome.hit for outcome in trace.outcomes)
                total += len(trace.outcomes)
            rates[name] = hits / total
        oracle = rates["oracle_future_relevance"]
        for name in ("random", "fifo", "lru"):
            self.assertLessEqual(rates[name], oracle + 1e-12, name)

    def test_simulation_is_deterministic_under_a_fixed_seed(self) -> None:
        episode = self.episodes[3]
        for name, policy in self.policies.items():
            first = simulate(episode, policy, task.NUM_SLOTS, random.Random(17))
            again = simulate(episode, policy, task.NUM_SLOTS, random.Random(17))
            self.assertEqual(
                [o.hit for o in first.outcomes], [o.hit for o in again.outcomes], name
            )

    def test_a_resident_binding_survives_its_own_query(self) -> None:
        """Queries must not consume the binding, or LRU becomes meaningless."""
        repeats = 0
        for episode in self.episodes:
            trace = simulate(
                episode,
                self.policies["oracle_future_relevance"],
                task.NUM_SLOTS,
                random.Random(0),
            )
            for outcome in trace.outcomes:
                if outcome.occurrence > 0 and outcome.hit:
                    repeats += 1
        self.assertGreater(repeats, 0)


def _replay_occupancy(episode, policy, index: int) -> list[int]:
    """Independent replay that records occupancy after every write."""
    rng = random.Random(index)
    occupants: dict[int, SlotRecord] = {}
    slot_of_item: dict[int, int] = {}
    sizes = [0]
    events = [(item.write_value_position, 0, item) for item in episode.items]
    events += [(query.key_position, 1, query) for query in episode.queries]
    events.sort(key=lambda event: (event[0], event[1]))
    for now, kind, payload in events:
        if kind == 1:
            slot = slot_of_item.get(payload.item_index)
            if slot is not None:
                occupants[slot].last_access = now
            continue
        free = [slot for slot in range(task.NUM_SLOTS) if slot not in occupants]
        if free:
            slot = free[0]
        else:
            next_use = {}
            for candidate, record in occupants.items():
                upcoming = [p for p in record.item.query_key_positions if p > now]
                next_use[candidate] = upcoming[0] if upcoming else None
            slot = policy.choose_victim(
                EvictionContext(now=now, occupants=occupants, rng=rng, next_use=next_use)
            )
            del slot_of_item[occupants[slot].item.index]
            del occupants[slot]
        occupants[slot] = SlotRecord(item=payload, inserted_at=now, last_access=now)
        slot_of_item[payload.index] = slot
        sizes.append(len(occupants))
    return sizes


if __name__ == "__main__":
    unittest.main()
