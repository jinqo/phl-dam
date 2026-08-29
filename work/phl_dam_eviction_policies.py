"""Eviction policies and the exact-content-store simulator for PHL-DAM-004.

The simulator here deliberately stores key/value content *exactly*: a query is
a hit if and only if the binding is still resident. That makes eviction
quality the only variable, so PHL-DAM-004A measures the retention ceiling of
each policy without any content-addressing error mixed in.

Access, for LRU and for every recency feature derived from it, is defined as:

    a slot is accessed at the timestep it is written, and again at any query
    timestep whose queried key is the key held by that slot.

PHL-DAM-004B reuses the same definition with "the key held by that slot"
replaced by "the argmax of the content-address distribution".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from phl_dam_pressure_task import Episode, Item


@dataclass
class SlotRecord:
    item: Item
    inserted_at: int
    last_access: int
    accesses: int = 1


@dataclass
class EvictionContext:
    """Everything a policy may look at when choosing a victim."""

    now: int
    occupants: dict[int, SlotRecord]
    rng: random.Random
    next_use: dict[int, int | None] = field(default_factory=dict)


class EvictionPolicy:
    """Lowest priority is evicted. Ties break on the lowest slot index."""

    name = "base"
    uses_future = False

    def priority(self, slot: int, record: SlotRecord, context: EvictionContext) -> float:
        raise NotImplementedError

    def choose_victim(self, context: EvictionContext) -> int:
        best_slot = None
        best_priority = None
        for slot in sorted(context.occupants):
            value = self.priority(slot, context.occupants[slot], context)
            if best_priority is None or value < best_priority:
                best_slot = slot
                best_priority = value
        assert best_slot is not None
        return best_slot


class RandomPolicy(EvictionPolicy):
    name = "random"

    def priority(self, slot, record, context):
        return context.rng.random()


class FIFOPolicy(EvictionPolicy):
    name = "fifo"

    def priority(self, slot, record, context):
        return float(record.inserted_at)


class LRUPolicy(EvictionPolicy):
    name = "lru"

    def priority(self, slot, record, context):
        return float(record.last_access)


class OracleFutureRelevancePolicy(EvictionPolicy):
    """Evict a never-needed-again item; otherwise the farthest next use."""

    name = "oracle_future_relevance"
    uses_future = True

    def priority(self, slot, record, context):
        next_use = context.next_use.get(slot)
        if next_use is None:
            return -1.0
        return 1.0 / float(next_use - context.now + 1)


class BeladyPolicy(EvictionPolicy):
    """Classic farthest-next-use, treating never-used-again as infinity."""

    name = "belady"
    uses_future = True

    def priority(self, slot, record, context):
        next_use = context.next_use.get(slot)
        distance = float("inf") if next_use is None else float(next_use - context.now)
        return -distance


POLICY_CLASSES = (
    RandomPolicy,
    FIFOPolicy,
    LRUPolicy,
    OracleFutureRelevancePolicy,
    BeladyPolicy,
)
POLICY_NAMES = tuple(policy.name for policy in POLICY_CLASSES)


def build_policies() -> dict[str, EvictionPolicy]:
    return {policy.name: policy() for policy in POLICY_CLASSES}


@dataclass
class QueryOutcome:
    hit: bool
    delay: int
    intervening_writes: int
    occurrence: int
    time_since_access: int
    evictions_since_write: int
    is_contrast_anchor: bool


@dataclass
class EpisodeTrace:
    outcomes: list[QueryOutcome]
    evictions: int
    future_needed_evicted: int
    dead_evicted: int
    dead_retained_at_end: int
    live_retained_at_end: int
    wrong_protection: int
    correct_protection: int
    contrast_decisions: int
    contrast_correct: int
    occupancy_sum: float
    occupancy_samples: int
    lifetime_sum: int
    lifetime_count: int


def _next_use(item: Item, now: int) -> int | None:
    for position in item.query_key_positions:
        if position > now:
            return position
    return None


def simulate(
    episode: Episode,
    policy: EvictionPolicy,
    num_slots: int,
    rng: random.Random,
) -> EpisodeTrace:
    """Run one episode through one policy against an exact content store."""
    occupants: dict[int, SlotRecord] = {}
    slot_of_item: dict[int, int] = {}
    evictions_at: list[int] = []

    outcomes: list[QueryOutcome] = []
    evictions = 0
    future_needed_evicted = 0
    dead_evicted = 0
    wrong_protection = 0
    correct_protection = 0
    contrast_decisions = 0
    contrast_correct = 0
    occupancy_sum = 0.0
    occupancy_samples = 0
    lifetime_sum = 0
    lifetime_count = 0

    anchor_index = None
    if episode.is_contrast:
        live = [item for item in episode.items if not item.is_never]
        if live:
            anchor_index = min(live, key=lambda item: item.write_value_position).index

    events: list[tuple[int, int, object]] = []
    for item in episode.items:
        events.append((item.write_value_position, 0, item))
    for query in episode.queries:
        events.append((query.key_position, 1, query))
    events.sort(key=lambda event: (event[0], event[1]))

    for now, kind, payload in events:
        if kind == 1:
            query = payload
            item = episode.items[query.item_index]
            slot = slot_of_item.get(item.index)
            hit = slot is not None
            if hit:
                record = occupants[slot]
                time_since_access = now - record.last_access
                record.last_access = now
                record.accesses += 1
            else:
                time_since_access = -1
            outcomes.append(
                QueryOutcome(
                    hit=hit,
                    delay=query.delay,
                    intervening_writes=query.intervening_writes,
                    occurrence=query.occurrence,
                    time_since_access=time_since_access,
                    evictions_since_write=sum(
                        1
                        for moment in evictions_at
                        if moment >= item.write_value_position
                    ),
                    is_contrast_anchor=item.index == anchor_index,
                )
            )
            continue

        item = payload
        occupancy_sum += len(occupants)
        occupancy_samples += 1

        free = [slot for slot in range(num_slots) if slot not in occupants]
        if free:
            slot = free[0]
        else:
            next_use = {
                candidate: _next_use(record.item, now)
                for candidate, record in occupants.items()
            }
            context = EvictionContext(
                now=now, occupants=occupants, rng=rng, next_use=next_use
            )
            slot = policy.choose_victim(context)
            victim = occupants[slot]
            evictions += 1
            evictions_at.append(now)
            lifetime_sum += now - victim.inserted_at
            lifetime_count += 1
            victim_needed = next_use[slot] is not None
            if victim_needed:
                future_needed_evicted += 1
            else:
                dead_evicted += 1

            dead_available = any(value is None for value in next_use.values())
            if dead_available:
                if victim_needed:
                    wrong_protection += 1
                else:
                    correct_protection += 1

            # Controlled A/B contrast: the oldest still-needed resident item
            # versus the most recently inserted item that is never needed again.
            needed = [
                candidate for candidate, value in next_use.items() if value is not None
            ]
            dead = [
                candidate for candidate, value in next_use.items() if value is None
            ]
            if needed and dead:
                oldest_needed = min(needed, key=lambda c: occupants[c].inserted_at)
                newest_dead = max(dead, key=lambda c: occupants[c].inserted_at)
                if occupants[oldest_needed].inserted_at < occupants[newest_dead].inserted_at:
                    contrast_decisions += 1
                    if slot != oldest_needed:
                        contrast_correct += 1

            del slot_of_item[victim.item.index]
            del occupants[slot]

        occupants[slot] = SlotRecord(item=item, inserted_at=now, last_access=now)
        slot_of_item[item.index] = slot

    dead_retained = sum(1 for record in occupants.values() if record.item.is_never)
    return EpisodeTrace(
        outcomes=outcomes,
        evictions=evictions,
        future_needed_evicted=future_needed_evicted,
        dead_evicted=dead_evicted,
        dead_retained_at_end=dead_retained,
        live_retained_at_end=len(occupants) - dead_retained,
        wrong_protection=wrong_protection,
        correct_protection=correct_protection,
        contrast_decisions=contrast_decisions,
        contrast_correct=contrast_correct,
        occupancy_sum=occupancy_sum,
        occupancy_samples=occupancy_samples,
        lifetime_sum=lifetime_sum,
        lifetime_count=lifetime_count,
    )
