"""PHL-DAM-004 shared episode generator: bounded-memory pressure streams.

One generator serves both stages. PHL-DAM-004A consumes only the event
schedule (exact content store, so eviction quality is the sole variable);
PHL-DAM-004B additionally consumes the tokenised stream.

Episodes come from a pure ``random.Random`` stream, so a given
(base seed, episode index) yields byte-identical episodes regardless of torch
RNG state. Every policy and every arm therefore sees exactly the same
bindings, queries, positions and delays.

Design constraints this module enforces:

* Writes always outnumber slots, so eviction is forced.
* Only a subset of written bindings is ever queried; the rest are ``never``.
* Queries do NOT consume the binding, so an item can be queried repeatedly and
  LRU is a genuinely strong baseline.
* Every latent use-class is placeable from every legal write position, so
  class identity is not recoverable from position.
* The write-time ``tag`` token carries partial, calibrated information about
  the latent use-class and is drawn independently of write order, so recency
  policies cannot exploit it.
"""

from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass, field

import torch
from torch import Tensor


PAD = 0
BOS = 1
WRITE = 2
QUERY = 3
FILL = 4
TAG_START = 5
NUM_TAGS = 8
KEY_START = TAG_START + NUM_TAGS
NUM_KEYS = 64
VALUE_START = KEY_START + NUM_KEYS
NUM_VALUES = 10
VOCAB_SIZE = VALUE_START + NUM_VALUES

NUM_SLOTS = 8

# ---------------------------------------------------------------------------
# Scale profiles
# ---------------------------------------------------------------------------
# "full" is the PHL-DAM-004 brief as written: 456 tokens, 32-256 token delays,
# 16/24/32 writes. "compact" is the same task shape at 176 tokens - Stage B's
# length - with delays and write counts scaled down proportionally.
#
# The compact profile exists because of a measured fact, not a preference: at
# 456 tokens the content controller does not reach its recall breakthrough
# within 600 updates, so every arm sits at chance and no eviction policy can be
# compared. At 176 tokens the identical model breaks through by step ~200. The
# compact profile keeps everything structurally load-bearing - eight slots,
# over-subscription, the noisy tag cue, never-queried distractors, repeat
# queries and the contrast subset - and reduces only length and magnitude.

SCALE_PROFILES = {
    "full": {
        "sequence_length": 456,
        "write_region_end": 190,
        "min_delay": 32,
        "max_delay": 256,
        "first_use_delay": {0: (32, 56), 1: (57, 96), 2: (97, 160), 3: (161, 256), 4: (40, 110)},
        "persistent_gap": (48, 110),
        "delay_bins": (("32-79", 32, 79), ("80-159", 80, 159), ("160-256", 160, 256)),
        "pressure_levels": (16, 24, 32),
        "query_budget": {
            "canonical": {8: (4, 7), 16: (7, 11), 24: (13, 19), 32: (18, 26)},
            "spec": {8: (4, 8), 16: (4, 8), 24: (4, 8), 32: (4, 8)},
        },
    },
    "compact": {
        "sequence_length": 176,
        "write_region_end": 64,
        "min_delay": 29,
        "max_delay": 104,
        "first_use_delay": {0: (29, 44), 1: (45, 62), 2: (63, 82), 3: (83, 104), 4: (32, 58)},
        "persistent_gap": (28, 46),
        "delay_bins": (("29-49", 29, 49), ("50-74", 50, 74), ("75-104", 75, 104)),
        "pressure_levels": (8, 12, 16),
        "query_budget": {
            "canonical": {8: (5, 8), 12: (8, 12), 16: (11, 16)},
            "spec": {8: (4, 8), 12: (4, 8), 16: (4, 8)},
        },
    },
}

SCALE = "full"
SEQUENCE_LENGTH = 456
WRITE_REGION_END = 190
WRITE_EVENT_TOKENS = 4
QUERY_EVENT_TOKENS = 3

MIN_DELAY = 32
MAX_DELAY = 256
FIRST_USE_DELAY: dict[int, tuple[int, int]] = {}
PERSISTENT_GAP = (48, 110)
PERSISTENT_USES = (2, 3)
DELAY_BINS: tuple[tuple[str, int, int], ...] = ()
PRESSURE_LEVELS: tuple[int, ...] = (16, 24, 32)
QUERY_BUDGET: dict[str, dict[int, tuple[int, int]]] = {}
QUERY_CONDITIONS = ("canonical", "spec")

INTERVENING_WRITE_BINS = (
    ("0-4", 0, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17+", 17, 10_000),
)

NEAR, SHORT, MEDIUM, FAR, PERSISTENT, NEVER = range(6)
CLASS_NAMES = ("near", "short", "medium", "far", "persistent", "never")
NUM_CLASSES = len(CLASS_NAMES)

# Live-class mixture. Biased towards the longer horizons on purpose:
# concurrency of future-needed items, not query count, creates capacity
# pressure.
LIVE_CLASS_WEIGHTS = {
    NEAR: 0.08,
    SHORT: 0.14,
    MEDIUM: 0.26,
    FAR: 0.34,
    PERSISTENT: 0.18,
}

CONTRAST_FRACTION = 0.15


def dilated_profile(factor: float, base: str = "compact") -> dict:
    """Stretch a profile in time by ``factor``, holding the task itself fixed.

    Writes, query budgets, class mixture and cue strength are untouched; only
    positions and delays scale. This isolates temporal dilation as a single
    factor, which is what separates "the task got harder" from "the same task
    became unlearnable when stretched".
    """
    profile = SCALE_PROFILES[base]

    def scale(pair):
        return (max(1, round(pair[0] * factor)), max(2, round(pair[1] * factor)))

    delays = {key: scale(value) for key, value in profile["first_use_delay"].items()}
    minimum = min(low for low, _ in delays.values())
    maximum = max(high for _, high in delays.values())
    write_end = round(profile["write_region_end"] * factor)
    length = write_end + maximum + QUERY_EVENT_TOKENS + WRITE_EVENT_TOKENS + 8
    return {
        "sequence_length": int(length),
        "write_region_end": int(write_end),
        "min_delay": int(minimum),
        "max_delay": int(maximum),
        "first_use_delay": delays,
        "persistent_gap": scale(profile["persistent_gap"]),
        "delay_bins": (
            ("short", minimum, delays[SHORT][1]),
            ("medium", delays[SHORT][1] + 1, delays[MEDIUM][1]),
            ("long", delays[MEDIUM][1] + 1, maximum),
        ),
        "pressure_levels": profile["pressure_levels"],
        "query_budget": {k: dict(v) for k, v in profile["query_budget"].items()},
    }


for _factor, _name in ((1.0, "dilate10"), (1.5, "dilate15"), (2.0, "dilate20"), (2.6, "dilate26")):
    SCALE_PROFILES[_name] = dilated_profile(_factor)

# ---------------------------------------------------------------------------
# PHL-DAM-004D write-pressure ladder
# ---------------------------------------------------------------------------
# One profile, one sequence length, one delay distribution, one query budget.
# The ONLY thing that varies across its pressure levels is how many write
# events an episode contains. Everything a write count would normally drag
# along - sequence length, delay range, number of queried items - is pinned, so
# a difference between levels is attributable to write load rather than to the
# longer sequence that a higher write count would otherwise require.
#
# Live (queried) items are held at a fixed small number, below slot capacity,
# so rising pressure comes purely from never-queried distractor writes
# competing for the same eight slots.
WRITE_PRESSURE_LEVELS = (8, 12, 16, 20, 24, 28, 32)
WRITE_PRESSURE_QUERY_BUDGET = (6, 9)

SCALE_PROFILES["pressure"] = {
    "sequence_length": 456,
    "write_region_end": 190,
    "min_delay": 32,
    "max_delay": 256,
    "first_use_delay": {
        NEAR: (32, 56),
        SHORT: (57, 96),
        MEDIUM: (97, 160),
        FAR: (161, 256),
        PERSISTENT: (40, 110),
    },
    "persistent_gap": (48, 110),
    "delay_bins": (("32-79", 32, 79), ("80-159", 80, 159), ("160-256", 160, 256)),
    "pressure_levels": WRITE_PRESSURE_LEVELS,
    "query_budget": {
        "canonical": {w: WRITE_PRESSURE_QUERY_BUDGET for w in WRITE_PRESSURE_LEVELS},
        "spec": {w: WRITE_PRESSURE_QUERY_BUDGET for w in WRITE_PRESSURE_LEVELS},
    },
}


def set_scale(name: str) -> None:
    """Select a scale profile. Must be called before generating episodes."""
    if name not in SCALE_PROFILES:
        raise ValueError(f"unknown scale: {name}")
    profile = SCALE_PROFILES[name]
    globals().update(
        SCALE=name,
        SEQUENCE_LENGTH=profile["sequence_length"],
        WRITE_REGION_END=profile["write_region_end"],
        MIN_DELAY=profile["min_delay"],
        MAX_DELAY=profile["max_delay"],
        FIRST_USE_DELAY=dict(profile["first_use_delay"]),
        PERSISTENT_GAP=profile["persistent_gap"],
        DELAY_BINS=profile["delay_bins"],
        PRESSURE_LEVELS=profile["pressure_levels"],
        QUERY_BUDGET={k: dict(v) for k, v in profile["query_budget"].items()},
    )


set_scale("full")

TAG_PRIMARY_PROBABILITY = 0.65


def tag_distribution(primary: float = TAG_PRIMARY_PROBABILITY) -> Tensor:
    """P(tag | class): primary tag ``c`` for class ``c``, rest spread evenly."""
    table = torch.full((NUM_CLASSES, NUM_TAGS), (1.0 - primary) / (NUM_TAGS - 1))
    for class_index in range(NUM_CLASSES):
        table[class_index, class_index] = primary
    return table


TAG_TABLE = tag_distribution()
TAG_CUMULATIVE = TAG_TABLE.cumsum(dim=-1).tolist()


def bayes_tag_accuracy(class_prior: Tensor, table: Tensor = TAG_TABLE) -> float:
    """Best achievable class accuracy from the tag alone, given a class prior."""
    joint = class_prior[:, None] * table
    return float(joint.max(dim=0).values.sum())


def bayes_live_auroc(class_prior: Tensor, table: Tensor = TAG_TABLE) -> float:
    """Best achievable AUROC for future-needed vs never-needed from the tag."""
    never_prior = float(class_prior[NEVER])
    live_prior = 1.0 - never_prior
    if live_prior <= 0.0 or never_prior <= 0.0:
        return float("nan")
    live_class_prior = class_prior.clone()
    live_class_prior[NEVER] = 0.0
    live_class_prior = live_class_prior / live_class_prior.sum()
    live_tag = (live_class_prior[:, None] * table).sum(dim=0)
    never_tag = table[NEVER]
    ratio = live_tag / never_tag.clamp_min(1e-12)
    order = torch.argsort(ratio, descending=True)
    concordant = 0.0
    for rank, tag in enumerate(order.tolist()):
        below = float(never_tag[order[rank + 1 :]].sum())
        concordant += float(live_tag[tag]) * (below + 0.5 * float(never_tag[tag]))
    return concordant


@dataclass
class QueryEvent:
    item_index: int
    occurrence: int
    marker_position: int
    key_position: int
    target_position: int
    delay: int
    intervening_writes: int = 0


@dataclass
class Item:
    index: int
    key_token: int
    value_token: int
    tag_token: int
    use_class: int
    write_start: int
    write_value_position: int
    query_key_positions: list[int] = field(default_factory=list)

    @property
    def is_never(self) -> bool:
        return not self.query_key_positions


@dataclass
class Episode:
    writes: int
    query_condition: str
    is_contrast: bool
    items: list[Item]
    queries: list[QueryEvent]
    tokens: Tensor
    max_concurrent_live: int
    seed: int


def _spaced_positions(
    rng: random.Random, count: int, low: int, high: int, span: int
) -> list[int]:
    """``count`` sorted starts in [low, high] with pairwise gap >= ``span``.

    Uniform over all valid configurations: draw distinct values from the
    shrunken range, sort, then re-expand.
    """
    reduced_high = high - (count - 1) * (span - 1)
    if reduced_high < low:
        raise ValueError("cannot place events without overlap")
    draws = sorted(rng.sample(range(low, reduced_high + 1), count))
    return [value + index * (span - 1) for index, value in enumerate(draws)]


def _claim_window(
    occupied: list[bool],
    target: int,
    length: int,
    lowest: int,
    highest: int,
) -> int | None:
    """Nearest free ``length``-token window to ``target`` inside [lowest, highest].

    The bounds are hard: they are what keeps every realised delay causal and
    inside [MIN_DELAY, MAX_DELAY]. A window that cannot be placed within them
    is dropped rather than nudged out of range.
    """
    lowest = max(lowest, 1)
    highest = min(highest, len(occupied) - length)
    if highest < lowest:
        return None
    limit = highest - lowest
    for offset in range(0, limit + 1):
        for candidate in (target + offset, target - offset):
            if candidate < lowest or candidate > highest:
                continue
            if not any(occupied[candidate : candidate + length]):
                for position in range(candidate, candidate + length):
                    occupied[position] = True
                return candidate
    return None


def _sample_live_class(rng: random.Random) -> int:
    classes = list(LIVE_CLASS_WEIGHTS)
    weights = [LIVE_CLASS_WEIGHTS[name] for name in classes]
    return rng.choices(classes, weights=weights, k=1)[0]


def _sample_tag(rng: random.Random, use_class: int) -> int:
    threshold = rng.random()
    for tag_index, cumulative in enumerate(TAG_CUMULATIVE[use_class]):
        if threshold < cumulative:
            return tag_index
    return NUM_TAGS - 1


def _use_offsets(rng: random.Random, use_class: int) -> list[int]:
    """Delays, in tokens from the write-value position, for one live item.

    Cumulative offsets are capped at ``MAX_DELAY``: a persistent item drops the
    repeat uses that would fall outside the preregistered delay range rather
    than reporting a delay the protocol does not claim to cover.
    """
    low, high = FIRST_USE_DELAY[use_class]
    offsets = [rng.randint(low, high)]
    if use_class == PERSISTENT:
        uses = rng.randint(*PERSISTENT_USES)
        for _ in range(uses - 1):
            candidate = offsets[-1] + rng.randint(*PERSISTENT_GAP)
            if candidate > MAX_DELAY:
                break
            offsets.append(candidate)
    return offsets


def episode_seed(base_seed: int, episode_index: int) -> int:
    return (base_seed * 1_000_003 + episode_index * 10_007 + 7) % (2**31 - 1)


def generate_episode(
    base_seed: int,
    episode_index: int,
    writes: int,
    query_condition: str = "canonical",
    contrast_fraction: float = CONTRAST_FRACTION,
) -> Episode:
    seed = episode_seed(base_seed, episode_index)
    rng = random.Random(seed)

    is_contrast = rng.random() < contrast_fraction

    write_starts = _spaced_positions(
        rng, writes, 1, WRITE_REGION_END, WRITE_EVENT_TOKENS
    )
    key_tokens = rng.sample(range(KEY_START, KEY_START + NUM_KEYS), writes)

    low, high = QUERY_BUDGET[query_condition][writes]
    query_budget = rng.randint(low, high)

    order = list(range(writes))
    rng.shuffle(order)
    if is_contrast:
        # Guarantee one old-but-future-relevant anchor among the first writes
        # so the controlled A/B contrast has power.
        anchor = rng.choice((0, 1))
        order.remove(anchor)
        order.insert(0, anchor)

    classes = [NEVER] * writes
    offsets: dict[int, list[int]] = {}
    remaining = query_budget
    for position_in_order, item_index in enumerate(order):
        if remaining <= 0:
            break
        if is_contrast and position_in_order == 0:
            use_class = rng.choice((FAR, PERSISTENT))
        else:
            use_class = _sample_live_class(rng)
        item_offsets = _use_offsets(rng, use_class)
        if len(item_offsets) > remaining:
            item_offsets = item_offsets[:remaining]
        classes[item_index] = use_class
        offsets[item_index] = item_offsets
        remaining -= len(item_offsets)

    if is_contrast:
        # Force a run of dead writes after the anchor so memory fills with
        # never-queried items before the anchor is needed again.
        anchor = order[0]
        forced_dead = 0
        for item_index in range(anchor + 1, writes):
            if forced_dead >= 10:
                break
            if item_index in offsets:
                classes[item_index] = NEVER
                offsets.pop(item_index)
            forced_dead += 1

    occupied = [False] * SEQUENCE_LENGTH
    occupied[0] = True
    for start in write_starts:
        for position in range(start, start + WRITE_EVENT_TOKENS):
            occupied[position] = True

    items: list[Item] = []
    for item_index in range(writes):
        use_class = classes[item_index]
        items.append(
            Item(
                index=item_index,
                key_token=key_tokens[item_index],
                value_token=rng.randrange(VALUE_START, VALUE_START + NUM_VALUES),
                tag_token=TAG_START + _sample_tag(rng, use_class),
                use_class=use_class,
                write_start=write_starts[item_index],
                write_value_position=write_starts[item_index] + 3,
            )
        )

    queries: list[QueryEvent] = []
    for item_index, item_offsets in sorted(offsets.items()):
        item = items[item_index]
        previous_marker = -1
        for occurrence, offset in enumerate(item_offsets):
            # marker + 1 is the query-key position, so the delay bounds become
            # marker bounds. Repeat uses must also stay strictly ordered.
            lowest = max(
                item.write_value_position + MIN_DELAY - 1, previous_marker + 1
            )
            highest = min(
                item.write_value_position + MAX_DELAY - 1,
                SEQUENCE_LENGTH - QUERY_EVENT_TOKENS,
            )
            marker = _claim_window(
                occupied,
                item.write_value_position + offset - 1,
                QUERY_EVENT_TOKENS,
                lowest,
                highest,
            )
            if marker is None:
                continue
            previous_marker = marker
            key_position = marker + 1
            queries.append(
                QueryEvent(
                    item_index=item_index,
                    occurrence=occurrence,
                    marker_position=marker,
                    key_position=key_position,
                    target_position=marker + 2,
                    delay=key_position - item.write_value_position,
                )
            )
            item.query_key_positions.append(key_position)

    for item in items:
        item.query_key_positions.sort()
        if not item.query_key_positions:
            item.use_class = NEVER

    queries.sort(key=lambda event: event.key_position)
    for index, event in enumerate(queries):
        event.occurrence = sum(
            1 for earlier in queries[:index] if earlier.item_index == event.item_index
        )

    write_value_positions = sorted(item.write_value_position for item in items)
    for event in queries:
        item = items[event.item_index]
        event.intervening_writes = (
            bisect_left(write_value_positions, event.key_position)
            - bisect_left(write_value_positions, item.write_value_position)
            - 1
        )

    return Episode(
        writes=writes,
        query_condition=query_condition,
        is_contrast=is_contrast,
        items=items,
        queries=queries,
        tokens=tokenise(items, queries),
        max_concurrent_live=max_concurrent_live(items),
        seed=seed,
    )


def tokenise(items: list[Item], queries: list[QueryEvent]) -> Tensor:
    tokens = torch.full((SEQUENCE_LENGTH,), FILL, dtype=torch.long)
    tokens[0] = BOS
    for item in items:
        start = item.write_start
        tokens[start] = WRITE
        tokens[start + 1] = item.tag_token
        tokens[start + 2] = item.key_token
        tokens[start + 3] = item.value_token
    for event in queries:
        item = items[event.item_index]
        tokens[event.marker_position] = QUERY
        tokens[event.key_position] = item.key_token
        tokens[event.target_position] = item.value_token
    return tokens


def max_concurrent_live(items: list[Item]) -> int:
    """Peak count of written-but-still-needed bindings.

    An item is live from its write-value position until its final query. This
    is the quantity that creates capacity pressure; query count alone is not.
    """
    changes: list[tuple[int, int]] = []
    for item in items:
        if not item.query_key_positions:
            continue
        changes.append((item.write_value_position, 1))
        changes.append((item.query_key_positions[-1] + 1, -1))
    if not changes:
        return 0
    changes.sort()
    running = 0
    peak = 0
    for _, delta in changes:
        running += delta
        peak = max(peak, running)
    return peak


def generate_episodes(
    base_seed: int,
    count: int,
    writes: int,
    query_condition: str = "canonical",
    start_index: int = 0,
) -> list[Episode]:
    return [
        generate_episode(base_seed, start_index + index, writes, query_condition)
        for index in range(count)
    ]


def stack_tokens(episodes: list[Episode]) -> Tensor:
    return torch.stack([episode.tokens for episode in episodes])


def delay_bin(delay: int) -> str:
    for name, low, high in DELAY_BINS:
        if low <= delay <= high:
            return name
    return "out-of-range"


def intervening_bin(count: int) -> str:
    for name, low, high in INTERVENING_WRITE_BINS:
        if low <= count <= high:
            return name
    return "out-of-range"
