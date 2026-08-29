"""PHL-DAM-004B: learned temporal lease inside a finite-capacity PHL-DAM.

The content mechanism is the zero-write-budget Stage B design: key path, value
path, soft content addressing, query retrieval and the simplified PHL
recurrence are unchanged in form. Two changes support bounded memory:

1. Allocation becomes one-hot - the first free slot, or the argmin of an
   eviction score when the memory is full - with a straight-through estimator
   so gradients still reach the key/value/write paths and the learned scorer.
2. A per-slot temporal lease ``L_j`` is carried *alongside* K_j and V_j. K and
   V never move between slots; only the lease evolves in time.

No arm receives next-use labels, at training or at evaluation. The oracle arm
is the only component with future information and it is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import phl_dam_pressure_task as task

torch.set_num_threads(1)


LEASE_BIN_NAMES = ("due", "near", "short", "medium", "far", "never")
NUM_LEASE_BINS = len(LEASE_BIN_NAMES)
DUE_LEAK = 0.015

LEARNED_ARMS = ("content_only", "static_priority", "phl_lease", "learned_utility")
POLICY_ARMS = ("random", "fifo", "lru", "oracle")
ALL_ARMS = POLICY_ARMS + LEARNED_ARMS

RESIDENCY_WRITE_THRESHOLD = 0.5
ACCESS_READ_THRESHOLD = 0.5
INFINITY = 10**9
TIMING_IGNORE_INDEX = -100

# How hard the eviction score pulls the allocation softmax once the slots are
# full: one standard deviation of score moves the pre-temperature logit by this
# much, i.e. ten times that after the 0.10 temperature.
EVICTION_SCORE_WEIGHT = 1.0

# Numerical constants of the content path, named so they can be ablated.
# Defaults reproduce Stage B exactly; changing one is an experiment, not a
# silent edit.
OCCUPANCY_LOG_EPSILON = 1e-6
READ_TEMPERATURE = 0.10
MERGE_TEMPERATURE = 0.10
MERGE_SHARPNESS = 12.0
# Floor on the slot-score spread used to standardise the eviction score.
# With a bare 1e-6 epsilon the Jacobian of the standardisation grows as
# 1/spread, reaching ~9e5 when all eight slots score alike - which happens on
# roughly 1 timestep in 450 and was the source of PHL-DAM's intermittent
# gradient explosions (localised by bisection in
# phl_dam_gradient_trigger.py). Clamping bounds that Jacobian at ~8.75 and
# leaves the eviction ordering unchanged wherever the spread is informative:
# when it is not, the score should contribute nothing rather than amplified
# noise. Set to 0.0 to recover the original behaviour exactly.
EVICTION_SCORE_SPREAD_FLOOR = 0.1


def _slot_unit(mixed_keys: Tensor, occupancy: Tensor) -> Tensor:
    """Unit-normalise occupied slot keys; leave empty slots exactly as they are.

    An empty slot holds an exactly zero key, and ``F.normalize`` back-propagates
    a Jacobian of order 1e12 through it. Under the slot recurrence those
    magnitudes overflow float32 within a few dozen steps and the loss is NaN
    from the first update. An empty slot's mixed key is exactly zero, so taking
    the un-normalised branch for it is forward-identical to Stage B while
    bounding its Jacobian at one. Occupied slots are normalised as before, which
    is what lets the recall loss train the addressing path from a barely-open
    write gate.
    """
    occupied = (occupancy > 1e-6)[:, :, None]
    divisor = mixed_keys.pow(2).sum(-1, keepdim=True).add(1e-8).sqrt()
    return torch.where(occupied, mixed_keys / divisor, mixed_keys)


def lease_bin_edges() -> tuple[tuple[int, int], ...]:
    """Lease horizons derived from the *active* task scale.

    These were once hardcoded to the full-scale delay ranges, which left the
    transport operator running roughly 2.5x too slow whenever the compact
    profile was active - a defect that penalised the transported-lease arm
    specifically. Deriving them from ``task.FIRST_USE_DELAY`` keeps each horizon
    equal to the delay band it represents at whatever scale is selected, so a
    lease initialised in the "far" horizon reaches "due" exactly as its query
    comes due.
    """
    return (
        (0, task.MIN_DELAY - 1),
        task.FIRST_USE_DELAY[task.NEAR],
        task.FIRST_USE_DELAY[task.SHORT],
        task.FIRST_USE_DELAY[task.MEDIUM],
        task.FIRST_USE_DELAY[task.FAR],
    )


def delay_to_lease_bin(delay: int | None) -> int:
    """Bin a true next-use delay; ``None`` (never queried) is the last horizon."""
    if delay is None:
        return NUM_LEASE_BINS - 1
    for index, (low, high) in enumerate(lease_bin_edges()):
        if low <= delay <= high:
            return index
    return NUM_LEASE_BINS - 1


def build_lease_transport(edges: tuple[tuple[int, int], ...] | None = None) -> Tensor:
    """One-token-step interval-overlap transport over the lease horizons.

    Each horizon is a delay interval. After one token the interval shifts down
    by one, so a ``1 / width`` slice crosses into the horizon below and the
    remainder stays. ``due`` is the imminent-use horizon and leaks slowly into
    ``never`` so an overdue lease decays instead of protecting forever. The
    centre-interpolation fallback of the validated hybrid operator is not
    exercised at a one-token step, because every shifted interval still
    overlaps its own horizon.
    """
    edges = edges if edges is not None else lease_bin_edges()
    transport = torch.zeros(NUM_LEASE_BINS, NUM_LEASE_BINS)
    for index, (low, high) in enumerate(edges):
        width = float(high - low + 1)
        if index == 0:
            transport[index, index] = 1.0 - DUE_LEAK
            transport[index, NUM_LEASE_BINS - 1] = DUE_LEAK
        else:
            transport[index, index] = (width - 1.0) / width
            transport[index, index - 1] = 1.0 / width
    transport[NUM_LEASE_BINS - 1, NUM_LEASE_BINS - 1] = 1.0
    return transport


@dataclass
class Batch:
    tokens: Tensor
    recall_mask: Tensor
    query_key_positions: Tensor
    query_valid: Tensor
    query_key_tokens: Tensor
    query_target_tokens: Tensor
    delays: Tensor
    intervening: Tensor
    occurrence: Tensor
    contrast_anchor: Tensor
    key_use_positions: Tensor
    timing_label: Tensor
    writes: int
    condition: str


def pack_batch(episodes: list[task.Episode], device: torch.device) -> Batch:
    batch_size = len(episodes)
    max_queries = max(len(episode.queries) for episode in episodes)
    tokens = torch.stack([episode.tokens for episode in episodes]).to(device)
    recall_mask = torch.zeros_like(tokens, dtype=torch.bool)

    positions = torch.zeros(batch_size, max_queries, dtype=torch.long)
    valid = torch.zeros(batch_size, max_queries, dtype=torch.bool)
    key_tokens = torch.zeros(batch_size, max_queries, dtype=torch.long)
    target_tokens = torch.zeros(batch_size, max_queries, dtype=torch.long)
    delays = torch.zeros(batch_size, max_queries, dtype=torch.long)
    intervening = torch.zeros(batch_size, max_queries, dtype=torch.long)
    occurrence = torch.zeros(batch_size, max_queries, dtype=torch.long)
    anchor = torch.zeros(batch_size, max_queries, dtype=torch.bool)

    max_uses = 1
    for episode in episodes:
        for item in episode.items:
            max_uses = max(max_uses, len(item.query_key_positions))
    key_use_positions = torch.full(
        (batch_size, task.NUM_KEYS, max_uses), INFINITY, dtype=torch.long
    )

    timing_label = torch.full(
        (batch_size, task.SEQUENCE_LENGTH), TIMING_IGNORE_INDEX, dtype=torch.long
    )

    for row, episode in enumerate(episodes):
        anchor_index = None
        if episode.is_contrast:
            live = [item for item in episode.items if not item.is_never]
            if live:
                anchor_index = min(
                    live, key=lambda item: item.write_value_position
                ).index
        for column, query in enumerate(episode.queries):
            item = episode.items[query.item_index]
            recall_mask[row, query.target_position] = True
            positions[row, column] = query.key_position
            valid[row, column] = True
            key_tokens[row, column] = item.key_token
            target_tokens[row, column] = item.value_token
            delays[row, column] = query.delay
            intervening[row, column] = query.intervening_writes
            occurrence[row, column] = query.occurrence
            anchor[row, column] = item.index == anchor_index
        for item in episode.items:
            slot = item.key_token - task.KEY_START
            for use_index, position in enumerate(item.query_key_positions):
                key_use_positions[row, slot, use_index] = position
            # Timing supervision label: the true next-use horizon of the
            # binding written here. Training only; never available at eval.
            first = item.query_key_positions[0] if item.query_key_positions else None
            delay = None if first is None else first - item.write_value_position
            timing_label[row, item.write_value_position] = delay_to_lease_bin(delay)

    return Batch(
        tokens=tokens,
        recall_mask=recall_mask.to(device),
        query_key_positions=positions.to(device),
        query_valid=valid.to(device),
        query_key_tokens=key_tokens.to(device),
        query_target_tokens=target_tokens.to(device),
        delays=delays.to(device),
        intervening=intervening.to(device),
        occurrence=occurrence.to(device),
        contrast_anchor=anchor.to(device),
        key_use_positions=key_use_positions.to(device),
        timing_label=timing_label.to(device),
        writes=episodes[0].writes,
        condition=episodes[0].query_condition,
    )


@dataclass
class DAMState:
    keys: Tensor
    values: Tensor
    occupancy: Tensor
    leases: Tensor
    inserted_at: Tensor
    last_access: Tensor
    access_count: Tensor
    owner_token: Tensor
    insert_write_strength: Tensor


@dataclass
class ModelState:
    phl: Tensor
    dam: DAMState
    step_index: int


@dataclass
class StepTrace:
    write_strength: Tensor
    read_strength: Tensor
    attention: Tensor
    allocation: Tensor
    eviction_score: Tensor
    evicted: Tensor
    victim_slot: Tensor
    lease_before: Tensor
    allocation_entropy: Tensor
    allocation_max: Tensor
    allocation_margin: Tensor
    occupied_count: Tensor
    readback_error: Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class PHLDAMLease(nn.Module):
    """Finite-capacity PHL-DAM with an optional per-slot temporal lease."""

    def __init__(
        self,
        arm: str = "phl_lease",
        d_model: int = 64,
        horizons: int = 4,
        horizon_width: int = 16,
        num_slots: int = task.NUM_SLOTS,
        d_key: int = 24,
        d_value: int = 24,
        eviction_temperature: float = 0.5,
        backbone: str = "phl",
    ) -> None:
        super().__init__()
        if horizons * horizon_width != d_model:
            raise ValueError("horizons * horizon_width must equal d_model")
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown arm: {arm}")
        if backbone not in ("phl", "ssm", "none"):
            raise ValueError(f"unknown backbone: {backbone}")
        self.backbone = backbone
        self.arm = arm
        self.d_model = d_model
        self.horizons = horizons
        self.horizon_width = horizon_width
        self.num_slots = num_slots
        self.d_key = d_key
        self.d_value = d_value
        self.eviction_temperature = eviction_temperature

        self.token_embedding = nn.Embedding(task.VOCAB_SIZE, d_model)
        self.context_encoder = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Tanh())

        self.phl_input = nn.Linear(d_model, d_model, bias=False)
        self.phl_norm = nn.RMSNorm((horizons, horizon_width))
        self.phl_readout = nn.Linear(d_model, d_model, bias=False)
        transport = torch.zeros(horizons, horizons)
        for horizon in range(horizons - 1):
            transport[horizon, horizon] = 0.65
            transport[horizon, horizon + 1] = 0.35
        transport[-1, -1] = 0.98
        self.register_buffer("phl_transport", transport)
        if backbone == "ssm":
            # Replacement for the hand-designed transport lattice: one
            # learned decay per channel instead of fixed 0.65/0.35/0.98
            # mixing. decay = exp(-exp(rate)) lies in [0, 1] for any
            # parameter value, so this recurrence provably cannot amplify -
            # unlike the slot path, whose unbounded Jacobian caused this
            # model's gradient explosions. Rates are log-spaced so the
            # channels span short and long horizons at initialisation, the
            # same coverage PHL got by construction but learned rather than
            # imposed.
            rates = torch.exp(
                torch.linspace(math.log(1e-3), math.log(1.0), d_model)
            )
            self.ssm_log_rate = nn.Parameter(torch.log(rates))
        self.register_buffer("lease_transport", build_lease_transport())

        self.key_projection = nn.Linear(d_model, d_key, bias=False)
        self.value_projection = nn.Linear(d_model, d_value, bias=False)
        self.query_projection = nn.Linear(d_model, d_key, bias=False)
        self.write_gate = nn.Linear(d_model, 1)
        self.memory_projection = nn.Linear(d_value, d_model, bias=False)
        self.read_gate = nn.Linear(d_model + d_value + 1, 1)

        self.output_norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, task.VOCAB_SIZE)

        # Lease head: shared, parameter-identical, by static_priority and
        # phl_lease. The only structural difference between those two arms is
        # whether the lease is transported through the horizon lattice.
        # Every learned arm carries the timing head, so equal timing
        # supervision reaches every arm's shared representation. Only the
        # lease arms let its output enter the eviction score.
        if arm in LEARNED_ARMS:
            self.lease_head = nn.Linear(d_model, NUM_LEASE_BINS)
        if arm in ("static_priority", "phl_lease"):
            # Small random init, not zeros: identical scores across slots would
            # make argmin always pick slot 0 and freeze the other seven.
            self.lease_readout = nn.Parameter(torch.randn(NUM_LEASE_BINS) * 0.1)
            self.eviction_recency = nn.Parameter(torch.randn(3) * 0.1)
            self.eviction_bias = nn.Parameter(torch.zeros(1))
        if arm == "content_only":
            self.content_scorer = nn.Sequential(
                nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 1)
            )
        if arm == "learned_utility":
            self.utility_scorer = nn.Sequential(
                nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)
            )

        self.register_buffer("slot_bias", torch.linspace(0.14, 0.0, num_slots))
        nn.init.constant_(self.write_gate.bias, -3.0)
        nn.init.constant_(self.read_gate.bias, 1.0)

    @property
    def uses_lease(self) -> bool:
        return self.arm in ("static_priority", "phl_lease")

    @property
    def transports_lease(self) -> bool:
        return self.arm == "phl_lease"

    def init_state(self, batch_size: int, device: torch.device) -> ModelState:
        zeros = lambda *shape: torch.zeros(*shape, device=device)
        return ModelState(
            phl=zeros(batch_size, self.horizons, self.horizon_width),
            dam=DAMState(
                keys=zeros(batch_size, self.num_slots, self.d_key),
                values=zeros(batch_size, self.num_slots, self.d_value),
                occupancy=zeros(batch_size, self.num_slots),
                leases=zeros(batch_size, self.num_slots, NUM_LEASE_BINS),
                inserted_at=zeros(batch_size, self.num_slots),
                last_access=zeros(batch_size, self.num_slots),
                access_count=zeros(batch_size, self.num_slots),
                owner_token=torch.full(
                    (batch_size, self.num_slots), -1, dtype=torch.long, device=device
                ),
                insert_write_strength=zeros(batch_size, self.num_slots),
            ),
            step_index=0,
        )

    def precompute(self, context: Tensor, previous: Tensor, current: Tensor) -> dict:
        """Projections that depend only on tokens, not on recurrent state.

        The candidate key/value, the query and the write gate are functions of
        the token window alone, so they can be evaluated for the whole
        sequence in four batched matmuls instead of 4*T small ones inside the
        recurrence. Only the read gate genuinely depends on state.
        """
        return {
            "candidate_key": F.normalize(self.key_projection(previous), dim=-1),
            "candidate_value": self.value_projection(current),
            "query": F.normalize(self.query_projection(current), dim=-1),
            "write_strength": torch.sigmoid(self.write_gate(context)).squeeze(-1),
            "lease_logits": (
                torch.softmax(self.lease_head(context), dim=-1)
                if self.uses_lease
                else None
            ),
        }

    def encode_features(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embedded = self.token_embedding(tokens)
        padding = torch.zeros(
            embedded.shape[0], 2, embedded.shape[2], device=embedded.device
        )
        padded = torch.cat([padding, embedded], dim=1)
        windows = torch.cat([padded[:, 0:-2], padded[:, 1:-1], padded[:, 2:]], dim=-1)
        context = self.context_encoder(windows)
        previous_tokens = torch.cat(
            [torch.zeros_like(embedded[:, :1]), embedded[:, :-1]], dim=1
        )
        return context, previous_tokens, embedded

    def eviction_score(
        self,
        state: ModelState,
        arm: str,
        rng: torch.Generator | None,
        slot_next_use: Tensor | None,
    ) -> Tensor:
        """Higher score means keep. The argmin is evicted."""
        dam = state.dam
        now = float(state.step_index)
        batch_size = dam.occupancy.shape[0]
        age = (now - dam.inserted_at) / 100.0
        staleness = (now - dam.last_access) / 100.0
        accesses = dam.access_count / 4.0

        if arm == "random":
            return torch.rand(
                batch_size, self.num_slots, device=dam.keys.device, generator=rng
            )
        if arm == "fifo":
            return dam.inserted_at.clone()
        if arm == "lru":
            return dam.last_access.clone()
        if arm == "oracle":
            assert slot_next_use is not None
            distance = (slot_next_use - now).clamp_min(0.0)
            never = slot_next_use >= INFINITY / 2
            return torch.where(
                never, torch.full_like(distance, -1.0), 1.0 / (distance + 1.0)
            )
        if arm == "content_only":
            features = torch.stack(
                [
                    dam.occupancy,
                    dam.keys.abs().mean(dim=-1),
                    dam.values.abs().mean(dim=-1),
                    age,
                    staleness,
                ],
                dim=-1,
            )
            return self.content_scorer(features).squeeze(-1)
        if arm == "learned_utility":
            features = torch.stack(
                [age, staleness, accesses, dam.insert_write_strength], dim=-1
            )
            return self.utility_scorer(features).squeeze(-1)

        lease_term = (dam.leases * self.lease_readout).sum(dim=-1)
        recency_term = (
            self.eviction_recency[0] * age
            + self.eviction_recency[1] * staleness
            + self.eviction_recency[2] * accesses
        )
        return lease_term + recency_term + self.eviction_bias

    def step(
        self,
        context: Tensor,
        previous_token: Tensor,
        current_token: Tensor,
        state: ModelState,
        arm: str,
        disable_retrieval: bool = False,
        rng: torch.Generator | None = None,
        slot_next_use: Tensor | None = None,
        current_token_ids: Tensor | None = None,
        previous_token_ids: Tensor | None = None,
        precomputed: dict | None = None,
        collect: bool = True,
    ) -> tuple[Tensor, ModelState, StepTrace]:
        batch_size = context.shape[0]
        dam = state.dam
        device = context.device

        if self.backbone == "phl":
            transported = torch.einsum("ij,bjd->bid", self.phl_transport, state.phl)
            injected = self.phl_input(context).view(
                batch_size, self.horizons, self.horizon_width
            )
            phl = self.phl_norm(transported + injected)
            phl_contribution = self.phl_readout(phl.flatten(1))
        elif self.backbone == "ssm":
            decay = torch.exp(-torch.exp(self.ssm_log_rate))  # in [0, 1]
            flat = state.phl.flatten(1)
            updated = decay * flat + self.phl_input(context)
            phl = self.phl_norm(
                updated.view(batch_size, self.horizons, self.horizon_width)
            )
            phl_contribution = self.phl_readout(phl.flatten(1))
        else:
            phl = state.phl
            phl_contribution = torch.zeros_like(context)

        # Lease transport happens before any decision that reads the lease, so
        # the lease is a strictly causal function of past writes.
        if self.uses_lease and self.transports_lease:
            leases = torch.einsum("ij,bnj->bni", self.lease_transport.T, dam.leases)
        else:
            leases = dam.leases
        dam = DAMState(
            keys=dam.keys,
            values=dam.values,
            occupancy=dam.occupancy,
            leases=leases,
            inserted_at=dam.inserted_at,
            last_access=dam.last_access,
            access_count=dam.access_count,
            owner_token=dam.owner_token,
            insert_write_strength=dam.insert_write_strength,
        )
        state = ModelState(phl=phl, dam=dam, step_index=state.step_index)

        if precomputed is not None:
            candidate_key = precomputed["candidate_key"]
            candidate_value = precomputed["candidate_value"]
            write_strength = precomputed["write_strength"]
        else:
            candidate_key = F.normalize(self.key_projection(previous_token), dim=-1)
            candidate_value = self.value_projection(current_token)
            write_strength = torch.sigmoid(self.write_gate(context)).squeeze(-1)

        occupied = dam.occupancy > 0.05
        similarity = torch.einsum("bd,bnd->bn", candidate_key, dam.keys)
        masked_similarity = similarity.masked_fill(~occupied, -torch.inf)
        any_occupied = occupied.any(dim=-1, keepdim=True)
        safe_similarity = torch.where(
            any_occupied, masked_similarity, torch.zeros_like(similarity)
        )
        merge_distribution = torch.softmax(safe_similarity / MERGE_TEMPERATURE, dim=-1)
        merge_distribution = torch.where(
            any_occupied, merge_distribution, torch.zeros_like(merge_distribution)
        )

        max_similarity = torch.where(
            any_occupied.squeeze(-1),
            safe_similarity.max(dim=-1).values,
            -torch.ones(batch_size, device=device),
        )
        merge_strength = torch.sigmoid(MERGE_SHARPNESS * (max_similarity - 0.72))

        # Allocation is Stage B's softmax over slots with one term added: the
        # arm's eviction score. While slots are free the occupancy term
        # dominates and behaviour is Stage B's exactly, which is what lets the
        # recall loss bootstrap the addressing path from a barely-open write
        # gate. Once every slot is occupied the occupancy term is flat and the
        # eviction score alone decides which slot is overwritten - that is the
        # eviction decision. The score is standardised across slots so that
        # timestamps, probabilities and learned logits are all comparable and
        # no arm gets an accidental temperature advantage.
        free = dam.occupancy < 0.5
        has_free = free.any(dim=-1)
        raw_score = self.eviction_score(state, arm, rng, slot_next_use)
        centred = raw_score - raw_score.mean(dim=-1, keepdim=True)
        spread = raw_score.std(dim=-1, keepdim=True)
        if EVICTION_SCORE_SPREAD_FLOOR > 0.0:
            spread = spread.clamp_min(EVICTION_SCORE_SPREAD_FLOOR)
        else:
            spread = spread + 1e-6
        keep_score = centred / spread

        allocation_score = (
            5.0 * (1.0 - dam.occupancy)
            + self.slot_bias[None, :]
            - EVICTION_SCORE_WEIGHT * keep_score
        )
        allocation_distribution = torch.softmax(allocation_score / 0.10, dim=-1)
        allocation = (
            merge_strength[:, None] * merge_distribution
            + (1.0 - merge_strength[:, None]) * allocation_distribution
        )
        victim_slot = allocation_distribution.argmax(dim=-1)
        write_amount = write_strength[:, None] * allocation

        mixed_keys = (
            (1.0 - write_amount[:, :, None]) * dam.keys
            + write_amount[:, :, None] * candidate_key[:, None, :]
        )
        keys = _slot_unit(mixed_keys, dam.occupancy)
        values = (
            (1.0 - write_amount[:, :, None]) * dam.values
            + write_amount[:, :, None] * candidate_value[:, None, :]
        )
        occupancy = dam.occupancy + write_amount * (1.0 - dam.occupancy)

        # Read-back consistency: query the freshly updated memory with the key
        # that was just written and see whether the value comes back. This is
        # the only signal the write path gets that does not have to survive
        # eviction and a 32-256 token delay first, which is why the gate fails
        # to become selective under high write load. Fully self-supervised -
        # the target is the model's own just-written content.
        readback_score = torch.einsum("bd,bnd->bn", candidate_key, keys)
        readback_attention = torch.softmax(readback_score / READ_TEMPERATURE, dim=-1)
        readback_value = torch.einsum("bn,bnv->bv", readback_attention, values)
        readback_error = (
            (readback_value - candidate_value).pow(2).mean(dim=-1) * write_strength
        )

        query = (
            precomputed["query"]
            if precomputed is not None
            else F.normalize(self.query_projection(current_token), dim=-1)
        )
        content_score = torch.einsum("bd,bnd->bn", query, keys)
        read_score = content_score / READ_TEMPERATURE + 0.25 * torch.log(
            occupancy + OCCUPANCY_LOG_EPSILON
        )
        attention = torch.softmax(read_score, dim=-1)
        retrieved = torch.einsum("bn,bnv->bv", attention, values)
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(self.num_slots)
        read_strength = torch.sigmoid(
            self.read_gate(torch.cat([context, retrieved, confidence[:, None]], dim=-1))
        ).squeeze(-1)
        if disable_retrieval:
            memory_contribution = torch.zeros_like(context)
        else:
            memory_contribution = self.memory_projection(
                read_strength[:, None] * retrieved
            )

        hidden = self.output_norm(context + phl_contribution + memory_contribution)
        logits = self.output(hidden)

        # Discrete slot ledger. This is bookkeeping only: it drives the
        # non-learned policies and the residency diagnostics, and never feeds
        # back into the content computation above.
        with torch.no_grad():
            now = float(state.step_index)
            committed = (write_strength > RESIDENCY_WRITE_THRESHOLD).float()
            is_merge = (merge_strength > 0.5).float()
            # The ledger records the slot that received the largest share of a
            # committed write. Allocation is a softmax, so thresholding its mass
            # would miss writes that are decisive but spread; the argmax is the
            # slot the write actually went to.
            chosen = F.one_hot(allocation_distribution.argmax(dim=-1), self.num_slots)
            merged = F.one_hot(merge_distribution.argmax(dim=-1), self.num_slots)
            fresh = (committed * (1.0 - is_merge))[:, None] * chosen.float()
            refresh = (committed * is_merge)[:, None] * merged.float()

            inserted_at = torch.where(fresh > 0.5, now, dam.inserted_at)
            last_access = torch.where(
                (fresh + refresh) > 0.5, now, dam.last_access
            )
            access_count = torch.where(
                fresh > 0.5,
                torch.ones_like(dam.access_count),
                dam.access_count + (refresh > 0.5).float(),
            )
            insert_write_strength = torch.where(
                fresh > 0.5,
                write_strength[:, None].expand_as(dam.insert_write_strength),
                dam.insert_write_strength,
            )
            if previous_token_ids is not None and (collect or arm == "oracle"):
                owner_token = torch.where(
                    fresh > 0.5, previous_token_ids[:, None], dam.owner_token
                )
            else:
                owner_token = dam.owner_token

            # Access on a successful read: the argmax of the content address.
            read_slot = F.one_hot(attention.argmax(dim=-1), self.num_slots).float()
            read_hit = (
                (read_strength > ACCESS_READ_THRESHOLD).float()[:, None]
                * read_slot
                * (1.0 - committed)[:, None]
            )
            last_access = torch.where(read_hit > 0.5, now, last_access)
            access_count = access_count + (read_hit > 0.5).float()

            evicted = ((~has_free) & (write_strength > RESIDENCY_WRITE_THRESHOLD)
                       & (merge_strength <= 0.5))
            lease_before = dam.leases.detach().clone()

        if self.uses_lease:
            new_lease = (
                precomputed["lease_logits"]
                if precomputed is not None and precomputed["lease_logits"] is not None
                else torch.softmax(self.lease_head(context), dim=-1)
            )
            gate = write_amount[:, :, None]
            leases_out = (1.0 - gate) * dam.leases + gate * new_lease[:, None, :]
        else:
            leases_out = dam.leases

        new_state = ModelState(
            phl=phl,
            dam=DAMState(
                keys=keys,
                values=values,
                occupancy=occupancy,
                leases=leases_out,
                inserted_at=inserted_at,
                last_access=last_access,
                access_count=access_count,
                owner_token=owner_token,
                insert_write_strength=insert_write_strength,
            ),
            step_index=state.step_index + 1,
        )
        # Allocator telemetry costs six tensor ops per timestep and is only
        # ever read when diagnostics are being collected, so it is skipped
        # entirely during training.
        allocation_entropy = allocation_max = allocation_margin = None
        occupied_count = None
        if collect:
            with torch.no_grad():
                probabilities = allocation.clamp_min(1e-12)
                allocation_entropy = -(probabilities * probabilities.log()).sum(dim=-1)
                top2 = allocation.topk(2, dim=-1).values
                allocation_max = top2[:, 0]
                allocation_margin = top2[:, 0] - top2[:, 1]
                occupied_count = (occupancy > 0.5).sum(dim=-1).to(allocation.dtype)

        trace = StepTrace(
            write_strength=write_strength,
            read_strength=read_strength,
            attention=attention,
            allocation=allocation,
            eviction_score=keep_score,
            evicted=evicted,
            victim_slot=victim_slot,
            lease_before=lease_before,
            allocation_entropy=allocation_entropy,
            allocation_max=allocation_max,
            allocation_margin=allocation_margin,
            occupied_count=occupied_count,
            readback_error=readback_error,
        )
        return logits, new_state, trace

    def forward(
        self,
        tokens: Tensor,
        arm: str | None = None,
        disable_retrieval: bool = False,
        rng: torch.Generator | None = None,
        oracle_future: Tensor | None = None,
        collect: bool = False,
        initial_dam: DAMState | None = None,
    ) -> tuple[Tensor, dict[str, object] | None]:
        arm = arm or self.arm
        if arm == "oracle" and oracle_future is None:
            raise ValueError("the oracle arm requires generator future information")
        if arm != "oracle" and oracle_future is not None:
            raise ValueError("only the oracle arm may receive future information")

        context_sequence, previous_tokens, current_tokens = self.encode_features(tokens)
        cached = self.precompute(context_sequence, previous_tokens, current_tokens)
        state = self.init_state(tokens.shape[0], tokens.device)
        if initial_dam is not None:
            # Continual learning: resume from a memory bank carried across
            # episode boundaries instead of starting empty.
            state = ModelState(phl=state.phl, dam=initial_dam, step_index=0)
        logits: list[Tensor] = []
        readback: list[Tensor] = []
        ledger: list[Tensor] = []
        occupancy_log: list[Tensor] = []
        lease_log: list[Tensor] = []
        write_log: list[Tensor] = []
        read_log: list[Tensor] = []
        entropy_log: list[Tensor] = []
        margin_log: list[Tensor] = []
        allocmax_log: list[Tensor] = []
        occupied_log: list[Tensor] = []
        eviction_events: list[dict[str, Tensor]] = []

        for timestep in range(tokens.shape[1]):
            slot_next_use = None
            if arm == "oracle":
                slot_next_use = _slot_next_use(
                    state.dam.owner_token, oracle_future, timestep
                )
            previous_ids = (
                tokens[:, timestep - 1] if timestep > 0 else torch.full_like(
                    tokens[:, 0], -1
                )
            )
            step_logits, state, trace = self.step(
                context_sequence[:, timestep],
                previous_tokens[:, timestep],
                current_tokens[:, timestep],
                state,
                arm=arm,
                disable_retrieval=disable_retrieval,
                rng=rng,
                slot_next_use=slot_next_use,
                current_token_ids=tokens[:, timestep],
                previous_token_ids=previous_ids,
                collect=collect,
                precomputed={
                    key: (None if value is None else value[:, timestep])
                    for key, value in cached.items()
                },
            )
            logits.append(step_logits)
            readback.append(trace.readback_error)
            if collect:
                ledger.append(state.dam.owner_token.clone())
                occupancy_log.append(state.dam.occupancy.detach().clone())
                write_log.append(trace.write_strength.detach())
                read_log.append(trace.read_strength.detach())
                entropy_log.append(trace.allocation_entropy)
                margin_log.append(trace.allocation_margin)
                allocmax_log.append(trace.allocation_max)
                occupied_log.append(trace.occupied_count)
                if self.uses_lease:
                    lease_log.append(state.dam.leases.detach().clone())
                if trace.evicted.any():
                    eviction_events.append(
                        {
                            "timestep": timestep,
                            "evicted": trace.evicted.clone(),
                            "victim_slot": trace.victim_slot.clone(),
                            "owner_before": ledger[-2].clone()
                            if len(ledger) > 1
                            else state.dam.owner_token.clone(),
                            "lease_before": trace.lease_before,
                            "eviction_score": trace.eviction_score.detach().clone(),
                        }
                    )

        stacked = torch.stack(logits, dim=1)
        self._last_readback = torch.stack(readback, dim=1)
        if not collect:
            return stacked, None
        diagnostics = {
            "owner_token": torch.stack(ledger, dim=1),
            "occupancy": torch.stack(occupancy_log, dim=1),
            "write_strength": torch.stack(write_log, dim=1),
            "read_strength": torch.stack(read_log, dim=1),
            "allocation_entropy": torch.stack(entropy_log, dim=1),
            "allocation_margin": torch.stack(margin_log, dim=1),
            "allocation_max": torch.stack(allocmax_log, dim=1),
            "occupied_count": torch.stack(occupied_log, dim=1),
            "leases": torch.stack(lease_log, dim=1) if lease_log else None,
            "eviction_events": eviction_events,
            "readback_error": self._last_readback,
            "final_state": state,
        }
        return stacked, diagnostics


def _slot_next_use(
    owner_token: Tensor, key_use_positions: Tensor, now: int
) -> Tensor:
    """Next query position, strictly after ``now``, of each slot's occupant."""
    batch_size, num_slots = owner_token.shape
    index = (owner_token - task.KEY_START).clamp(0, task.NUM_KEYS - 1)
    uses = key_use_positions.gather(
        1, index[:, :, None].expand(-1, -1, key_use_positions.shape[-1])
    )
    future = uses.masked_fill(uses <= now, INFINITY)
    next_use = future.min(dim=-1).values.float()
    return next_use.masked_fill(owner_token < 0, float(INFINITY))


def common_objective(logits: Tensor, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
    next_logits = logits[:, :-1].reshape(-1, task.VOCAB_SIZE)
    next_targets = batch.tokens[:, 1:].reshape(-1)
    all_token_ce = F.cross_entropy(next_logits, next_targets)
    aligned = batch.recall_mask[:, 1:].reshape(-1)
    recall_ce = F.cross_entropy(next_logits[aligned], next_targets[aligned])
    return all_token_ce + recall_ce, all_token_ce, recall_ce


def readback_objective(model: "PHLDAMLease") -> Tensor:
    """Mean read-back error from the most recent forward pass.

    Self-supervised: it compares what memory returns for a key against the
    value the model itself just wrote there, so it uses no labels, no future
    information and no oracle. Weighted by the write gate, so it only asks
    for consistency where the model actually chose to write.
    """
    error = getattr(model, "_last_readback", None)
    if error is None:
        raise RuntimeError("call forward() before readback_objective()")
    return error.mean()


def timing_logits(model: "PHLDAMLease", tokens: Tensor) -> Tensor:
    """Per-token timing-horizon logits from the shared context encoder.

    The encoder is re-applied here rather than threaded out of ``forward``.
    The parameters are shared, so the gradient of the auxiliary loss is
    identical to computing it inside the recurrence; only one extra linear map
    over the context window is spent.
    """
    context, _, _ = model.encode_features(tokens)
    return model.lease_head(context)


def timing_objective(logits: Tensor, batch: Batch) -> Tensor:
    """Supervised next-use-horizon loss. Training only; equal for every arm."""
    return F.cross_entropy(
        logits.reshape(-1, NUM_LEASE_BINS),
        batch.timing_label.reshape(-1),
        ignore_index=TIMING_IGNORE_INDEX,
    )


TRAIN_SEED_OFFSET = 500_000
EVAL_SEED_OFFSET = 20_000
EVAL_SETTINGS = (("canonical", 16), ("canonical", 24), ("canonical", 32), ("spec", 24))


def make_training_batch(
    seed: int, step: int, batch_size: int, device: torch.device
) -> Batch:
    """Mixed-pressure training batch; writes are homogeneous within a batch."""
    chooser = random.Random(seed * 7_919 + step)
    writes = chooser.choice(task.PRESSURE_LEVELS)
    episodes = [
        task.generate_episode(
            seed + TRAIN_SEED_OFFSET, step * batch_size + index, writes, "canonical"
        )
        for index in range(batch_size)
    ]
    return pack_batch(episodes, device)


def train(
    arm: str,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    timing_weight: float = 0.0,
) -> tuple[PHLDAMLease, list[dict[str, float]]]:
    seed_everything(seed)
    model = PHLDAMLease(arm=arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        batch = make_training_batch(seed, step, batch_size, device)
        logits, _ = model(batch.tokens)
        loss, all_ce, recall_ce = common_objective(logits, batch)
        timing_ce = torch.zeros((), device=device)
        if timing_weight > 0.0:
            timing_ce = timing_objective(timing_logits(model, batch.tokens), batch)
            loss = loss + timing_weight * timing_ce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            item = {
                "step": step,
                "writes": batch.writes,
                "loss": loss.item(),
                "all_token_ce": all_ce.item(),
                "recall_ce": recall_ce.item(),
                "timing_ce": timing_ce.item(),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(item)
            print(
                f"arm={arm} seed={seed} step={step:4d} loss={item['loss']:.4f} "
                f"all_ce={item['all_token_ce']:.4f} recall_ce={item['recall_ce']:.4f} "
                f"timing_ce={item['timing_ce']:.4f} "
                f"elapsed={item['elapsed_seconds']:.0f}s",
                flush=True,
            )
    return model, history


def _gather(values: Tensor, positions: Tensor) -> Tensor:
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[rows, positions]


class Tally:
    """Hit/total counters keyed by a string bucket."""

    def __init__(self) -> None:
        self.table: dict[str, list[int]] = {}

    def add(self, key: str, hit: bool) -> None:
        entry = self.table.setdefault(key, [0, 0])
        entry[0] += int(hit)
        entry[1] += 1

    def rates(self) -> dict[str, object]:
        return {
            name: {"recall": correct / total if total else None, "queries": total}
            for name, (correct, total) in sorted(self.table.items())
        }


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and scores[ordered[stop + 1]] == scores[ordered[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[ordered[position]] = average
        index = stop + 1
    positive_rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    count_positive = len(positives)
    count_negative = len(negatives)
    return (
        positive_rank_sum - count_positive * (count_positive + 1) / 2.0
    ) / (count_positive * count_negative)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > 0 else None


@torch.no_grad()
def evaluate_arm(
    model: PHLDAMLease,
    arm: str,
    seed: int,
    writes: int,
    condition: str,
    episodes: int,
    batch_size: int,
    device: torch.device,
    with_disabled_retrieval: bool = False,
) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed * 977 + 13)

    by_delay = Tally()
    by_intervening = Tally()
    by_occurrence = Tally()
    by_pressure = Tally()
    by_residency = Tally()
    anchor = Tally()
    residency = Tally()

    total_queries = 0
    total_correct = 0
    total_resident = 0
    total_correct_given_resident = 0
    disabled_correct = 0
    all_ce_sum = 0.0
    recall_ce_sum = 0.0
    batches = 0

    evictions = 0
    future_needed_evicted = 0
    dead_evicted = 0
    wrong_protection = 0
    correct_protection = 0
    contrast_decisions = 0
    contrast_correct = 0
    soonest_rank_sum = 0.0
    soonest_rank_count = 0
    occupancy_sum = 0.0
    occupancy_count = 0
    write_at_binding = 0.0
    write_at_binding_count = 0
    write_elsewhere = 0.0
    write_elsewhere_count = 0

    timing_correct = 0
    timing_total = 0
    timing_live_scores: list[float] = []
    timing_live_labels: list[int] = []
    score_values: list[float] = []
    score_labels: list[int] = []
    lease_scores: list[float] = []
    next_use_distances: list[float] = []
    lease_live: list[list[float]] = []
    lease_dead: list[list[float]] = []
    lease_entropy_sum = 0.0
    lease_entropy_count = 0

    seen = 0
    while seen < episodes:
        count = min(batch_size, episodes - seen)
        generated = [
            task.generate_episode(seed + EVAL_SEED_OFFSET, seen + index, writes, condition)
            for index in range(count)
        ]
        seen += count
        batch = pack_batch(generated, device)
        oracle_future = batch.key_use_positions if arm == "oracle" else None
        logits, diagnostics = model(
            batch.tokens,
            arm=arm,
            rng=generator,
            oracle_future=oracle_future,
            collect=True,
        )
        assert diagnostics is not None
        _, all_ce, recall_ce = common_objective(logits, batch)
        all_ce_sum += all_ce.item()
        recall_ce_sum += recall_ce.item()
        batches += 1

        predictions = _gather(logits.argmax(dim=-1), batch.query_key_positions)
        correct = predictions.eq(batch.query_target_tokens) & batch.query_valid

        if with_disabled_retrieval:
            disabled_logits, _ = model(
                batch.tokens, arm=arm, disable_retrieval=True,
                rng=generator, oracle_future=oracle_future,
            )
            disabled_predictions = _gather(
                disabled_logits.argmax(dim=-1), batch.query_key_positions
            )
            disabled_correct += (
                disabled_predictions.eq(batch.query_target_tokens) & batch.query_valid
            ).sum().item()

        if arm in LEARNED_ARMS:
            # Timing-head quality at write positions. Labels are generator
            # truth and are used here for measurement only.
            heads = timing_logits(model, batch.tokens)
            mask = batch.timing_label != TIMING_IGNORE_INDEX
            predicted = heads.argmax(dim=-1)[mask]
            truth = batch.timing_label[mask]
            timing_correct += int(predicted.eq(truth).sum())
            timing_total += int(mask.sum())
            live_probability = 1.0 - heads.softmax(dim=-1)[..., -1]
            timing_live_scores.extend(live_probability[mask].tolist())
            timing_live_labels.extend(
                (truth != NUM_LEASE_BINS - 1).long().tolist()
            )

        owner = diagnostics["owner_token"]
        owner_at_query = owner[
            torch.arange(count, device=device)[:, None], batch.query_key_positions
        ]
        resident = (
            owner_at_query.eq(batch.query_key_tokens[:, :, None]).any(dim=-1)
            & batch.query_valid
        )

        occupancy_sum += diagnostics["occupancy"].gt(0.5).float().sum().item()
        occupancy_count += diagnostics["occupancy"].shape[0] * diagnostics["occupancy"].shape[1]

        write_strength = diagnostics["write_strength"]
        binding_mask = torch.zeros_like(write_strength, dtype=torch.bool)
        for row, episode in enumerate(generated):
            for item in episode.items:
                binding_mask[row, item.write_value_position] = True
        write_at_binding += write_strength[binding_mask].sum().item()
        write_at_binding_count += int(binding_mask.sum())
        write_elsewhere += write_strength[~binding_mask].sum().item()
        write_elsewhere_count += int((~binding_mask).sum())

        for row, episode in enumerate(generated):
            bucket = (
                ">8 live"
                if episode.max_concurrent_live > task.NUM_SLOTS
                else "<=8 live"
            )
            for column in range(batch.query_valid.shape[1]):
                if not bool(batch.query_valid[row, column]):
                    continue
                hit = bool(correct[row, column])
                is_resident = bool(resident[row, column])
                total_queries += 1
                total_correct += int(hit)
                total_resident += int(is_resident)
                if is_resident:
                    total_correct_given_resident += int(hit)
                delay = int(batch.delays[row, column])
                by_delay.add(task.delay_bin(delay), hit)
                by_intervening.add(
                    task.intervening_bin(int(batch.intervening[row, column])), hit
                )
                by_occurrence.add(
                    "first-use" if int(batch.occurrence[row, column]) == 0
                    else "repeat-use",
                    hit,
                )
                by_pressure.add(bucket, hit)
                by_residency.add("resident" if is_resident else "evicted", hit)
                residency.add(bucket, is_resident)
                if bool(batch.contrast_anchor[row, column]):
                    anchor.add("contrast-anchor", hit)

        key_uses = batch.key_use_positions
        for event in diagnostics["eviction_events"]:
            timestep = event["timestep"]
            evicted_mask = event["evicted"]
            owner_before = event["owner_before"]
            scores = event["eviction_score"]
            leases_before = event["lease_before"]
            next_use = _slot_next_use(owner_before, key_uses, timestep)
            rows = evicted_mask.nonzero(as_tuple=True)[0]
            for row in rows.tolist():
                victim = int(event["victim_slot"][row])
                slot_next = next_use[row]
                needed = slot_next < INFINITY / 2
                occupied_slots = (owner_before[row] >= 0).nonzero(as_tuple=True)[0]
                if occupied_slots.numel() < model.num_slots:
                    continue
                evictions += 1
                if bool(needed[victim]):
                    future_needed_evicted += 1
                else:
                    dead_evicted += 1
                if bool((~needed).any()):
                    if bool(needed[victim]):
                        wrong_protection += 1
                    else:
                        correct_protection += 1
                if bool(needed.any()) and bool((~needed).any()):
                    contrast_decisions += 1
                    if not bool(needed[victim]):
                        contrast_correct += 1
                if bool(needed.any()):
                    soonest = int(slot_next.argmin())
                    order = scores[row].argsort()
                    rank = int((order == soonest).nonzero(as_tuple=True)[0])
                    soonest_rank_sum += rank / (model.num_slots - 1)
                    soonest_rank_count += 1
                for slot in range(model.num_slots):
                    score_values.append(float(scores[row, slot]))
                    score_labels.append(int(bool(needed[slot])))
                if model.uses_lease:
                    protection = (leases_before[row] * model.lease_readout).sum(dim=-1)
                    for slot in range(model.num_slots):
                        distribution = leases_before[row, slot]
                        if float(distribution.sum()) < 0.5:
                            continue
                        normalised = distribution / distribution.sum()
                        lease_entropy_sum += float(
                            -(normalised * normalised.clamp_min(1e-12).log()).sum()
                        )
                        lease_entropy_count += 1
                        if bool(needed[slot]):
                            lease_live.append(normalised.tolist())
                            lease_scores.append(float(protection[slot]))
                            next_use_distances.append(
                                float(slot_next[slot]) - timestep
                            )
                        else:
                            lease_dead.append(normalised.tolist())

    protections = wrong_protection + correct_protection
    lease_block: dict[str, object] | None = None
    if model.uses_lease:
        lease_block = {
            "mean_lease_of_future_needed": (
                [statistics.fmean(column) for column in zip(*lease_live)]
                if lease_live
                else None
            ),
            "mean_lease_of_never_needed": (
                [statistics.fmean(column) for column in zip(*lease_dead)]
                if lease_dead
                else None
            ),
            "lease_bin_names": list(LEASE_BIN_NAMES),
            "mean_lease_entropy": (
                lease_entropy_sum / lease_entropy_count if lease_entropy_count else None
            ),
            "lease_protection_vs_next_use_pearson": _pearson(
                lease_scores, next_use_distances
            ),
            "lease_readout": model.lease_readout.detach().tolist(),
            "observations_future_needed": len(lease_live),
            "observations_never_needed": len(lease_dead),
        }

    return {
        "arm": arm,
        "writes": writes,
        "condition": condition,
        "episodes": episodes,
        "queries": total_queries,
        "recall": total_correct / total_queries,
        "residency": total_resident / total_queries,
        "recall_given_resident": (
            total_correct_given_resident / total_resident if total_resident else None
        ),
        "retrieval_disabled_recall": (
            disabled_correct / total_queries if with_disabled_retrieval else None
        ),
        "all_token_ce": all_ce_sum / batches,
        "recall_token_ce": recall_ce_sum / batches,
        "recall_by_delay": by_delay.rates(),
        "recall_by_intervening_writes": by_intervening.rates(),
        "recall_by_query_occurrence": by_occurrence.rates(),
        "recall_by_live_pressure_bucket": by_pressure.rates(),
        "recall_by_residency": by_residency.rates(),
        "residency_by_live_pressure_bucket": residency.rates(),
        "recall_contrast_anchor": anchor.rates(),
        "eviction": {
            "decisions": evictions,
            "fraction_future_needed_evicted": (
                future_needed_evicted / evictions if evictions else None
            ),
            "fraction_dead_evicted": dead_evicted / evictions if evictions else None,
            "wrong_protection_rate": (
                wrong_protection / protections if protections else None
            ),
            "protection_decisions": protections,
            "contrast_decisions": contrast_decisions,
            "contrast_correct_rate": (
                contrast_correct / contrast_decisions if contrast_decisions else None
            ),
            "mean_normalised_rank_of_soonest_needed": (
                soonest_rank_sum / soonest_rank_count if soonest_rank_count else None
            ),
            "score_auroc_future_needed_vs_never": _auroc(score_values, score_labels),
        },
        "controller": {
            "mean_occupied_slots": (
                occupancy_sum / occupancy_count if occupancy_count else None
            ),
            "mean_write_gate_at_binding": (
                write_at_binding / write_at_binding_count if write_at_binding_count else None
            ),
            "mean_write_gate_elsewhere": (
                write_elsewhere / write_elsewhere_count if write_elsewhere_count else None
            ),
        },
        "lease": lease_block,
        "timing_head": {
            "class_accuracy": timing_correct / timing_total if timing_total else None,
            "live_vs_never_auroc": _auroc(timing_live_scores, timing_live_labels),
            "labelled_writes": timing_total,
            "note": "measured with generator truth; never supplied to the model at eval",
        },
    }


def parameter_report(model: PHLDAMLease) -> dict[str, object]:
    total = sum(parameter.numel() for parameter in model.parameters())
    policy_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(("lease_", "eviction_", "content_scorer", "utility_scorer"))
    )
    recurrent = (
        model.horizons * model.horizon_width
        + model.num_slots * (model.d_key + model.d_value + 1)
        + model.num_slots * NUM_LEASE_BINS * int(model.uses_lease)
        + model.num_slots * 5
    )
    return {
        "total_parameters": total,
        "eviction_policy_parameters": policy_parameters,
        "content_parameters": total - policy_parameters,
        "recurrent_state_floats": recurrent,
    }


def run(
    arm: str,
    seed: int,
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
    device: torch.device,
    timing_weight: float = 0.0,
) -> dict[str, object]:
    model, history = train(
        arm, seed, steps, batch_size, learning_rate, device, timing_weight
    )
    evaluations: dict[str, dict[str, object]] = {}
    for condition, writes in EVAL_SETTINGS:
        key = f"{condition}_w{writes}"
        evaluations[key] = {}
        # Only this run's own learned arm is evaluable here: a model trained
        # under one learned policy has no scorer for the others.
        for evaluated_arm in POLICY_ARMS + (arm,):
            started = time.perf_counter()
            evaluations[key][evaluated_arm] = evaluate_arm(
                model,
                evaluated_arm,
                seed=seed,
                writes=writes,
                condition=condition,
                episodes=eval_episodes,
                batch_size=batch_size,
                device=device,
                with_disabled_retrieval=(
                    evaluated_arm == arm and condition == "canonical" and writes == 24
                ),
            )
            print(
                f"arm={arm} seed={seed} eval={key} policy={evaluated_arm} "
                f"recall={evaluations[key][evaluated_arm]['recall']:.4f} "
                f"residency={evaluations[key][evaluated_arm]['residency']:.4f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    finite_parameters = all(
        parameter.isfinite().all().item() for parameter in model.parameters()
    )
    return {
        "experiment": (
            "PHL-DAM-004B-S - Supervised-timing lease"
            if timing_weight > 0.0
            else "PHL-DAM-004B - Learned temporal lease under memory pressure"
        ),
        "model": arm,
        "configuration": {
            "scale": task.SCALE,
            "seed": seed,
            "trained_arm": arm,
            "slots": model.num_slots,
            "d_model": model.d_model,
            "horizons": model.horizons,
            "horizon_width": model.horizon_width,
            "d_key": model.d_key,
            "d_value": model.d_value,
            "lease_bins": list(LEASE_BIN_NAMES),
            "lease_state_present": model.uses_lease,
            "lease_transported_through_phl_horizons": model.transports_lease,
            "backbone": model.backbone,
            "phl_enabled": model.backbone == "phl",
            "promotion": False,
            "read_to_write_feedback": False,
            "hard_allocation": True,
            "straight_through_eviction": True,
            "eviction_temperature": model.eviction_temperature,
            "write_budget_weight": 0.0,
            "timing_supervision_weight": timing_weight,
            "timing_supervision_equal_across_arms": True,
            "lease_bin_edges": [list(edge) for edge in lease_bin_edges()],
            "objective": "all-token next-token CE + marked recall-token CE",
            "future_use_labels_to_learned_arms": timing_weight > 0.0,
            "future_use_labels_at_evaluation": False,
            "oracle_uses_generator_truth": True,
            "sequence_length": task.SEQUENCE_LENGTH,
            "delay_range": [task.MIN_DELAY, task.MAX_DELAY],
            "training_pressure_levels": list(task.PRESSURE_LEVELS),
            "training_query_condition": "canonical",
            "training_steps": steps,
            "batch_size": batch_size,
            "eval_episodes_per_setting": eval_episodes,
            "eval_settings": [list(setting) for setting in EVAL_SETTINGS],
            "learning_rate": learning_rate,
            "optimizer": "AdamW(weight_decay=1e-4)",
            "tag_primary_probability": task.TAG_PRIMARY_PROBABILITY,
            "residency_write_threshold": RESIDENCY_WRITE_THRESHOLD,
            "access_read_threshold": ACCESS_READ_THRESHOLD,
        },
        "accounting": parameter_report(model),
        "metrics": evaluations,
        "training_history": history,
        "finite": finite_parameters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=list(LEARNED_ARMS), required=True)
    parser.add_argument("--scale", default="full", choices=list(task.SCALE_PROFILES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--timing-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task.set_scale(args.scale)
    global EVAL_SETTINGS
    EVAL_SETTINGS = tuple(
        [("canonical", writes) for writes in task.PRESSURE_LEVELS]
        + [("spec", task.PRESSURE_LEVELS[-2])]
    )
    summary = run(
        arm=args.arm,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        device=torch.device("cpu"),
        timing_weight=args.timing_weight,
    )
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered[:2000])


if __name__ == "__main__":
    main()
