"""PHL-DAM Stage B: learned-controller, one-seed content-only pilot.

The model receives only an autoregressive token sequence. WRITE/QUERY markers
are ordinary vocabulary items: their locations, destination slots, and recall
labels are never passed to ``forward``. Training uses all-token next-token CE
plus marked recall-token CE. Temporal leases and promotion are absent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# These tiny recurrent matrix operations are faster and more reproducible on
# the pilot CPU with one intra-op worker than with thread-pool fanout.
torch.set_num_threads(1)


PAD = 0
BOS = 1
WRITE = 2
QUERY = 3
FILL = 4
KEY_START = 5
NUM_KEYS = 32
VALUE_START = KEY_START + NUM_KEYS
NUM_VALUES = 10
VOCAB_SIZE = VALUE_START + NUM_VALUES
SEQUENCE_LENGTH = 176
WRITE_VALUE_POSITIONS = (3, 7, 11)


@dataclass
class Batch:
    tokens: Tensor
    recall_mask: Tensor
    delays: Tensor
    write_value_positions: Tensor
    query_key_positions: Tensor
    query_binding_indices: Tensor


@dataclass
class DAMState:
    keys: Tensor
    values: Tensor
    occupancy: Tensor


@dataclass
class ModelState:
    phl: Tensor
    dam: DAMState


@dataclass
class Diagnostics:
    write_gates: Tensor
    read_gates: Tensor
    attention: Tensor
    allocation: Tensor
    final_occupancy: Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_batch(
    generator: torch.Generator,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
) -> Batch:
    """Build fixed-length randomized WRITE/delay/QUERY episodes.

    Query marker positions occupy three non-overlapping time bands, ensuring
    that every episode exercises short, medium, and long recall distances.
    """
    tokens = torch.full((batch_size, SEQUENCE_LENGTH), FILL, dtype=torch.long)
    tokens[:, 0] = BOS

    key_noise = torch.rand(batch_size, NUM_KEYS, generator=generator)
    key_ids = key_noise.topk(3, dim=-1).indices + KEY_START
    value_ids = torch.randint(
        VALUE_START, VALUE_START + NUM_VALUES, (batch_size, 3), generator=generator
    )

    write_value_positions = torch.tensor(WRITE_VALUE_POSITIONS).repeat(batch_size, 1)
    for binding, value_position in enumerate(WRITE_VALUE_POSITIONS):
        tokens[:, value_position - 2] = WRITE
        tokens[:, value_position - 1] = key_ids[:, binding]
        tokens[:, value_position] = value_ids[:, binding]

    # Marker ranges are chosen so query-key minus write-value distance falls
    # into 29-63, 64-95, and 96-169 respectively for every binding.
    query_marker_positions = torch.stack(
        [
            torch.randint(43, 55, (batch_size,), generator=generator),
            torch.randint(79, 91, (batch_size,), generator=generator),
            torch.randint(135, 166, (batch_size,), generator=generator),
        ],
        dim=1,
    )
    query_order = torch.rand(batch_size, 3, generator=generator).argsort(dim=-1)
    query_keys = key_ids.gather(1, query_order)
    query_values = value_ids.gather(1, query_order)
    query_key_positions = query_marker_positions + 1
    target_positions = query_marker_positions + 2

    rows = torch.arange(batch_size)[:, None]
    tokens[rows, query_marker_positions] = QUERY
    tokens[rows, query_key_positions] = query_keys
    tokens[rows, target_positions] = query_values

    selected_write_positions = write_value_positions.gather(1, query_order)
    delays = query_key_positions - selected_write_positions
    recall_mask = torch.zeros_like(tokens, dtype=torch.bool)
    recall_mask[rows, target_positions] = True

    return Batch(
        tokens=tokens.to(device),
        recall_mask=recall_mask.to(device),
        delays=delays.to(device),
        write_value_positions=write_value_positions.to(device),
        query_key_positions=query_key_positions.to(device),
        query_binding_indices=query_order.to(device),
    )


class PHLDAM(nn.Module):
    """Minimal PHL backbone plus content-only explicit-slot DAM."""

    def __init__(
        self,
        d_model: int = 64,
        horizons: int = 4,
        horizon_width: int = 16,
        num_slots: int = 8,
        d_key: int = 24,
        d_value: int = 24,
        use_phl: bool = True,
    ) -> None:
        super().__init__()
        if horizons * horizon_width != d_model:
            raise ValueError("horizons * horizon_width must equal d_model")
        self.d_model = d_model
        self.horizons = horizons
        self.horizon_width = horizon_width
        self.num_slots = num_slots
        self.d_key = d_key
        self.d_value = d_value
        self.use_phl = use_phl

        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.context_encoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.Tanh(),
        )
        if use_phl:
            self.phl_input = nn.Linear(d_model, d_model, bias=False)
            self.phl_norm = nn.RMSNorm((horizons, horizon_width))
            self.phl_readout = nn.Linear(d_model, d_model, bias=False)

            transport = torch.zeros(horizons, horizons)
            for horizon in range(horizons - 1):
                transport[horizon, horizon] = 0.65
                transport[horizon, horizon + 1] = 0.35
            transport[-1, -1] = 0.98
            self.register_buffer("phl_transport", transport)

        self.key_projection = nn.Linear(d_model, d_key, bias=False)
        self.value_projection = nn.Linear(d_model, d_value, bias=False)
        self.query_projection = nn.Linear(d_model, d_key, bias=False)
        self.write_gate = nn.Linear(d_model, 1)
        self.memory_projection = nn.Linear(d_value, d_model, bias=False)
        self.read_gate = nn.Linear(d_model + d_value + 1, 1)

        self.output_norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)

        # Fixed slot ordering breaks the symmetry between equally empty slots;
        # allocation remains a differentiable softmax (hard allocation is off).
        self.register_buffer("slot_bias", torch.linspace(0.14, 0.0, num_slots))
        nn.init.constant_(self.write_gate.bias, -3.0)
        # Start reads open enough for the recall loss to train the addressing
        # path; the controller remains learned and can close irrelevant reads.
        nn.init.constant_(self.read_gate.bias, 1.0)

    def init_state(self, batch_size: int, device: torch.device) -> ModelState:
        phl = (
            torch.zeros(batch_size, self.horizons, self.horizon_width, device=device)
            if self.use_phl
            else torch.empty(batch_size, 0, device=device)
        )
        return ModelState(
            phl=phl,
            dam=DAMState(
                keys=torch.zeros(batch_size, self.num_slots, self.d_key, device=device),
                values=torch.zeros(
                    batch_size, self.num_slots, self.d_value, device=device
                ),
                occupancy=torch.zeros(batch_size, self.num_slots, device=device),
            ),
        )

    def encode_features(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embedded = self.token_embedding(tokens)
        padding = torch.zeros(
            embedded.shape[0], 2, embedded.shape[2], device=embedded.device
        )
        padded = torch.cat([padding, embedded], dim=1)
        windows = torch.cat(
            [padded[:, 0:-2], padded[:, 1:-1], padded[:, 2:]], dim=-1
        )
        context = self.context_encoder(windows)
        previous_tokens = torch.cat(
            [torch.zeros_like(embedded[:, :1]), embedded[:, :-1]], dim=1
        )
        return context, previous_tokens, embedded

    def step(
        self,
        context: Tensor,
        previous_token: Tensor,
        current_token: Tensor,
        state: ModelState,
        disable_retrieval: bool = False,
    ) -> tuple[Tensor, ModelState, tuple[Tensor, Tensor, Tensor, Tensor]]:
        batch_size = context.shape[0]
        if self.use_phl:
            transported = torch.einsum("ij,bjd->bid", self.phl_transport, state.phl)
            injected = self.phl_input(context).view(
                batch_size, self.horizons, self.horizon_width
            )
            phl = self.phl_norm(transported + injected)
            phl_contribution = self.phl_readout(phl.flatten(1))
        else:
            phl = state.phl
            phl_contribution = torch.zeros_like(context)

        # The relation roles are local and causal: at a binding value token,
        # the preceding token is the candidate key and the current token is the
        # candidate value. The learned write gate must still decide whether the
        # local triple is a binding; no event label reaches this method.
        candidate_key = F.normalize(self.key_projection(previous_token), dim=-1)
        candidate_value = self.value_projection(current_token)
        write_strength = torch.sigmoid(self.write_gate(context)).squeeze(-1)

        occupied = state.dam.occupancy > 0.05
        similarity = torch.einsum("bd,bnd->bn", candidate_key, state.dam.keys)
        masked_similarity = similarity.masked_fill(~occupied, -torch.inf)
        any_occupied = occupied.any(dim=-1, keepdim=True)
        safe_similarity = torch.where(
            any_occupied, masked_similarity, torch.zeros_like(similarity)
        )
        merge_distribution = torch.softmax(safe_similarity / 0.10, dim=-1)
        merge_distribution = torch.where(
            any_occupied, merge_distribution, torch.zeros_like(merge_distribution)
        )
        max_similarity = torch.where(
            any_occupied.squeeze(-1), safe_similarity.max(dim=-1).values, -torch.ones(batch_size, device=context.device)
        )
        merge_strength = torch.sigmoid(12.0 * (max_similarity - 0.72))

        allocation_score = (
            5.0 * (1.0 - state.dam.occupancy) + self.slot_bias[None, :]
        )
        allocation_distribution = torch.softmax(allocation_score / 0.10, dim=-1)
        allocation = (
            merge_strength[:, None] * merge_distribution
            + (1.0 - merge_strength[:, None]) * allocation_distribution
        )
        write_amount = write_strength[:, None] * allocation

        mixed_keys = (
            (1.0 - write_amount[:, :, None]) * state.dam.keys
            + write_amount[:, :, None] * candidate_key[:, None, :]
        )
        keys = F.normalize(mixed_keys, dim=-1)
        values = (
            (1.0 - write_amount[:, :, None]) * state.dam.values
            + write_amount[:, :, None] * candidate_value[:, None, :]
        )
        occupancy = state.dam.occupancy + write_amount * (1.0 - state.dam.occupancy)

        query = F.normalize(self.query_projection(current_token), dim=-1)
        content_score = torch.einsum("bd,bnd->bn", query, keys)
        read_score = content_score / 0.10 + 0.25 * torch.log(occupancy + 1e-6)
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

        hidden = self.output_norm(
            context + phl_contribution + memory_contribution
        )
        logits = self.output(hidden)
        new_state = ModelState(phl=phl, dam=DAMState(keys, values, occupancy))
        return logits, new_state, (
            write_strength,
            read_strength,
            attention,
            allocation,
        )

    def forward(
        self,
        tokens: Tensor,
        disable_retrieval: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, Diagnostics | None]:
        context_sequence, previous_tokens, current_tokens = self.encode_features(tokens)
        state = self.init_state(tokens.shape[0], tokens.device)
        logits = []
        write_gates = []
        read_gates = []
        attention = []
        allocation = []
        for timestep in range(tokens.shape[1]):
            step_logits, state, step_diagnostics = self.step(
                context_sequence[:, timestep],
                previous_tokens[:, timestep],
                current_tokens[:, timestep],
                state,
                disable_retrieval,
            )
            logits.append(step_logits)
            if return_diagnostics:
                write_gates.append(step_diagnostics[0])
                read_gates.append(step_diagnostics[1])
                attention.append(step_diagnostics[2])
                allocation.append(step_diagnostics[3])
        stacked_logits = torch.stack(logits, dim=1)
        if not return_diagnostics:
            return stacked_logits, None
        return stacked_logits, Diagnostics(
            write_gates=torch.stack(write_gates, dim=1),
            read_gates=torch.stack(read_gates, dim=1),
            attention=torch.stack(attention, dim=1),
            allocation=torch.stack(allocation, dim=1),
            final_occupancy=state.dam.occupancy,
        )


def common_objective(logits: Tensor, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
    next_logits = logits[:, :-1].reshape(-1, VOCAB_SIZE)
    next_targets = batch.tokens[:, 1:].reshape(-1)
    all_token_ce = F.cross_entropy(next_logits, next_targets)
    aligned_recall_mask = batch.recall_mask[:, 1:].reshape(-1)
    recall_ce = F.cross_entropy(
        next_logits[aligned_recall_mask], next_targets[aligned_recall_mask]
    )
    return all_token_ce + recall_ce, all_token_ce, recall_ce


def compose_training_loss(
    common_loss: Tensor,
    write_budget: Tensor,
    write_budget_weight: float,
) -> Tensor:
    return common_loss + write_budget_weight * write_budget


def train(
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    write_budget_weight: float = 0.05,
) -> tuple[PHLDAM, list[dict[str, float]]]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model = PHLDAM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        batch = make_batch(generator, batch_size, device)
        logits, diagnostics = model(batch.tokens, return_diagnostics=True)
        assert diagnostics is not None
        common_loss, all_ce, recall_ce = common_objective(logits, batch)
        write_budget = (((diagnostics.write_gates.sum(dim=1) - 3.0) / 3.0) ** 2).mean()
        loss = compose_training_loss(common_loss, write_budget, write_budget_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            item = {
                "step": step,
                "loss": loss.item(),
                "all_token_ce": all_ce.item(),
                "recall_ce": recall_ce.item(),
                "write_budget_penalty": write_budget.item(),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(item)
            print(
                f"step={step:4d} loss={item['loss']:.4f} "
                f"all_ce={item['all_token_ce']:.4f} recall_ce={item['recall_ce']:.4f} "
                f"write_budget={item['write_budget_penalty']:.4f} "
                f"elapsed={item['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return model, history


def _gather_positions(values: Tensor, positions: Tensor) -> Tensor:
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[rows, positions]


@torch.no_grad()
def evaluate(
    model: PHLDAM,
    seed: int,
    episodes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    totals = {
        "queries": 0,
        "correct": 0,
        "disabled_correct": 0,
        "all_ce_sum": 0.0,
        "recall_ce_sum": 0.0,
        "batches": 0,
        "write_true_sum": 0.0,
        "write_true_count": 0,
        "write_other_sum": 0.0,
        "write_other_count": 0,
        "read_query_sum": 0.0,
        "read_query_count": 0,
        "read_other_sum": 0.0,
        "read_other_count": 0,
        "address_correct": 0,
    }
    bin_counts = {"29-63": 0, "64-95": 0, "96-169": 0}
    bin_correct = {name: 0 for name in bin_counts}
    occupancy_sum = torch.zeros(model.num_slots, device=device)
    episodes_seen = 0
    model.eval()

    while episodes_seen < episodes:
        current_batch = min(batch_size, episodes - episodes_seen)
        episodes_seen += current_batch
        batch = make_batch(generator, current_batch, device)
        logits, diagnostics = model(batch.tokens, return_diagnostics=True)
        disabled_logits, _ = model(batch.tokens, disable_retrieval=True)
        assert diagnostics is not None
        _, all_ce, recall_ce = common_objective(logits, batch)
        totals["all_ce_sum"] += all_ce.item()
        totals["recall_ce_sum"] += recall_ce.item()
        totals["batches"] += 1

        target_positions = batch.query_key_positions + 1
        prediction_positions = batch.query_key_positions
        predictions = _gather_positions(logits.argmax(dim=-1), prediction_positions)
        disabled_predictions = _gather_positions(
            disabled_logits.argmax(dim=-1), prediction_positions
        )
        targets = _gather_positions(batch.tokens, target_positions)
        is_correct = predictions.eq(targets)
        totals["queries"] += targets.numel()
        totals["correct"] += is_correct.sum().item()
        totals["disabled_correct"] += disabled_predictions.eq(targets).sum().item()

        for name, low, high in (
            ("29-63", 29, 63),
            ("64-95", 64, 95),
            ("96-169", 96, 169),
        ):
            mask = (batch.delays >= low) & (batch.delays <= high)
            bin_counts[name] += mask.sum().item()
            bin_correct[name] += (is_correct & mask).sum().item()

        write_at_bindings = _gather_positions(
            diagnostics.write_gates, batch.write_value_positions
        )
        write_mask = torch.ones_like(diagnostics.write_gates, dtype=torch.bool)
        rows = torch.arange(current_batch, device=device)[:, None]
        write_mask[rows, batch.write_value_positions] = False
        totals["write_true_sum"] += write_at_bindings.sum().item()
        totals["write_true_count"] += write_at_bindings.numel()
        totals["write_other_sum"] += diagnostics.write_gates[write_mask].sum().item()
        totals["write_other_count"] += write_mask.sum().item()

        read_at_queries = _gather_positions(
            diagnostics.read_gates, batch.query_key_positions
        )
        read_mask = torch.ones_like(diagnostics.read_gates, dtype=torch.bool)
        read_mask[rows, batch.query_key_positions] = False
        totals["read_query_sum"] += read_at_queries.sum().item()
        totals["read_query_count"] += read_at_queries.numel()
        totals["read_other_sum"] += diagnostics.read_gates[read_mask].sum().item()
        totals["read_other_count"] += read_mask.sum().item()

        write_allocations = _gather_positions(
            diagnostics.allocation, batch.write_value_positions
        )
        selected_write_allocations = write_allocations.gather(
            1,
            batch.query_binding_indices[:, :, None].expand(-1, -1, model.num_slots),
        )
        query_attention = _gather_positions(
            diagnostics.attention, batch.query_key_positions
        )
        totals["address_correct"] += (
            query_attention.argmax(dim=-1)
            .eq(selected_write_allocations.argmax(dim=-1))
            .sum()
            .item()
        )
        occupancy_sum += diagnostics.final_occupancy.sum(dim=0)

    query_total = int(totals["queries"])
    batches = int(totals["batches"])
    return {
        "episodes": episodes,
        "queries": query_total,
        "recall_accuracy": totals["correct"] / query_total,
        "retrieval_disabled_accuracy": totals["disabled_correct"] / query_total,
        "address_top1_accuracy": totals["address_correct"] / query_total,
        "all_token_ce": totals["all_ce_sum"] / batches,
        "recall_token_ce": totals["recall_ce_sum"] / batches,
        "distance_accuracy": {
            name: bin_correct[name] / bin_counts[name] for name in bin_counts
        },
        "distance_counts": bin_counts,
        "controller": {
            "mean_write_gate_at_binding": totals["write_true_sum"]
            / totals["write_true_count"],
            "mean_write_gate_elsewhere": totals["write_other_sum"]
            / totals["write_other_count"],
            "mean_read_gate_at_query": totals["read_query_sum"]
            / totals["read_query_count"],
            "mean_read_gate_elsewhere": totals["read_other_sum"]
            / totals["read_other_count"],
            "mean_final_occupancy_by_slot": (occupancy_sum / episodes).tolist(),
        },
        "finite": all(parameter.isfinite().all().item() for parameter in model.parameters()),
    }


def run(
    seed: int,
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
    device: torch.device,
    write_budget_weight: float = 0.05,
) -> dict[str, object]:
    model, history = train(
        seed,
        steps,
        batch_size,
        learning_rate,
        device,
        write_budget_weight=write_budget_weight,
    )
    metrics = evaluate(
        model,
        seed=seed + 20_000,
        episodes=eval_episodes,
        batch_size=batch_size,
        device=device,
    )
    gate_threshold = 0.20
    return {
        "experiment": "PHL-DAM Stage B — Learned One-Seed Pilot",
        "configuration": {
            "seed": seed,
            "bindings": 3,
            "possible_values": NUM_VALUES,
            "possible_keys": NUM_KEYS,
            "slots": model.num_slots,
            "d_model": model.d_model,
            "horizons": model.horizons,
            "horizon_width": model.horizon_width,
            "phl_enabled": model.use_phl,
            "d_key": model.d_key,
            "d_value": model.d_value,
            "sequence_length": SEQUENCE_LENGTH,
            "delay_range": [29, 169],
            "oracle_controller_labels": False,
            "oracle_slot_allocation": False,
            "hard_allocation": False,
            "lease_state_present": False,
            "lease_influence": False,
            "promotion": False,
            "objective": (
                "all-token next-token CE + marked recall-token CE"
                if write_budget_weight == 0.0
                else "all-token next-token CE + marked recall-token CE + "
                f"{write_budget_weight:g} position-agnostic write-budget penalty"
            ),
            "write_budget_weight": write_budget_weight,
            "training_steps": steps,
            "batch_size": batch_size,
            "eval_episodes": eval_episodes,
            "learning_rate": learning_rate,
        },
        "gate": {
            "threshold": gate_threshold,
            "passed": metrics["recall_accuracy"] >= gate_threshold
            and metrics["finite"],
        },
        "metrics": metrics,
        "training_history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--write-budget-weight", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    summary = run(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        device=device,
        write_budget_weight=args.write_budget_weight,
    )
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
