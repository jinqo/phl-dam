"""Content-only PHL-DAM attribution baselines.

Implements the two missing controls for the Stage C protocol:

* DAM-only: the exact learned slot memory/controller with no PHL parameters or
  recurrent PHL state.
* fast-weight: a learned delta-rule associative matrix with the same local
  encoder, write/read controllers, objective, and training/evaluation budget.

Temporal leases and promotion are absent from both models.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from phl_dam_stage_b import (
    Diagnostics,
    PHLDAM,
    VOCAB_SIZE,
    _gather_positions,
    common_objective,
    make_batch,
    seed_everything,
)


@dataclass
class FastWeightState:
    matrix: Tensor


class FastWeightDelta(PHLDAM):
    """Delta-rule online key/value regression without explicit slots."""

    def __init__(self, d_model: int = 64, d_key: int = 24, d_value: int = 24) -> None:
        super().__init__(
            d_model=d_model,
            horizons=4,
            horizon_width=16,
            num_slots=8,
            d_key=d_key,
            d_value=d_value,
            use_phl=False,
        )
        # sigmoid(9) ~= 0.99988: almost-stationary at initialization, while
        # still allowing the task to learn retention if decay is useful.
        self.retention_logit = torch.nn.Parameter(torch.tensor(9.0))

    def init_state(self, batch_size: int, device: torch.device) -> FastWeightState:
        return FastWeightState(
            matrix=torch.zeros(
                batch_size, self.d_value, self.d_key, device=device
            )
        )

    def step(
        self,
        context: Tensor,
        previous_token: Tensor,
        current_token: Tensor,
        state: FastWeightState,
        disable_retrieval: bool = False,
    ) -> tuple[Tensor, FastWeightState, tuple[Tensor, Tensor, Tensor, Tensor]]:
        candidate_key = F.normalize(self.key_projection(previous_token), dim=-1)
        candidate_value = self.value_projection(current_token)
        write_strength = torch.sigmoid(self.write_gate(context)).squeeze(-1)

        predicted_value = torch.einsum("bvk,bk->bv", state.matrix, candidate_key)
        value_error = candidate_value - predicted_value
        delta = torch.einsum("bv,bk->bvk", value_error, candidate_key)
        retention = torch.sigmoid(self.retention_logit)
        matrix = retention * state.matrix + write_strength[:, None, None] * delta

        query = F.normalize(self.query_projection(current_token), dim=-1)
        retrieved = torch.einsum("bvk,bk->bv", matrix, query)
        confidence = torch.ones(context.shape[0], 1, device=context.device)
        read_strength = torch.sigmoid(
            self.read_gate(torch.cat([context, retrieved, confidence], dim=-1))
        ).squeeze(-1)
        memory_contribution = (
            torch.zeros_like(context)
            if disable_retrieval
            else self.memory_projection(read_strength[:, None] * retrieved)
        )
        hidden = self.output_norm(context + memory_contribution)
        logits = self.output(hidden)

        # Single-column placeholders keep the shared diagnostics interface;
        # explicit-slot address rank is reported as unavailable for this model.
        singleton = torch.ones(context.shape[0], 1, device=context.device)
        return logits, FastWeightState(matrix), (
            write_strength,
            read_strength,
            singleton,
            singleton,
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
        writes = []
        reads = []
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
                writes.append(step_diagnostics[0])
                reads.append(step_diagnostics[1])
                attention.append(step_diagnostics[2])
                allocation.append(step_diagnostics[3])
        stacked_logits = torch.stack(logits, dim=1)
        if not return_diagnostics:
            return stacked_logits, None
        return stacked_logits, Diagnostics(
            write_gates=torch.stack(writes, dim=1),
            read_gates=torch.stack(reads, dim=1),
            attention=torch.stack(attention, dim=1),
            allocation=torch.stack(allocation, dim=1),
            final_occupancy=torch.empty(tokens.shape[0], 0, device=tokens.device),
        )


class ParameterMatchedDAM(PHLDAM):
    """PHL-off slot DAM with generic residual capacity matched to PHL-DAM."""

    def __init__(self) -> None:
        super().__init__(use_phl=False)
        self.generic_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_model, self.d_model),
        )

    def encode_features(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        context, previous_tokens, current_tokens = super().encode_features(tokens)
        return context + self.generic_ffn(context), previous_tokens, current_tokens


def build_model(model_kind: str) -> PHLDAM:
    if model_kind == "dam_only":
        return PHLDAM(use_phl=False)
    if model_kind == "dam_only_matched":
        return ParameterMatchedDAM()
    if model_kind == "fast_weight":
        return FastWeightDelta()
    raise ValueError(f"unknown model kind: {model_kind}")


def active_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def recurrent_state_floats(model_kind: str, model: PHLDAM) -> int:
    if model_kind in ("dam_only", "dam_only_matched"):
        return model.num_slots * (model.d_key + model.d_value) + model.num_slots
    if model_kind == "fast_weight":
        return model.d_key * model.d_value
    raise ValueError(model_kind)


def train_variant(
    model_kind: str,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[PHLDAM, list[dict[str, float]]]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model = build_model(model_kind).to(device)
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
        loss = common_loss + 0.05 * write_budget
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
                f"model={model_kind} seed={seed} step={step:4d} "
                f"loss={item['loss']:.4f} all_ce={item['all_token_ce']:.4f} "
                f"recall_ce={item['recall_ce']:.4f} "
                f"write_budget={item['write_budget_penalty']:.4f} "
                f"elapsed={item['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return model, history


@torch.no_grad()
def evaluate_variant(
    model_kind: str,
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
        "all_ce": 0.0,
        "recall_ce": 0.0,
        "batches": 0,
        "write_true": 0.0,
        "write_other": 0.0,
        "read_query": 0.0,
        "read_other": 0.0,
        "write_true_count": 0,
        "write_other_count": 0,
        "read_query_count": 0,
        "read_other_count": 0,
        "address_correct": 0,
    }
    bin_counts = {"29-63": 0, "64-95": 0, "96-169": 0}
    bin_correct = {name: 0 for name in bin_counts}
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
        totals["all_ce"] += all_ce.item()
        totals["recall_ce"] += recall_ce.item()
        totals["batches"] += 1

        prediction_positions = batch.query_key_positions
        target_positions = prediction_positions + 1
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

        rows = torch.arange(current_batch, device=device)[:, None]
        write_at_bindings = _gather_positions(
            diagnostics.write_gates, batch.write_value_positions
        )
        write_mask = torch.ones_like(diagnostics.write_gates, dtype=torch.bool)
        write_mask[rows, batch.write_value_positions] = False
        read_at_queries = _gather_positions(
            diagnostics.read_gates, batch.query_key_positions
        )
        read_mask = torch.ones_like(diagnostics.read_gates, dtype=torch.bool)
        read_mask[rows, batch.query_key_positions] = False
        totals["write_true"] += write_at_bindings.sum().item()
        totals["write_other"] += diagnostics.write_gates[write_mask].sum().item()
        totals["read_query"] += read_at_queries.sum().item()
        totals["read_other"] += diagnostics.read_gates[read_mask].sum().item()
        totals["write_true_count"] += write_at_bindings.numel()
        totals["write_other_count"] += write_mask.sum().item()
        totals["read_query_count"] += read_at_queries.numel()
        totals["read_other_count"] += read_mask.sum().item()

        if model_kind in ("dam_only", "dam_only_matched"):
            write_allocations = _gather_positions(
                diagnostics.allocation, batch.write_value_positions
            )
            selected_allocations = write_allocations.gather(
                1,
                batch.query_binding_indices[:, :, None].expand(
                    -1, -1, model.num_slots
                ),
            )
            query_attention = _gather_positions(
                diagnostics.attention, batch.query_key_positions
            )
            totals["address_correct"] += (
                query_attention.argmax(dim=-1)
                .eq(selected_allocations.argmax(dim=-1))
                .sum()
                .item()
            )

    queries = int(totals["queries"])
    batches = int(totals["batches"])
    metrics: dict[str, object] = {
        "episodes": episodes,
        "queries": queries,
        "recall_accuracy": totals["correct"] / queries,
        "retrieval_disabled_accuracy": totals["disabled_correct"] / queries,
        "all_token_ce": totals["all_ce"] / batches,
        "recall_token_ce": totals["recall_ce"] / batches,
        "distance_accuracy": {
            name: bin_correct[name] / bin_counts[name] for name in bin_counts
        },
        "distance_counts": bin_counts,
        "controller": {
            "mean_write_gate_at_binding": totals["write_true"]
            / totals["write_true_count"],
            "mean_write_gate_elsewhere": totals["write_other"]
            / totals["write_other_count"],
            "mean_read_gate_at_query": totals["read_query"]
            / totals["read_query_count"],
            "mean_read_gate_elsewhere": totals["read_other"]
            / totals["read_other_count"],
        },
        "finite": all(parameter.isfinite().all().item() for parameter in model.parameters()),
    }
    metrics["address_top1_accuracy"] = (
        totals["address_correct"] / queries
        if model_kind in ("dam_only", "dam_only_matched")
        else None
    )
    if model_kind == "fast_weight":
        metrics["learned_retention"] = torch.sigmoid(model.retention_logit).item()
    return metrics


def run(
    model_kind: str,
    seed: int,
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    model, history = train_variant(
        model_kind, seed, steps, batch_size, learning_rate, device
    )
    metrics = evaluate_variant(
        model_kind,
        model,
        seed=seed + 20_000,
        episodes=eval_episodes,
        batch_size=batch_size,
        device=device,
    )
    return {
        "experiment": "PHL-DAM content-only backbone attribution",
        "model": model_kind,
        "configuration": {
            "seed": seed,
            "bindings": 3,
            "possible_values": 10,
            "possible_keys": 32,
            "d_model": model.d_model,
            "d_key": model.d_key,
            "d_value": model.d_value,
            "phl_enabled": False,
            "explicit_slots": model_kind in ("dam_only", "dam_only_matched"),
            "delta_rule": model_kind == "fast_weight",
            "generic_ffn_width": model.d_model
            if model_kind == "dam_only_matched"
            else None,
            "lease_state_present": False,
            "lease_influence": False,
            "promotion": False,
            "hard_allocation": False,
            "objective": "all-token next-token CE + marked recall-token CE + 0.05 position-agnostic write-budget penalty",
            "training_steps": steps,
            "batch_size": batch_size,
            "eval_episodes": eval_episodes,
            "learning_rate": learning_rate,
            "active_parameters": active_parameter_count(model),
            "recurrent_state_floats_per_sequence": recurrent_state_floats(
                model_kind, model
            ),
        },
        "metrics": metrics,
        "training_history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("dam_only", "dam_only_matched", "fast_weight"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        model_kind=args.model,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        device=torch.device("cpu"),
    )
    rendered = json.dumps(summary, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
