"""PHL-DAM Stage A: oracle-controlled associative-memory diagnostic.

This intentionally tests only the memory primitive. WRITE locations, QUERY
locations, and destination slots are supplied by the synthetic oracle; key,
query, value, and output projections are learned from randomized episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DAMState:
    keys: Tensor
    values: Tensor
    occupancy: Tensor

    def clone(self) -> "DAMState":
        return DAMState(
            keys=self.keys.clone(),
            values=self.values.clone(),
            occupancy=self.occupancy.clone(),
        )


class OracleDAM(nn.Module):
    """Small content-addressed memory with oracle-controlled writes.

    Allocation and event detection are deliberately out of scope for Stage A.
    Content addressing and the value/readout path remain fully learned.
    """

    def __init__(
        self,
        num_keys: int = 32,
        num_values: int = 10,
        num_slots: int = 8,
        d_model: int = 64,
        d_key: int = 24,
        d_value: int = 24,
        read_temperature: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_slots = num_slots
        self.d_key = d_key
        self.d_value = d_value
        self.read_temperature = read_temperature

        self.key_embedding = nn.Embedding(num_keys, d_model)
        self.value_embedding = nn.Embedding(num_values, d_model)
        self.binding_encoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Tanh(),
        )
        self.key_projection = nn.Linear(d_model, d_key, bias=False)
        self.value_projection = nn.Linear(d_model, d_value, bias=False)
        self.query_projection = nn.Linear(d_model, d_key, bias=False)
        self.output = nn.Linear(d_value, num_values)

    def init_state(self, batch_size: int, device: torch.device) -> DAMState:
        return DAMState(
            keys=torch.zeros(batch_size, self.num_slots, self.d_key, device=device),
            values=torch.zeros(
                batch_size, self.num_slots, self.d_value, device=device
            ),
            occupancy=torch.zeros(
                batch_size, self.num_slots, dtype=torch.bool, device=device
            ),
        )

    def encode_binding(self, key_ids: Tensor, value_ids: Tensor) -> tuple[Tensor, Tensor]:
        binding = self.binding_encoder(
            torch.cat(
                [self.key_embedding(key_ids), self.value_embedding(value_ids)], dim=-1
            )
        )
        keys = F.normalize(self.key_projection(binding), dim=-1)
        values = self.value_projection(binding)
        return keys, values

    def write_slot(
        self, state: DAMState, slot: int, key_ids: Tensor, value_ids: Tensor
    ) -> DAMState:
        """Return a new state with one oracle-selected slot overwritten."""
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} outside [0, {self.num_slots})")
        new_state = state.clone()
        keys, values = self.encode_binding(key_ids, value_ids)
        new_state.keys[:, slot] = keys
        new_state.values[:, slot] = values
        new_state.occupancy[:, slot] = True
        return new_state

    def read(self, state: DAMState, query_key_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        query = F.normalize(
            self.query_projection(self.key_embedding(query_key_ids)), dim=-1
        )
        scores = torch.einsum("bd,bnd->bn", query, state.keys)
        scores = scores / self.read_temperature

        any_occupied = state.occupancy.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~state.occupancy, -torch.inf)
        safe_scores = torch.where(any_occupied, masked_scores, torch.zeros_like(scores))
        attention = torch.softmax(safe_scores, dim=-1)
        attention = torch.where(any_occupied, attention, torch.zeros_like(attention))
        retrieved = torch.einsum("bn,bnv->bv", attention, state.values)
        logits = self.output(retrieved)
        return logits, attention, retrieved

    def write_episode(self, key_ids: Tensor, value_ids: Tensor) -> DAMState:
        if key_ids.shape != value_ids.shape:
            raise ValueError("key_ids and value_ids must have identical shapes")
        if key_ids.ndim != 2:
            raise ValueError("episode tensors must be [batch, bindings]")
        bindings = key_ids.shape[1]
        if bindings > self.num_slots:
            raise ValueError("oracle episode has more bindings than slots")
        state = self.init_state(key_ids.shape[0], key_ids.device)
        for slot in range(bindings):
            state = self.write_slot(state, slot, key_ids[:, slot], value_ids[:, slot])
        return state


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_episodes(
    generator: torch.Generator,
    batch_size: int,
    num_keys: int,
    num_values: int,
    bindings: int = 3,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create randomized bindings, query order, and independent long delays."""
    key_noise = torch.rand(batch_size, num_keys, generator=generator)
    keys = key_noise.topk(bindings, dim=-1).indices.to(device)
    values = torch.randint(
        num_values, (batch_size, bindings), generator=generator, device=device
    )
    query_order = torch.rand(batch_size, bindings, generator=generator).argsort(dim=-1)
    query_order = query_order.to(device)
    delays = torch.randint(
        29, 170, (batch_size, bindings), generator=generator, device=device
    )
    return keys, values, query_order, delays


def episode_loss(
    model: OracleDAM, keys: Tensor, values: Tensor, query_order: Tensor
) -> Tensor:
    state = model.write_episode(keys, values)
    losses = []
    for query_index in range(keys.shape[1]):
        slots = query_order[:, query_index]
        query_keys = keys.gather(1, slots[:, None]).squeeze(1)
        targets = values.gather(1, slots[:, None]).squeeze(1)
        logits, _, _ = model.read(state, query_keys)
        losses.append(F.cross_entropy(logits, targets))
    return torch.stack(losses).mean()


def train_seed(
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> OracleDAM:
    seed_everything(seed)
    model = OracleDAM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model.train()
    for _ in range(steps):
        keys, values, query_order, _ = make_episodes(
            generator, batch_size, model.num_keys, model.num_values, device=device
        )
        loss = episode_loss(model, keys, values, query_order)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    model: OracleDAM,
    seed: int,
    episodes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int | dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    total = correct = address_correct = 0
    shuffled_correct = 0
    entropy_sum = margin_sum = 0.0
    bin_counts = {"29-63": 0, "64-95": 0, "96-169": 0}
    bin_correct = {name: 0 for name in bin_counts}

    batches = math.ceil(episodes / batch_size)
    episodes_seen = 0
    for _ in range(batches):
        current_batch = min(batch_size, episodes - episodes_seen)
        episodes_seen += current_batch
        keys, values, query_order, delays = make_episodes(
            generator,
            current_batch,
            model.num_keys,
            model.num_values,
            device=device,
        )
        state = model.write_episode(keys, values)
        shuffled_state = DAMState(
            keys=state.keys,
            values=torch.roll(state.values, shifts=1, dims=0),
            occupancy=state.occupancy,
        )
        for query_index in range(keys.shape[1]):
            slots = query_order[:, query_index]
            query_keys = keys.gather(1, slots[:, None]).squeeze(1)
            targets = values.gather(1, slots[:, None]).squeeze(1)
            query_delays = delays.gather(1, slots[:, None]).squeeze(1)

            logits, attention, _ = model.read(state, query_keys)
            shuffled_logits, _, _ = model.read(shuffled_state, query_keys)
            predictions = logits.argmax(dim=-1)
            is_correct = predictions.eq(targets)
            total += targets.numel()
            correct += is_correct.sum().item()
            shuffled_correct += shuffled_logits.argmax(dim=-1).eq(targets).sum().item()
            address_correct += attention.argmax(dim=-1).eq(slots).sum().item()

            occupied_attention = attention[:, : keys.shape[1]]
            entropy = -(occupied_attention * occupied_attention.clamp_min(1e-12).log()).sum(-1)
            entropy_sum += entropy.sum().item()
            top2 = attention.topk(2, dim=-1).values
            margin_sum += (top2[:, 0] - top2[:, 1]).sum().item()

            for name, low, high in (
                ("29-63", 29, 63),
                ("64-95", 64, 95),
                ("96-169", 96, 169),
            ):
                mask = (query_delays >= low) & (query_delays <= high)
                bin_counts[name] += mask.sum().item()
                bin_correct[name] += (is_correct & mask).sum().item()

    return {
        "episodes": episodes,
        "queries": total,
        "recall_accuracy": correct / total,
        "address_top1_accuracy": address_correct / total,
        "mean_read_entropy": entropy_sum / total,
        "mean_top1_top2_margin": margin_sum / total,
        "shuffled_memory_accuracy": shuffled_correct / total,
        "distance_accuracy": {
            name: bin_correct[name] / bin_counts[name] for name in bin_counts
        },
        "distance_counts": bin_counts,
    }


def run(
    seeds: Iterable[int],
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    results = []
    for seed in seeds:
        model = train_seed(seed, steps, batch_size, learning_rate, device)
        metrics = evaluate(
            model,
            seed=seed + 20_000,
            episodes=eval_episodes,
            batch_size=batch_size,
            device=device,
        )
        metrics["seed"] = seed
        results.append(metrics)
        print(
            f"seed={seed} recall={metrics['recall_accuracy']:.4f} "
            f"address={metrics['address_top1_accuracy']:.4f} "
            f"shuffled={metrics['shuffled_memory_accuracy']:.4f}"
        )

    recalls = [float(result["recall_accuracy"]) for result in results]
    gate_threshold = 0.90
    summary = {
        "experiment": "PHL-DAM Stage A — Oracle Memory Primitive",
        "configuration": {
            "bindings": 3,
            "possible_values": 10,
            "possible_keys": 32,
            "slots": 8,
            "d_key": 24,
            "d_value": 24,
            "delay_range": [29, 169],
            "oracle_write_locations": True,
            "oracle_query_locations": True,
            "oracle_slot_allocation": True,
            "lease_influence": False,
            "promotion": False,
            "training_steps": steps,
            "batch_size": batch_size,
            "eval_episodes_per_seed": eval_episodes,
            "learning_rate": learning_rate,
        },
        "gate": {
            "threshold": gate_threshold,
            "rule": "every tested seed must achieve recall >= threshold",
            "passed": all(recall >= gate_threshold for recall in recalls),
        },
        "aggregate": {
            "mean_recall_accuracy": sum(recalls) / len(recalls),
            "min_recall_accuracy": min(recalls),
            "max_recall_accuracy": max(recalls),
        },
        "seeds": results,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        device=torch.device("cpu"),
    )
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
