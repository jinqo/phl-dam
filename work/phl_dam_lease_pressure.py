"""PHL-DAM Lease-001: learned temporal retention under memory pressure.

This retention-only pilot keeps key/value storage exact so the measured variable is
eviction quality. A learned predictor receives a noisy causal cue at write time,
predicts a future-use horizon, and transports that relevance distribution forward.
No future query metadata is available to learned policies during evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NUM_SLOTS = 8
NUM_WRITES = 32
CUE_DIM = 12
NUM_CLASSES = 4
CLASS_NAMES = ("near", "medium", "far", "never")
DELAY_RANGES = ((5, 9), (14, 22), (32, 48))
CLASS_PROBABILITIES = (0.28, 0.28, 0.28, 0.16)
POLICIES = (
    "random",
    "fifo",
    "static_learned",
    "randomized_lease",
    "phl_transported_lease",
    "oracle_next_use",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def make_codebook() -> Tensor:
    generator = torch.Generator().manual_seed(71_921)
    codebook = torch.randn(NUM_CLASSES, CUE_DIM, generator=generator)
    return F.normalize(codebook, dim=-1) * 2.5


CODEBOOK = make_codebook()


def sample_cues(
    generator: torch.Generator,
    count: int,
    noise_std: float = 0.45,
) -> tuple[Tensor, Tensor]:
    probabilities = torch.tensor(CLASS_PROBABILITIES)
    labels = torch.multinomial(probabilities, count, replacement=True, generator=generator)
    noise = torch.randn(count, CUE_DIM, generator=generator) * noise_std
    cues = CODEBOOK[labels] + noise
    return cues, labels


class LeasePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CUE_DIM, 32),
            nn.GELU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, cue: Tensor) -> Tensor:
        return self.network(cue)


@dataclass(frozen=True)
class Binding:
    item_id: int
    write_time: int
    query_time: int | None
    cue: Tensor
    label: int


@dataclass
class CacheEntry:
    binding: Binding
    probabilities: Tensor
    randomized_probabilities: Tensor
    inserted_at: int


def generate_episode(generator: torch.Generator) -> list[Binding]:
    cues, labels = sample_cues(generator, NUM_WRITES)
    bindings = []
    for item_id in range(NUM_WRITES):
        write_time = item_id * 2
        label = int(labels[item_id])
        if label == NUM_CLASSES - 1:
            query_time = None
        else:
            low, high = DELAY_RANGES[label]
            delay = int(torch.randint(low, high + 1, (1,), generator=generator))
            query_time = write_time + delay
        bindings.append(
            Binding(
                item_id=item_id,
                write_time=write_time,
                query_time=query_time,
                cue=cues[item_id],
                label=label,
            )
        )
    return bindings


def transported_lease_priority(probabilities: Tensor, elapsed: int) -> float:
    """Apply fixed countdown transport, then score remaining near-term mass."""
    score = 0.0
    for class_index, (low, high) in enumerate(DELAY_RANGES):
        mass_per_delay = float(probabilities[class_index]) / (high - low + 1)
        for delay in range(low, high + 1):
            remaining = delay - elapsed
            if remaining >= 0:
                score += mass_per_delay / (remaining + 1.0)
    return score


def _entry_priority(
    entry: CacheEntry,
    policy: str,
    now: int,
    random_generator: random.Random,
) -> float:
    if policy == "random":
        return random_generator.random()
    if policy == "fifo":
        return float(entry.inserted_at)
    if policy == "static_learned":
        return 1.0 - float(entry.probabilities[-1])
    if policy == "randomized_lease":
        return transported_lease_priority(
            entry.randomized_probabilities, now - entry.binding.write_time
        )
    if policy == "phl_transported_lease":
        return transported_lease_priority(
            entry.probabilities, now - entry.binding.write_time
        )
    if policy == "oracle_next_use":
        query_time = entry.binding.query_time
        if query_time is None or query_time < now:
            return 0.0
        return 1.0 / (query_time - now + 1.0)
    raise ValueError(f"unknown policy: {policy}")


def simulate_policy(
    bindings: list[Binding],
    probabilities: Tensor,
    policy: str,
    seed: int,
) -> tuple[int, int]:
    random_generator = random.Random(seed)
    cache: dict[int, CacheEntry] = {}
    writes_by_time: dict[int, list[Binding]] = {}
    queries_by_time: dict[int, list[Binding]] = {}
    for binding in bindings:
        writes_by_time.setdefault(binding.write_time, []).append(binding)
        if binding.query_time is not None:
            queries_by_time.setdefault(binding.query_time, []).append(binding)

    max_time = max(
        max(writes_by_time),
        max(queries_by_time) if queries_by_time else 0,
    )
    hits = 0
    queries = 0
    for now in range(max_time + 1):
        for binding in queries_by_time.get(now, ()):
            queries += 1
            if binding.item_id in cache:
                hits += 1
                del cache[binding.item_id]

        for binding in writes_by_time.get(now, ()):
            item_probabilities = probabilities[binding.item_id].detach().cpu()
            temporal = item_probabilities[:3]
            permutation = torch.tensor(
                random_generator.sample(range(3), 3), dtype=torch.long
            )
            randomized = item_probabilities.clone()
            randomized[:3] = temporal[permutation]
            candidate = CacheEntry(
                binding=binding,
                probabilities=item_probabilities,
                randomized_probabilities=randomized,
                inserted_at=now,
            )
            if len(cache) < NUM_SLOTS:
                cache[binding.item_id] = candidate
                continue

            candidates = list(cache.values()) + [candidate]
            priorities = [
                _entry_priority(entry, policy, now, random_generator)
                for entry in candidates
            ]
            evicted = candidates[min(range(len(candidates)), key=priorities.__getitem__)]
            if evicted.binding.item_id != binding.item_id:
                del cache[evicted.binding.item_id]
                cache[binding.item_id] = candidate
    return hits, queries


def calibration_error(probabilities: Tensor, labels: Tensor, bins: int = 10) -> float:
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(labels).float()
    error = torch.tensor(0.0)
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            error += mask.float().mean() * (
                confidence[mask].mean() - correct[mask].mean()
            ).abs()
    return float(error)


def train_predictor(
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[LeasePredictor, list[dict[str, float]]]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + 30_000)
    model = LeasePredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    model.train()
    for step in range(1, steps + 1):
        cues, labels = sample_cues(generator, batch_size)
        logits = model(cues)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            accuracy = logits.argmax(dim=-1).eq(labels).float().mean()
            item = {
                "step": step,
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "gradient_norm": float(gradient_norm),
            }
            history.append(item)
            print(
                f"seed={seed} step={step:4d} loss={item['loss']:.4f} "
                f"accuracy={item['accuracy']:.3f}",
                flush=True,
            )
    return model, history


@torch.no_grad()
def evaluate(
    model: LeasePredictor,
    seed: int,
    episodes: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    totals = {policy: {"hits": 0, "queries": 0} for policy in POLICIES}
    prediction_probabilities = []
    prediction_labels = []
    for episode_index in range(episodes):
        bindings = generate_episode(generator)
        cues = torch.stack([binding.cue for binding in bindings])
        labels = torch.tensor([binding.label for binding in bindings])
        probabilities = model(cues).softmax(dim=-1)
        prediction_probabilities.append(probabilities)
        prediction_labels.append(labels)
        for policy_index, policy in enumerate(POLICIES):
            hits, queries = simulate_policy(
                bindings,
                probabilities,
                policy,
                seed=seed * 1_000_003 + episode_index * 101 + policy_index,
            )
            totals[policy]["hits"] += hits
            totals[policy]["queries"] += queries

    probabilities = torch.cat(prediction_probabilities)
    labels = torch.cat(prediction_labels)
    prediction = {
        "top1_accuracy": float(probabilities.argmax(dim=-1).eq(labels).float().mean()),
        "cross_entropy": float(F.nll_loss(probabilities.log(), labels)),
        "expected_calibration_error": calibration_error(probabilities, labels),
        "examples": int(labels.numel()),
    }
    retention = {
        policy: {
            "hits": values["hits"],
            "queries": values["queries"],
            "hit_rate": values["hits"] / values["queries"],
        }
        for policy, values in totals.items()
    }
    return {"prediction": prediction, "retention": retention}


def run(
    seed: int,
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
) -> dict[str, object]:
    model, history = train_predictor(seed, steps, batch_size, learning_rate)
    metrics = evaluate(model, seed + 40_000, eval_episodes)
    learned = metrics["retention"]["phl_transported_lease"]["hit_rate"]
    static = metrics["retention"]["static_learned"]["hit_rate"]
    heuristics = max(
        metrics["retention"][name]["hit_rate"]
        for name in ("random", "fifo", "randomized_lease")
    )
    oracle = metrics["retention"]["oracle_next_use"]["hit_rate"]
    return {
        "experiment": "PHL-DAM Lease-001 — Retention-only memory-pressure pilot",
        "configuration": {
            "seed": seed,
            "slots": NUM_SLOTS,
            "writes_per_episode": NUM_WRITES,
            "delay_classes": {
                name: list(delay_range)
                for name, delay_range in zip(CLASS_NAMES[:3], DELAY_RANGES)
            },
            "never_queried_class": True,
            "causal_cue_only_at_inference": True,
            "future_query_metadata_visible_to_learned_policy": False,
            "exact_content_store": True,
            "query_consumes_binding": True,
            "training_steps": steps,
            "training_batch_size": batch_size,
            "eval_episodes": eval_episodes,
            "lease_predictor_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "lease_influence": True,
            "promotion": False,
        },
        "gates": {
            "prediction_accuracy_at_least_0_85": (
                metrics["prediction"]["top1_accuracy"] >= 0.85
            ),
            "oracle_over_best_heuristic_at_least_0_10": oracle - heuristics >= 0.10,
            "phl_over_static_at_least_0_05": learned - static >= 0.05,
            "phl_over_best_heuristic_at_least_0_05": learned - heuristics >= 0.05,
        },
        "metrics": metrics,
        "training_history": history,
        "finite": all(parameter.isfinite().all().item() for parameter in model.parameters()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
