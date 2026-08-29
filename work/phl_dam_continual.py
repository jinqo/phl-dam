"""PHL-DAM continual learning: add knowledge by writing slots, not by moving weights.

The motivating property is structural, not incidental. PHL-DAM keeps its
knowledge in eight explicit key->value slots that are written at inference time
by a content-addressed controller. New facts can therefore be added *without any
gradient step at all*, which is the mechanism catastrophic forgetting attacks.
A purely parametric model has no such option: every new fact must move weights
that older facts also depend on.

This module turns that into a falsifiable experiment rather than a claim.

Protocol
--------
A run is a sequence of TASKS. Each task draws its keys from a disjoint slice of
the key vocabulary, so "remembering task 1" is unambiguous - the keys involved
appear nowhere else. After each task the model is evaluated on *every* task seen
so far, giving the standard continual-learning quantities:

    retention  - accuracy on old tasks after later tasks have been learned
    plasticity - accuracy on the task just learned
    forgetting - drop from a task's peak accuracy to its final accuracy

Three modes are compared, all sharing frozen streams and seeds:

    weights    - gradient updates per task, memory reset per episode.
                 The conventional continual-learning setting.
    memory     - NO gradient updates after task 0. Later tasks are absorbed
                 only by writing slots at inference time.
    both       - gradient updates AND a memory bank carried across episodes.
    persistent - frozen weights after task 0 AND a carried memory bank.

    Note on carrying memory: with only eight slots and roughly eight writes
    per episode, a carried bank is almost entirely overwritten within one
    episode. Whether that helps, hurts or does nothing is an empirical
    question this module measures rather than assumes.

The memory mode is the interesting one: if the architecture's claim is real,
it should retain old tasks far better than `weights` while still absorbing new
ones. If it does not, the claim fails and the honest result is that explicit
slots buy nothing here.

Nothing about the lease is used; that mechanism was rejected in PHL-DAM-004B-S.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import phl_dam_pressure_task as task
from phl_dam_004b_lease import (
    DAMState,
    PHLDAMLease,
    common_objective,
    pack_batch,
    seed_everything,
)

torch.set_num_threads(1)


MODES = ("weights", "memory", "both", "persistent")


@dataclass
class TaskSpec:
    """One task.

    ``disjoint`` regime: each task owns a private slice of the key
    vocabulary, so tasks cannot contradict one another.

    ``conflicting`` regime: every task uses the SAME keys but shifts the
    value each key maps to. Learning task 2 therefore actively invalidates
    what task 1 taught about the same key. This is the regime where
    catastrophic forgetting actually bites, and the disjoint regime turned
    out to be too easy to test it - `weights` mode forgot only 0.029 there,
    because the model learns a general store-and-retrieve skill rather than
    task-specific facts.
    """

    index: int
    key_low: int
    key_high: int
    seed: int
    value_shift: int = 0
    regime: str = "disjoint"


def build_tasks(count: int, seed: int, regime: str = "disjoint") -> list[TaskSpec]:
    if regime not in ("disjoint", "conflicting"):
        raise ValueError(regime)
    if regime == "conflicting":
        if count > task.NUM_VALUES:
            raise ValueError("more tasks than values to rotate through")
        # Same key block for every task; only the key->value mapping moves.
        return [
            TaskSpec(
                index=index,
                key_low=0,
                key_high=task.NUM_KEYS // count,
                seed=seed * 7919 + index * 101,
                value_shift=index + 1,
                regime=regime,
            )
            for index in range(count)
        ]
    per_task = task.NUM_KEYS // count
    if per_task < 4:
        raise ValueError("too many tasks for the key vocabulary")
    return [
        TaskSpec(
            index=index,
            key_low=index * per_task,
            key_high=(index + 1) * per_task,
            seed=seed * 7919 + index * 101,
            regime=regime,
        )
        for index in range(count)
    ]


def task_episode(spec: TaskSpec, episode_index: int, writes: int):
    """An episode whose keys are remapped into this task's disjoint block.

    The generator is untouched; only the key identities are relabelled, so every
    other property of the stream - delays, positions, query structure, the
    live/dead mixture - is identical across tasks. That keeps task difficulty
    constant and makes retention differences attributable to interference alone.
    """
    episode = task.generate_episode(spec.seed, episode_index, writes, "canonical")
    width = spec.key_high - spec.key_low
    for item in episode.items:
        offset = (item.key_token - task.KEY_START) % width
        item.key_token = task.KEY_START + spec.key_low + offset
        if spec.value_shift:
            # Deterministic per-key remap: the same key means a different
            # value in every task, so later tasks contradict earlier ones.
            index = (offset + spec.value_shift) % task.NUM_VALUES
            item.value_token = task.VALUE_START + index
    episode.tokens = task.tokenise(episode.items, episode.queries)
    return episode


def make_batch(spec: TaskSpec, start: int, count: int, writes: int, device):
    return pack_batch(
        [task_episode(spec, start + i, writes) for i in range(count)], device
    )


def slice_memory(carried: DAMState | None, count: int) -> DAMState | None:
    """Adapt a stored bank to a batch of ``count`` episodes."""
    if carried is None:
        return None
    take = lambda t: t[:1].expand(count, *t.shape[1:]).clone()
    return DAMState(
        keys=take(carried.keys), values=take(carried.values),
        occupancy=take(carried.occupancy), leases=take(carried.leases),
        inserted_at=take(carried.inserted_at), last_access=take(carried.last_access),
        access_count=take(carried.access_count), owner_token=take(carried.owner_token),
        insert_write_strength=take(carried.insert_write_strength),
    )


def detach_memory(state) -> DAMState:
    """Snapshot the slot bank so it can be carried across episode boundaries."""
    dam = state.dam
    return DAMState(
        keys=dam.keys.detach().clone(),
        values=dam.values.detach().clone(),
        occupancy=dam.occupancy.detach().clone(),
        leases=dam.leases.detach().clone(),
        inserted_at=dam.inserted_at.detach().clone(),
        last_access=dam.last_access.detach().clone(),
        access_count=dam.access_count.detach().clone(),
        owner_token=dam.owner_token.detach().clone(),
        insert_write_strength=dam.insert_write_strength.detach().clone(),
    )


@torch.no_grad()
def evaluate_task(
    model: PHLDAMLease, spec: TaskSpec, episodes: int, batch_size: int, writes: int,
    device, carried: DAMState | None = None,
) -> dict[str, float]:
    model.eval()
    correct = total = 0
    ce_sum = 0.0
    batches = 0
    seen = 0
    while seen < episodes:
        count = min(batch_size, episodes - seen)
        batch = make_batch(spec, 900_000 + seen, count, writes, device)
        seen += count
        logits, _ = model(
            batch.tokens, arm="content_only", collect=False,
            initial_dam=slice_memory(carried, count),
        )
        _, _, recall_ce = common_objective(logits, batch)
        ce_sum += float(recall_ce)
        batches += 1
        rows = torch.arange(count, device=device)[:, None]
        predictions = logits.argmax(dim=-1)[rows, batch.query_key_positions]
        hit = predictions.eq(batch.query_target_tokens) & batch.query_valid
        correct += int(hit.sum())
        total += int(batch.query_valid.sum())
    model.train()
    return {"recall": correct / total, "recall_ce": ce_sum / batches, "queries": total}


def run(
    mode: str,
    seed: int,
    tasks: int,
    steps_per_task: int,
    batch_size: int,
    eval_episodes: int,
    writes: int,
    learning_rate: float,
    device: torch.device,
    regime: str = "disjoint",
) -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    task.set_scale("compact")
    seed_everything(seed)
    model = PHLDAMLease(arm="content_only").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    specs = build_tasks(tasks, seed, regime)

    # accuracy_after[i][j] = accuracy on task j after finishing task i
    carries_memory = mode in ("both", "persistent")
    carried: DAMState | None = None
    accuracy_after: list[list[float]] = []
    history: list[dict] = []
    started = time.perf_counter()

    for spec in specs:
        # `memory` mode trains only on the first task; every later task must be
        # absorbed by writing slots at inference time, with weights frozen.
        train_this_task = mode in ("weights", "both") or spec.index == 0
        if train_this_task:
            for step in range(1, steps_per_task + 1):
                batch = make_batch(spec, step * batch_size, batch_size, writes, device)
                logits, diagnostics = model(
                    batch.tokens, arm="content_only", collect=carries_memory,
                    initial_dam=slice_memory(carried, batch_size),
                )
                loss, all_ce, recall_ce = common_objective(logits, batch)
                if carries_memory and diagnostics is not None:
                    carried = detach_memory(diagnostics["final_state"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % 50 == 0 or step == steps_per_task:
                    history.append(
                        {
                            "task": spec.index,
                            "step": step,
                            "loss": float(loss.detach()),
                            "recall_ce": float(recall_ce.detach()),
                            "gradient_norm": float(norm),
                        }
                    )
                    print(
                        f"mode={mode} seed={seed} task={spec.index} step={step:4d} "
                        f"recall_ce={float(recall_ce.detach()):.4f} grad={float(norm):.3g}",
                        flush=True,
                    )

        row = [
            evaluate_task(
                model, earlier, eval_episodes, batch_size, writes, device, carried
            )["recall"]
            for earlier in specs[: spec.index + 1]
        ]
        accuracy_after.append(row)
        print(
            f"mode={mode} seed={seed} after task {spec.index}: "
            + " ".join(f"t{j}={v:.3f}" for j, v in enumerate(row)),
            flush=True,
        )

    final = accuracy_after[-1]
    peaks = [max(accuracy_after[i][j] for i in range(j, tasks)) for j in range(tasks)]
    forgetting = [peaks[j] - final[j] for j in range(tasks - 1)]
    return {
        "experiment": "PHL-DAM continual learning - disjoint-key task sequence",
        "mode": mode,
        "configuration": {
            "scale": task.SCALE,
            "regime": regime,
            "seed": seed,
            "tasks": tasks,
            "keys_per_task": task.NUM_KEYS // tasks,
            "steps_per_task": steps_per_task,
            "batch_size": batch_size,
            "writes_per_episode": writes,
            "eval_episodes_per_task": eval_episodes,
            "learning_rate": learning_rate,
            "trains_after_first_task": mode in ("weights", "both"),
            "carries_memory_across_episodes": mode in ("both", "persistent"),
            "lease_state_present": False,
        },
        "accuracy_after_each_task": accuracy_after,
        "final_accuracy_per_task": final,
        "peak_accuracy_per_task": peaks,
        "forgetting_per_task": forgetting,
        "mean_forgetting": statistics.fmean(forgetting) if forgetting else 0.0,
        "retention_old_tasks": (
            statistics.fmean(final[:-1]) if len(final) > 1 else None
        ),
        "plasticity_last_task": final[-1],
        "mean_final_accuracy": statistics.fmean(final),
        "training_history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "finite": all(torch.isfinite(p).all().item() for p in model.parameters()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--regime", choices=("disjoint", "conflicting"),
                        default="disjoint")
    parser.add_argument("--steps-per-task", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=192)
    parser.add_argument("--writes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.mode, args.seed, args.tasks, args.steps_per_task, args.batch_size,
        args.eval_episodes, args.writes, args.learning_rate, torch.device("cpu"),
        regime=args.regime,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": summary["mode"],
        "final_accuracy_per_task": summary["final_accuracy_per_task"],
        "retention_old_tasks": summary["retention_old_tasks"],
        "plasticity_last_task": summary["plasticity_last_task"],
        "mean_forgetting": summary["mean_forgetting"],
    }, indent=2))


if __name__ == "__main__":
    main()
