"""PHL-DAM-004D: why does the write/content controller collapse under write load?

PHL-DAM-004C established that the full-scale failure is *associated* with high
slot over-subscription but could not isolate it, because write count and
sequence length are confounded by construction in the earlier profiles. The
``pressure`` scale profile removes that confound: sequence length (456), delay
distribution (32-256), query budget and live-item count are pinned, and the only
thing that varies across levels is how many never-queried distractor writes
compete for the same eight slots.

This script trains one model per (write level, seed) and records controller
telemetry *through* training, not just at the end, so the ordering of events in
a collapse can be read off rather than assumed:

    gradient explosion -> gate failure -> thrashing
        versus
    gate failure -> thrashing -> gradient explosion

Nothing here tests the lease hypothesis, which was settled in PHL-DAM-004B-S.
The arm is fixed to ``content_only``: the object of study is the write/content
controller itself.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor

import phl_dam_pressure_task as task
import phl_dam_004b_lease as lease_module
from phl_dam_004b_lease import (
    PHLDAMLease,
    common_objective,
    evaluate_arm,
    readback_objective,
    pack_batch,
    parameter_report,
    seed_everything,
)

torch.set_num_threads(1)


BACKBONE = "phl"
READBACK_WEIGHT = 0.0
PROBE_EPISODES = 32
TELEMETRY_EVERY = 25
GRADIENT_THRESHOLDS = (1e3, 1e5, 1e7, 1e9)

# Preregistered, frozen before any 004D comparison was run. A seed counts as
# content-path-informative if that seed's lowest-pressure run (the baseline
# level) crosses the breakthrough threshold. This is decided by the baseline
# level alone, never by the condition under test, so it cannot be used to drop
# seeds that happen to be inconvenient for a contrast.
BREAKTHROUGH_RECALL_CE = 2.0
INFORMATIVE_BASELINE_LEVEL = 8


def probe_batch(seed: int, writes: int, device: torch.device):
    """A fixed held-out batch used only for telemetry, never for updates."""
    episodes = [
        task.generate_episode(seed + 900_000, index, writes, "canonical")
        for index in range(PROBE_EPISODES)
    ]
    batch = pack_batch(episodes, device)
    binding = torch.zeros_like(batch.tokens, dtype=torch.bool)
    for row, episode in enumerate(episodes):
        for item in episode.items:
            binding[row, item.write_value_position] = True
    return batch, binding, episodes


@torch.no_grad()
def controller_telemetry(
    model: PHLDAMLease, batch, binding: Tensor, episodes
) -> dict[str, float]:
    """Snapshot of what the write controller and allocator are doing."""
    model.eval()
    _, diagnostics = model(batch.tokens, arm="content_only", collect=True)
    model.train()

    write = diagnostics["write_strength"]
    entropy = diagnostics["allocation_entropy"]
    margin = diagnostics["allocation_margin"]
    occupied = diagnostics["occupied_count"]
    owner = diagnostics["owner_token"]

    at_binding = write[binding]
    elsewhere = write[~binding]
    committed = write > 0.5
    commitments = int(committed.sum())
    committed_off_binding = int((committed & ~binding).sum())

    # Residency: was each queried key present in the ledger at query time?
    rows = torch.arange(batch.tokens.shape[0])[:, None]
    owner_at_query = owner[rows, batch.query_key_positions]
    resident = (
        owner_at_query.eq(batch.query_key_tokens[:, :, None]).any(dim=-1)
        & batch.query_valid
    )

    # Slot replacement: how often the ledger's owner changes step to step.
    changed = (owner[:, 1:] != owner[:, :-1]).any(dim=-1).float().mean()

    return {
        "write_gate_at_binding": float(at_binding.mean()),
        "write_gate_elsewhere": float(elsewhere.mean()),
        "write_selectivity": float(at_binding.mean() - elsewhere.mean()),
        "write_commitments": commitments,
        "commitments_off_binding_fraction": (
            committed_off_binding / commitments if commitments else 0.0
        ),
        "allocation_entropy_at_binding": float(entropy[binding].mean()),
        "allocation_entropy_elsewhere": float(entropy[~binding].mean()),
        "allocation_entropy_max_possible": math.log(model.num_slots),
        "allocation_margin_at_binding": float(margin[binding].mean()),
        "mean_occupied_slots": float(occupied.mean()),
        "slot_replacement_rate": float(changed),
        "residency_estimate": float(
            resident.sum() / batch.query_valid.sum().clamp_min(1)
        ),
    }


def train_with_telemetry(
    writes: int,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[PHLDAMLease, list[dict], dict]:
    seed_everything(seed)
    model = PHLDAMLease(arm="content_only", backbone=BACKBONE).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    batch, binding, episodes = probe_batch(seed, writes, device)

    history: list[dict] = []
    started = time.perf_counter()
    diverged = {
        "finite": True,
        "first_nonfinite_step": None,
        "failure_reason": None,
        "max_gradient_norm": 0.0,
        "first_step_gradient_above": {str(t): None for t in GRADIENT_THRESHOLDS},
        "last_valid_step": None,
    }
    last_valid: dict | None = None
    model.train()

    for step in range(1, steps + 1):
        episodes_step = [
            task.generate_episode(seed + 500_000, step * batch_size + i, writes, "canonical")
            for i in range(batch_size)
        ]
        training_batch = pack_batch(episodes_step, device)
        logits, _ = model(training_batch.tokens, arm="content_only")
        loss, all_ce, recall_ce = common_objective(logits, training_batch)
        readback = torch.zeros((), device=device)
        if READBACK_WEIGHT > 0.0:
            readback = readback_objective(model)
            loss = loss + READBACK_WEIGHT * readback

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm = float(gradient_norm)

        if math.isfinite(norm):
            diverged["max_gradient_norm"] = max(diverged["max_gradient_norm"], norm)
            for threshold in GRADIENT_THRESHOLDS:
                key = str(threshold)
                if diverged["first_step_gradient_above"][key] is None and norm > threshold:
                    diverged["first_step_gradient_above"][key] = step

        # Divergence protection: detect, record, and keep going rather than
        # silently discarding the run.
        problems = []
        if not torch.isfinite(loss):
            problems.append("loss")
        if not math.isfinite(norm):
            problems.append("gradient")
        if problems and diverged["finite"]:
            diverged.update(
                finite=False,
                first_nonfinite_step=step,
                failure_reason="+".join(problems),
                last_valid_step=last_valid["step"] if last_valid else None,
            )

        optimizer.step()

        if diverged["finite"] and not all(
            torch.isfinite(p).all() for p in model.parameters()
        ):
            diverged.update(
                finite=False,
                first_nonfinite_step=step,
                failure_reason="parameters",
                last_valid_step=last_valid["step"] if last_valid else None,
            )

        if step == 1 or step % TELEMETRY_EVERY == 0 or step == steps:
            record = {
                "step": step,
                "loss": float(loss),
                "all_token_ce": float(all_ce),
                "recall_ce": float(recall_ce),
                "readback": float(readback.detach()),
                "gradient_norm": norm,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if all(torch.isfinite(p).all() for p in model.parameters()):
                record.update(controller_telemetry(model, batch, binding, episodes))
                last_valid = record
            history.append(record)
            print(
                f"W={writes} seed={seed} step={step:4d} "
                f"recall_ce={record['recall_ce']:.4f} grad={norm:.3g} "
                f"sel={record.get('write_selectivity', float('nan')):.3f} "
                f"occ={record.get('mean_occupied_slots', float('nan')):.2f} "
                f"ent={record.get('allocation_entropy_at_binding', float('nan')):.3f}",
                flush=True,
            )

    diverged["last_valid_metrics"] = last_valid
    return model, history, diverged


def run(
    writes: int,
    seed: int,
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
    device: torch.device,
) -> dict:
    model, history, divergence = train_with_telemetry(
        writes, seed, steps, batch_size, learning_rate, device
    )
    finite_model = all(torch.isfinite(p).all() for p in model.parameters())
    metrics = {}
    if finite_model:
        metrics = evaluate_arm(
            model,
            "content_only",
            seed=seed,
            writes=writes,
            condition="canonical",
            episodes=eval_episodes,
            batch_size=batch_size,
            device=device,
        )

    finished = [r for r in history if "write_selectivity" in r]
    breakthrough = next(
        (r["step"] for r in history if r["recall_ce"] < BREAKTHROUGH_RECALL_CE), None
    )
    return {
        "experiment": "PHL-DAM-004D - Write-pressure ladder at fixed sequence length",
        "model": "content_only",
        "configuration": {
            "scale": task.SCALE,
            "backbone": model.backbone,
            "readback_weight": READBACK_WEIGHT,
            "seed": seed,
            "writes": writes,
            "sequence_length": task.SEQUENCE_LENGTH,
            "delay_range": [task.MIN_DELAY, task.MAX_DELAY],
            "query_budget": list(task.QUERY_BUDGET["canonical"][writes]),
            "slots": model.num_slots,
            "training_steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "AdamW(weight_decay=1e-4)",
            "merge_sharpness": lease_module.MERGE_SHARPNESS,
            "merge_temperature": lease_module.MERGE_TEMPERATURE,
            "eval_episodes": eval_episodes,
            "telemetry_every": TELEMETRY_EVERY,
            "breakthrough_recall_ce": BREAKTHROUGH_RECALL_CE,
            "informative_baseline_level": INFORMATIVE_BASELINE_LEVEL,
            "lease_state_present": False,
            "timing_supervision_weight": 0.0,
            "future_use_labels_to_learned_arms": False,
        },
        "accounting": parameter_report(model),
        "breakthrough_step": breakthrough,
        "final_recall_ce": history[-1]["recall_ce"] if history else None,
        "divergence": divergence,
        "final_controller": finished[-1] if finished else None,
        "metrics": metrics,
        "training_history": history,
        "finite": finite_model and divergence["finite"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    # Identified by controlled ablation in phl_dam_gradient_attribution.py as
    # the dominant source of the 1e11 gradient spikes. Default reproduces the
    # published behaviour exactly.
    parser.add_argument("--backbone", choices=("phl", "ssm", "none"), default="phl")
    # Self-supervised read-back consistency: gives the write path a signal
    # that does not have to survive eviction and a long delay first.
    parser.add_argument("--readback-weight", type=float, default=0.0)
    parser.add_argument("--merge-sharpness", type=float, default=12.0)
    parser.add_argument("--merge-temperature", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global BACKBONE, READBACK_WEIGHT
    BACKBONE = args.backbone
    READBACK_WEIGHT = args.readback_weight
    lease_module.MERGE_SHARPNESS = args.merge_sharpness
    lease_module.MERGE_TEMPERATURE = args.merge_temperature
    task.set_scale("pressure")
    if args.writes not in task.PRESSURE_LEVELS:
        raise SystemExit(f"--writes must be one of {task.PRESSURE_LEVELS}")
    summary = run(
        writes=args.writes,
        seed=args.seed,
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
    print(json.dumps({k: summary[k] for k in ("breakthrough_step", "final_recall_ce", "finite")}))


if __name__ == "__main__":
    main()
