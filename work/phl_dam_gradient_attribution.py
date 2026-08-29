"""Where do PHL-DAM's 1e11-scale gradients come from?

PHL-DAM-004D found gradient norms reaching 1e19, crossing 1e3 as early as step
1-3, and cleanly separating runs that learn (median max 7.75e6) from runs that
do not (median 1.56e11). That instability contaminates every write-pressure
level including the baseline, so it has to be understood before any scaling or
architecture comparison is meaningful.

This script attributes the gradient norm to individual parameter groups and to
individual timesteps, so the source can be named rather than guessed at. It
trains nothing: it takes a fresh model, runs a handful of updates, and records
where the magnitude lives.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

import phl_dam_pressure_task as task
from phl_dam_004b_lease import PHLDAMLease, common_objective, pack_batch, seed_everything


GROUPS = (
    ("embedding", ("token_embedding",)),
    ("context", ("context_encoder",)),
    ("phl", ("phl_input", "phl_readout", "phl_norm")),
    ("key", ("key_projection",)),
    ("value", ("value_projection",)),
    ("query", ("query_projection",)),
    ("write_gate", ("write_gate",)),
    ("read_gate", ("read_gate",)),
    ("memory_projection", ("memory_projection",)),
    ("output", ("output", "output_norm")),
    ("eviction_scorer", ("content_scorer", "utility_scorer", "lease_", "eviction_")),
)


def group_of(name: str) -> str:
    for label, prefixes in GROUPS:
        if any(name.startswith(prefix) for prefix in prefixes):
            return label
    return "other"


def per_group_norms(model: PHLDAMLease) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        totals[group_of(name)] += float(parameter.grad.detach().pow(2).sum())
    return {label: math.sqrt(value) for label, value in sorted(totals.items())}


def timestep_gradient_profile(
    model: PHLDAMLease, tokens: torch.Tensor, probes: int = 12
) -> list[dict[str, float]]:
    """Gradient of the loss w.r.t. the hidden state at evenly spaced timesteps.

    A recurrence that amplifies will show the norm growing as it is carried
    backwards; a local blow-up will show a spike at particular steps.
    """
    context, previous, current = model.encode_features(tokens)
    state = model.init_state(tokens.shape[0], tokens.device)
    length = tokens.shape[1]
    marks = sorted({int(round(i * (length - 1) / (probes - 1))) for i in range(probes)})

    retained: dict[int, torch.Tensor] = {}
    logits = []
    for step in range(length):
        if step in marks and state.dam.keys.requires_grad:
            state.dam.keys.retain_grad()
            retained[step] = state.dam.keys
        previous_ids = tokens[:, step - 1] if step > 0 else torch.full_like(tokens[:, 0], -1)
        step_logits, state, _ = model.step(
            context[:, step],
            previous[:, step],
            current[:, step],
            state,
            arm="content_only",
            previous_token_ids=previous_ids,
            current_token_ids=tokens[:, step],
        )
        logits.append(step_logits)
    torch.stack(logits, dim=1).pow(2).mean().backward()

    profile = []
    for step in sorted(retained):
        grad = retained[step].grad
        profile.append(
            {
                "timestep": step,
                "slot_key_grad_norm": float(grad.norm()) if grad is not None else None,
            }
        )
    return profile


def run(writes: int, seed: int, steps: int, batch_size: int, scale: str) -> dict:
    task.set_scale(scale)
    seed_everything(seed)
    model = PHLDAMLease(arm="content_only")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    history = []
    for step in range(1, steps + 1):
        episodes = [
            task.generate_episode(seed + 500_000, step * batch_size + i, writes, "canonical")
            for i in range(batch_size)
        ]
        batch = pack_batch(episodes, torch.device("cpu"))
        logits, _ = model(batch.tokens, arm="content_only")
        loss, all_ce, recall_ce = common_objective(logits, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        groups = per_group_norms(model)
        total = math.sqrt(sum(v * v for v in groups.values()))
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "total_grad_norm": total,
                "group_grad_norms": groups,
                "dominant_group": max(groups, key=groups.get) if groups else None,
                "dominant_share": (
                    max(groups.values()) / total if groups and total > 0 else None
                ),
            }
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(
            f"step={step:3d} total={total:.4g} "
            f"dominant={history[-1]['dominant_group']} "
            f"share={history[-1]['dominant_share']:.3f}",
            flush=True,
        )

    seed_everything(seed)
    fresh = PHLDAMLease(arm="content_only")
    episodes = [
        task.generate_episode(seed + 500_000, i, writes, "canonical")
        for i in range(min(batch_size, 8))
    ]
    profile = timestep_gradient_profile(fresh, pack_batch(episodes, torch.device("cpu")).tokens)

    return {
        "experiment": "PHL-DAM gradient attribution",
        "configuration": {
            "scale": scale,
            "writes": writes,
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "sequence_length": task.SEQUENCE_LENGTH,
        },
        "history": history,
        "slot_key_gradient_by_timestep": profile,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--scale", default="pressure")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.writes, args.seed, args.steps, args.batch_size, args.scale)
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("\n=== slot-key gradient by timestep (backward amplification probe) ===")
    for row in summary["slot_key_gradient_by_timestep"]:
        print(f"  t={row['timestep']:4d} |dL/dKeys|={row['slot_key_grad_norm']:.6g}")


if __name__ == "__main__":
    main()
