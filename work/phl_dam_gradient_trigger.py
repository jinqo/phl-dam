"""Localise PHL-DAM's intermittent gradient explosion by bisection.

Three hypotheses about this instability have now been tested and refuted:

* recurrent amplification through the slot keys - backward gradients *decay*
  (0.003 -> 4e-5 across 456 steps);
* the log-occupancy term in the read score - widening its epsilon made peak
  gradients worse (1.0e11 -> 3.0e11);
* the merge path - softening it cut a 4-step ablation from 1.0e11 to 3.0e5 but
  made a full 700-step run worse (median max gradient 204 -> 1.12e13).

That last one is the cautionary case: a short-horizon ablation does not predict
training-horizon stability. So this script stops proposing causes and instead
finds the exact (batch, episode, timestep) that produces a spike, by bisection:

1. train until a step whose gradient norm exceeds a threshold;
2. re-run that batch one episode at a time to find which episode carries it;
3. truncate that episode's loss to increasing prefixes to find the timestep
   where the norm jumps;
4. dump the model and task state at that timestep for inspection.

Nothing is concluded here. The output is a location and a state dump.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import phl_dam_pressure_task as task
from phl_dam_004b_lease import (
    PHLDAMLease,
    common_objective,
    pack_batch,
    seed_everything,
)

torch.set_num_threads(1)


def gradient_norm_for(model: PHLDAMLease, tokens: torch.Tensor, batch) -> float:
    model.zero_grad(set_to_none=True)
    logits, _ = model(tokens, arm="content_only")
    loss, _, _ = common_objective(logits, batch)
    loss.backward()
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    model.zero_grad(set_to_none=True)
    return math.sqrt(total)


def prefix_gradient_norm(
    model: PHLDAMLease, tokens: torch.Tensor, upto: int
) -> float:
    """Gradient of a loss that only sees the first ``upto`` timesteps."""
    model.zero_grad(set_to_none=True)
    logits, _ = model(tokens, arm="content_only")
    truncated = logits[:, :upto]
    truncated.pow(2).mean().backward()
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    model.zero_grad(set_to_none=True)
    return math.sqrt(total)


@torch.no_grad()
def episode_signature(model: PHLDAMLease, tokens: torch.Tensor) -> dict:
    """What is structurally distinctive about this episode at this moment."""
    _, diagnostics = model(tokens, arm="content_only", collect=True)
    occupancy = diagnostics["occupancy"]
    write = diagnostics["write_strength"]
    entropy = diagnostics["allocation_entropy"]
    margin = diagnostics["allocation_margin"]
    return {
        "occupancy_min": float(occupancy.min()),
        "occupancy_max": float(occupancy.max()),
        "occupancy_mean": float(occupancy.mean()),
        "write_strength_max": float(write.max()),
        "write_strength_mean": float(write.mean()),
        "allocation_entropy_min": float(entropy.min()),
        "allocation_entropy_max": float(entropy.max()),
        "allocation_margin_min": float(margin.min()),
        "steps_with_tiny_margin": int((margin < 1e-3).sum()),
        "steps_with_tiny_occupancy": int((occupancy < 1e-4).sum()),
    }


def run(
    writes: int,
    seed: int,
    max_steps: int,
    batch_size: int,
    threshold: float,
    scale: str,
) -> dict:
    task.set_scale(scale)
    seed_everything(seed)
    model = PHLDAMLease(arm="content_only")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    trace = []
    spike_step = None
    spike_tokens = None
    spike_batch = None
    for step in range(1, max_steps + 1):
        episodes = [
            task.generate_episode(seed + 500_000, step * batch_size + i, writes, "canonical")
            for i in range(batch_size)
        ]
        batch = pack_batch(episodes, torch.device("cpu"))
        logits, _ = model(batch.tokens, arm="content_only")
        loss, _, _ = common_objective(logits, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        trace.append({"step": step, "loss": float(loss.detach()), "gradient_norm": norm})
        print(f"step={step:4d} grad={norm:.4g}", flush=True)
        if norm > threshold and spike_step is None:
            spike_step = step
            spike_tokens = batch.tokens.clone()
            spike_batch = batch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            break
        optimizer.step()

    if spike_step is None:
        return {"found": False, "trace": trace}

    # 2. Which episode in the batch carries the spike?
    model.load_state_dict(state)
    per_episode = []
    for index in range(spike_tokens.shape[0]):
        single = pack_batch(
            [
                task.generate_episode(
                    seed + 500_000, spike_step * batch_size + index, writes, "canonical"
                )
            ],
            torch.device("cpu"),
        )
        per_episode.append(
            {
                "index": index,
                "gradient_norm": gradient_norm_for(model, single.tokens, single),
            }
        )
    worst = max(per_episode, key=lambda row: row["gradient_norm"])

    # 3. Which timestep does it jump at?
    culprit = pack_batch(
        [
            task.generate_episode(
                seed + 500_000, spike_step * batch_size + worst["index"], writes, "canonical"
            )
        ],
        torch.device("cpu"),
    )
    length = culprit.tokens.shape[1]
    marks = sorted({int(round(f * length)) for f in
                    (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)} - {0})
    profile = [
        {"upto": upto, "gradient_norm": prefix_gradient_norm(model, culprit.tokens, upto)}
        for upto in marks
    ]
    jump = None
    for earlier, later in zip(profile, profile[1:]):
        if earlier["gradient_norm"] > 0 and later["gradient_norm"] / max(
            earlier["gradient_norm"], 1e-30
        ) > 100:
            jump = {"from": earlier["upto"], "to": later["upto"],
                    "ratio": later["gradient_norm"] / earlier["gradient_norm"]}
            break

    return {
        "found": True,
        "experiment": "PHL-DAM gradient trigger localisation",
        "configuration": {
            "scale": scale, "writes": writes, "seed": seed,
            "batch_size": batch_size, "threshold": threshold,
            "sequence_length": task.SEQUENCE_LENGTH,
        },
        "spike_step": spike_step,
        "spike_gradient_norm": trace[-1]["gradient_norm"],
        "trace": trace,
        "per_episode_gradient": per_episode,
        "worst_episode": worst,
        "prefix_profile": profile,
        "first_large_jump": jump,
        "worst_episode_signature": episode_signature(model, culprit.tokens),
        "batch_mean_gradient": sum(r["gradient_norm"] for r in per_episode) / len(per_episode),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=1e6)
    parser.add_argument("--scale", default="pressure")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.writes, args.seed, args.max_steps, args.batch_size, args.threshold, args.scale
    )
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["found"]:
        print(f"\nno spike above threshold in {args.max_steps} steps")
        return
    print(f"\n=== spike at step {summary['spike_step']}, "
          f"norm {summary['spike_gradient_norm']:.4g} ===")
    print("per-episode gradient norms:")
    for row in summary["per_episode_gradient"]:
        mark = "  <== worst" if row["index"] == summary["worst_episode"]["index"] else ""
        print(f"  episode {row['index']:2d}: {row['gradient_norm']:.6g}{mark}")
    print(f"\nbatch mean {summary['batch_mean_gradient']:.4g}, "
          f"worst {summary['worst_episode']['gradient_norm']:.4g}")
    print("\nprefix profile (loss truncated to first N timesteps):")
    for row in summary["prefix_profile"]:
        print(f"  upto={row['upto']:4d}  grad={row['gradient_norm']:.6g}")
    print(f"\nfirst >100x jump: {summary['first_large_jump']}")
    print("\nworst-episode signature:")
    for key, value in summary["worst_episode_signature"].items():
        print(f"  {key:28s} {value}")


if __name__ == "__main__":
    main()
