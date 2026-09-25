"""Learning curves for every architecture under one identical protocol.

The goal being tested is three-part: does a PHL-DAM variant (1) reach higher
recall than every baseline, (2) get there in fewer steps and less wall-clock
time, and (3) carry less recall loss through training? Final accuracy alone
cannot answer (2) or (3), so this harness records, for any arm:

* held-out probe recall every ``--probe-every`` steps on a fixed 256-episode
  probe drawn from a generator (``seed + 30_000``) that never touches training
  or the final evaluation stream;
* ``steps_to_target`` / ``seconds_to_target``: first probe at or above
  ``--target`` recall (default 0.90). Seconds count training time only;
* ``mean_training_recall_ce``: the mean of the recall-token CE logged at every
  probe point, i.e. the area under the training recall-loss curve;
* final recall on the standard 2,000-episode evaluation (``seed + 20_000``),
  the same stream every published report uses;
* peak resident memory of the process and the inference state size.

Training is untouched: the same ``seed_everything(seed)`` before construction,
the same ``seed + 10_000`` batch stream, objective, AdamW(lr 2e-3, wd 1e-4),
clip 1.0 and batch 16 as every Stage B-era script, so each baseline follows
the same trajectory as its published run. The probe uses its own generator, so
it cannot perturb that trajectory.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from phl_dam_stage_b import (
    PHLDAM,
    _gather_positions,
    common_objective,
    make_batch,
    seed_everything,
)

torch.set_num_threads(1)

PROBE_EPISODES = 256
PROBE_SEED_OFFSET = 30_000
EVAL_SEED_OFFSET = 20_000
TRAIN_SEED_OFFSET = 10_000


def build(arm: str, options: dict):
    if arm == "phl_dam":
        return PHLDAM()
    if arm == "phl_dam_v2":
        from phl_dam_v2 import PHLDAMv2
        return PHLDAMv2(**options)
    if arm in ("ntm_dnc", "ntm_dnc_factorized"):
        from phl_dam_ntm_baseline import build_model
        return build_model(arm == "ntm_dnc_factorized")
    if arm == "transformer":
        from phl_dam_transformer_rematch import CausalTransformer
        return CausalTransformer()
    if arm in ("ssm_selective", "ssm_diagonal"):
        from phl_dam_ssm_baseline import build_model
        return build_model(arm == "ssm_selective")
    raise ValueError(arm)


def logits_of(model, tokens):
    out = model(tokens)
    return out[0] if isinstance(out, tuple) else out


def state_floats(arm: str, model) -> int | None:
    if hasattr(model, "state_floats"):
        return model.state_floats()
    if arm == "phl_dam":
        return 456
    if arm.startswith("ntm"):
        from phl_dam_ntm_baseline import recurrent_state_floats
        return recurrent_state_floats(model)
    if arm.startswith("ssm"):
        return 96
    return None  # the Transformer's KV cache grows with length


@torch.no_grad()
def recall_on(model, seed: int, episodes: int, batch_size: int = 16) -> float:
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    correct = total = 0
    seen = 0
    while seen < episodes:
        count = min(batch_size, episodes - seen)
        seen += count
        batch = make_batch(generator, count)
        predictions = _gather_positions(
            logits_of(model, batch.tokens).argmax(-1), batch.query_key_positions
        )
        targets = _gather_positions(batch.tokens, batch.query_key_positions + 1)
        correct += int(predictions.eq(targets).sum())
        total += targets.numel()
    model.train(was_training)
    return correct / total


def run(arm: str, options: dict, seed: int, steps: int, probe_every: int,
        target: float, eval_episodes: int, batch_size: int = 16,
        learning_rate: float = 2e-3) -> dict:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + TRAIN_SEED_OFFSET)
    model = build(arm, options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    curve = []
    training_seconds = 0.0
    steps_to_target = seconds_to_target = None
    model.train()
    for step in range(1, steps + 1):
        started = time.perf_counter()
        batch = make_batch(generator, batch_size)
        loss, all_ce, recall_ce = common_objective(logits_of(model, batch.tokens), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        training_seconds += time.perf_counter() - started
        if step % probe_every == 0 or step == steps:
            probe = recall_on(model, seed + PROBE_SEED_OFFSET, PROBE_EPISODES)
            curve.append({
                "step": step,
                "recall_ce": recall_ce.item(),
                "all_token_ce": all_ce.item(),
                "probe_recall": probe,
                "gradient_norm": norm,
                "training_seconds": training_seconds,
            })
            print(f"{arm} seed={seed} step={step:4d} recall_ce={recall_ce.item():.4f} "
                  f"probe={probe:.3f} t={training_seconds:.0f}s", flush=True)
            if steps_to_target is None and probe >= target:
                steps_to_target, seconds_to_target = step, training_seconds

    final = recall_on(model, seed + EVAL_SEED_OFFSET, eval_episodes)
    return {
        "experiment": "PHL-DAM learning curves - common protocol",
        "arm": arm,
        "options": options,
        "seed": seed,
        "steps": steps,
        "target_recall": target,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_floats": state_floats(arm, model),
        "steps_to_target": steps_to_target,
        "seconds_to_target": seconds_to_target,
        "training_seconds": training_seconds,
        "seconds_per_step": training_seconds / steps,
        "mean_training_recall_ce": sum(c["recall_ce"] for c in curve) / len(curve),
        "final_recall": final,
        "eval_episodes": eval_episodes,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "finite": all(torch.isfinite(p).all() for p in model.parameters()),
        "curve": curve,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--options", default="{}", help="JSON kwargs for phl_dam_v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--probe-every", type=int, default=25)
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.arm, json.loads(args.options), args.seed, args.steps,
                  args.probe_every, args.target, args.eval_episodes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))


if __name__ == "__main__":
    main()
