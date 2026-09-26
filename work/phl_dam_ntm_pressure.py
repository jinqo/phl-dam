"""NTM/DNC baselines on the PHL-DAM-004 write-pressure ladder.

At 176 tokens with three bindings and eight slots nothing is ever evicted, so
that benchmark cannot exercise the part of PHL-DAM that differs most from a
DNC: its merge/allocation/eviction rules under slot competition. The 004D/004G
ladder can. This runs the plain and the role-factorized DNC from
``phl_dam_ntm_baseline`` on exactly the same pressure episodes as
``phl_dam_004d_write_pressure``:

* same training stream (``seed + 500_000``), same held-out evaluation stream
  (``seed + EVAL_SEED_OFFSET``), same objective, optimiser, clip, lr, batch,
  700 steps and 192 evaluation episodes;
* same "learned" criterion as ``phl_dam_report_stats.learned``: the final
  logged training recall CE ends below 2.0;
* parameters matched within 1% of PHL-DAM's 38,641 on this vocabulary.

Recall accuracy is computed the same way ``evaluate_arm`` computes ``recall``:
argmax at each valid query-key position against the bound value.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import phl_dam_pressure_task as task
from phl_dam_004b_lease import EVAL_SEED_OFFSET, common_objective, pack_batch, seed_everything
from phl_dam_ntm_baseline import NTMBaseline, active_parameter_count, recurrent_state_floats

torch.set_num_threads(1)

PHL_DAM_PRESSURE_PARAMETERS = 38_641
BREAKTHROUGH_RECALL_CE = 2.0
LOG_EVERY = 25
# Controller widths that land each arm within 1% of PHL-DAM on vocabulary 87.
CONTROLLER_WIDTH = {"ntm_dnc": 70, "ntm_dnc_factorized": 100}


ARMS = tuple(CONTROLLER_WIDTH) + ("phl_dam_v2",)


def build(arm: str, options: dict | None = None):
    if arm == "phl_dam_v2":
        from phl_dam_v2 import PHLDAMv2
        return PHLDAMv2(**{**(options or {}), "vocab_size": task.VOCAB_SIZE})
    return NTMBaseline(
        factorized=arm == "ntm_dnc_factorized",
        controller_width=CONTROLLER_WIDTH[arm],
        vocab_size=task.VOCAB_SIZE,
    )


def forward(model, tokens, disable_memory: bool = False):
    if isinstance(model, NTMBaseline):
        return model(tokens, disable_memory=disable_memory)
    return model(tokens, disable_retrieval=disable_memory)[0]


def _gather(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return values.gather(1, positions)


@torch.no_grad()
def evaluate(model: NTMBaseline, seed: int, writes: int, episodes: int, batch_size: int) -> dict:
    model.eval()
    correct = disabled = queries = 0
    all_ce = recall_ce = 0.0
    batches = seen = 0
    while seen < episodes:
        count = min(batch_size, episodes - seen)
        generated = [
            task.generate_episode(seed + EVAL_SEED_OFFSET, seen + index, writes, "canonical")
            for index in range(count)
        ]
        seen += count
        batch = pack_batch(generated, torch.device("cpu"))
        logits = forward(model, batch.tokens)
        ablated = forward(model, batch.tokens, disable_memory=True)
        _, a, r = common_objective(logits, batch)
        all_ce += a.item()
        recall_ce += r.item()
        batches += 1
        valid = batch.query_valid
        hit = _gather(logits.argmax(-1), batch.query_key_positions).eq(batch.query_target_tokens)
        miss = _gather(ablated.argmax(-1), batch.query_key_positions).eq(batch.query_target_tokens)
        correct += int((hit & valid).sum())
        disabled += int((miss & valid).sum())
        queries += int(valid.sum())
    return {
        "writes": writes,
        "condition": "canonical",
        "episodes": episodes,
        "queries": queries,
        "recall": correct / queries,
        "retrieval_disabled_recall": disabled / queries,
        "all_token_ce": all_ce / batches,
        "recall_token_ce": recall_ce / batches,
    }


def run(arm: str, writes: int, seed: int, steps: int, batch_size: int,
        eval_episodes: int, learning_rate: float, options: dict | None = None) -> dict:
    seed_everything(seed)
    model = build(arm, options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    max_norm = 0.0
    finite = True
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        episodes = [
            task.generate_episode(seed + 500_000, step * batch_size + i, writes, "canonical")
            for i in range(batch_size)
        ]
        batch = pack_batch(episodes, torch.device("cpu"))
        loss, all_ce, recall_ce = common_objective(forward(model, batch.tokens), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if math.isfinite(norm):
            max_norm = max(max_norm, norm)
        else:
            finite = False
        optimizer.step()
        if step == 1 or step % LOG_EVERY == 0 or step == steps:
            record = {
                "step": step,
                "loss": loss.item(),
                "all_token_ce": float(all_ce),
                "recall_ce": float(recall_ce),
                "gradient_norm": norm,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(f"{arm} W={writes} seed={seed} step={step:4d} "
                  f"recall_ce={record['recall_ce']:.4f} grad={norm:.3g}", flush=True)

    finite = finite and all(torch.isfinite(p).all() for p in model.parameters())
    metrics = evaluate(model, seed, writes, eval_episodes, batch_size) if finite else {}
    return {
        "experiment": "PHL-DAM-004 write-pressure ladder - NTM/DNC baselines",
        "model": arm,
        "configuration": {
            "scale": task.SCALE,
            "seed": seed,
            "writes": writes,
            "sequence_length": task.SEQUENCE_LENGTH,
            "delay_range": [task.MIN_DELAY, task.MAX_DELAY],
            "query_budget": list(task.QUERY_BUDGET["canonical"][writes]),
            "options": options or {},
            "slots": getattr(model, "slots", getattr(model, "num_slots", None)),
            "controller_width": getattr(model, "controller_width", None),
            "factorized_roles": getattr(model, "factorized", True),
            "training_steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "AdamW(weight_decay=1e-4)",
            "eval_episodes": eval_episodes,
            "breakthrough_recall_ce": BREAKTHROUGH_RECALL_CE,
        },
        "accounting": {
            "total_parameters": active_parameter_count(model),
            "phl_dam_reference_parameters": PHL_DAM_PRESSURE_PARAMETERS,
            "recurrent_state_floats": (
                model.state_floats() if hasattr(model, "state_floats")
                else recurrent_state_floats(model)
            ),
        },
        "breakthrough_step": next(
            (r["step"] for r in history if r["recall_ce"] < BREAKTHROUGH_RECALL_CE), None
        ),
        "final_recall_ce": history[-1]["recall_ce"],
        "max_gradient_norm": max_norm,
        "metrics": metrics,
        "training_history": history,
        "finite": finite,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="ntm_dnc_factorized")
    parser.add_argument("--options", default="{}", help="JSON kwargs for phl_dam_v2")
    parser.add_argument("--writes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task.set_scale("pressure")
    if args.writes not in task.PRESSURE_LEVELS:
        raise SystemExit(f"--writes must be one of {task.PRESSURE_LEVELS}")
    summary = run(args.arm, args.writes, args.seed, args.steps, args.batch_size,
                  args.eval_episodes, args.learning_rate, json.loads(args.options))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("breakthrough_step", "final_recall_ce", "finite")}
                     | {"recall": summary["metrics"].get("recall")}))


if __name__ == "__main__":
    main()
