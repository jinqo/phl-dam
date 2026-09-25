"""Seconds per training step for every arm, measured round-robin.

Wall-clock numbers from runs that share a machine with other jobs drift with
the load. Here every arm does one full training step (forward, backward,
clip, AdamW) on the same batch in turn, for several rounds, inside one
single-threaded process, so any drift in machine load hits every arm alike.
The median over rounds is reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from phl_dam_learning_curve import build, logits_of
from phl_dam_stage_b import common_objective, make_batch

torch.set_num_threads(1)

ARMS = {
    "v2_S": ("phl_dam_v2", {"fast": True, "normalized_values": True,
                            "use_phl": False, "d_model": 76}),
    "v2_T": ("phl_dam_v2", {"fast": True, "normalized_values": True,
                            "use_phl": False, "d_model": 84, "num_slots": 4,
                            "d_key": 8, "d_value": 8}),
    "phl_dam": ("phl_dam", {}),
    "ntm_dnc_factorized": ("ntm_dnc_factorized", {}),
    "ntm_dnc": ("ntm_dnc", {}),
    "transformer": ("transformer", {}),
    "ssm_selective": ("ssm_selective", {}),
    "ssm_diagonal": ("ssm_diagonal", {}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    batch = make_batch(torch.Generator().manual_seed(0), 16)
    models, optimizers = {}, {}
    for name, (arm, options) in ARMS.items():
        torch.manual_seed(0)
        models[name] = build(arm, options)
        optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=2e-3,
                                             weight_decay=1e-4)
    timings = {name: [] for name in ARMS}
    for round_index in range(args.rounds + 2):
        for name, model in models.items():
            started = time.perf_counter()
            loss = common_objective(logits_of(model, batch.tokens), batch)[0]
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            if round_index >= 2:          # two warm-up rounds
                timings[name].append(time.perf_counter() - started)
    summary = {
        name: {"median_seconds_per_step": statistics.median(values),
               "min_seconds_per_step": min(values), "rounds": len(values)}
        for name, values in timings.items()
    }
    for name, row in summary.items():
        print(f"{name:20s} median {row['median_seconds_per_step']:.3f} s/step")
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
