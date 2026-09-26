"""Apply a v2 preregistration to its learning curves.

``--round 1`` (default): PREREGISTRATION_v2_confirmation.md, seeds 0-5, files
``phl_dam_v2conf_{arm}_seed{seed}.json``. Its seconds criterion is applied as
written, including the flaw that charges never-converging arms 510 steps.

``--round 2``: PREREGISTRATION_v2_round2.md, seeds 6-11, files
``phl_dam_v2r2_{arm}_seed{seed}.json``, with the seconds criterion fixed (any
seed that never reaches 90% makes time-to-90% infinite) and the copy-readout
DNC reported as a control.

Every criterion for every candidate against every baseline is reported, pass
or fail, exactly as preregistered.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from phl_dam_report_stats import paired_verdict

BASELINES = ("phl_dam", "ntm_dnc_factorized", "ntm_dnc", "transformer",
             "ssm_selective", "ssm_diagonal")
ROUNDS = {
    1: {"candidates": ("v2_S", "v2_T"), "controls": (), "seeds": range(6),
        "prefix": "phl_dam_v2conf", "infinite_if_never": False,
        "timing": "phl_dam_v2_timing_benchmark.json",
        "state_limit": {"v2_S": 392, "v2_T": 95}},
    2: {"candidates": ("v2_TC", "v2_SC"), "controls": ("ntm_dnc_factorized_copy",),
        "seeds": range(6, 12), "prefix": "phl_dam_v2r2", "infinite_if_never": True,
        "timing": "phl_dam_v2_timing_benchmark_round2.json",
        "state_limit": {"v2_TC": 95, "v2_SC": 392}},
}
NEVER = 510            # steps charged to a run that never reaches 90%
CEILING = 0.995
CEILING_MARGIN = -0.005
CANDIDATES: tuple = ()
SEEDS: range = range(0)
STATE_LIMIT: dict = {}


def load(outputs: Path, prefix: str, arms: tuple) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for arm in arms:
        for seed in SEEDS:
            path = outputs / f"{prefix}_{arm}_seed{seed}.json"
            if path.exists():
                runs.setdefault(arm, {})[seed] = json.loads(path.read_text())
    return runs


def steps90(run: dict) -> int:
    return run["steps_to_target"] if run["steps_to_target"] is not None else NEVER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("../outputs"))
    parser.add_argument("--write", type=Path)
    parser.add_argument("--round", type=int, choices=tuple(ROUNDS), default=1)
    args = parser.parse_args()
    global CANDIDATES, SEEDS, STATE_LIMIT
    config = ROUNDS[args.round]
    CANDIDATES, SEEDS, STATE_LIMIT = (config["candidates"], config["seeds"],
                                      config["state_limit"])
    runs = load(args.outputs, config["prefix"],
                CANDIDATES + BASELINES + config["controls"])
    timing = json.loads((args.outputs / config["timing"]).read_text())

    arms = {}
    for arm, by_seed in runs.items():
        seeds = sorted(by_seed)
        s90 = [steps90(by_seed[s]) for s in seeds]
        per_step = timing[arm]["median_seconds_per_step"]
        arms[arm] = {
            "seeds": seeds,
            "parameters": by_seed[seeds[0]]["parameters"],
            "state_floats": by_seed[seeds[0]]["state_floats"],
            "final_recall": [by_seed[s]["final_recall"] for s in seeds],
            "mean_final_recall": statistics.fmean(by_seed[s]["final_recall"] for s in seeds),
            "steps_to_90": s90,
            "mean_steps_to_90": statistics.fmean(s90),
            "reached_90": sum(by_seed[s]["steps_to_target"] is not None for s in seeds),
            "median_seconds_per_step": per_step,
            "est_seconds_to_90": (
                float("inf") if config["infinite_if_never"] and NEVER in s90
                else statistics.fmean(s90) * per_step
            ),
            "mean_training_recall_ce": statistics.fmean(
                by_seed[s]["mean_training_recall_ce"] for s in seeds),
            "peak_rss_mb": max(by_seed[s]["peak_rss_mb"] for s in seeds),
        }

    verdicts = {}
    for candidate in CANDIDATES:
        if candidate not in runs:
            continue
        c = arms[candidate]
        rows = {}
        for baseline in BASELINES:
            if baseline not in runs:
                continue
            b = arms[baseline]
            shared = sorted(set(runs[candidate]) & set(runs[baseline]))
            acc = paired_verdict([runs[candidate][s]["final_recall"]
                                  - runs[baseline][s]["final_recall"] for s in shared])
            if b["mean_final_recall"] >= CEILING:
                accuracy_pass = acc["mean"] > CEILING_MARGIN
                accuracy_rule = "baseline at ceiling: not worse than -0.5 pp"
            else:
                accuracy_pass = (c["mean_final_recall"] >= b["mean_final_recall"]
                                 and acc["robust"])
                accuracy_rule = "mean higher and robust paired win"
            faster_seeds = sum(steps90(runs[candidate][s]) <= steps90(runs[baseline][s])
                               for s in shared)
            steps_pass = (c["mean_steps_to_90"] < b["mean_steps_to_90"]
                          and faster_seeds >= 5)
            time_pass = c["est_seconds_to_90"] < b["est_seconds_to_90"]
            loss = paired_verdict([runs[baseline][s]["mean_training_recall_ce"]
                                   - runs[candidate][s]["mean_training_recall_ce"]
                                   for s in shared])
            loss_pass = (c["mean_training_recall_ce"] < b["mean_training_recall_ce"]
                         and loss["robust"])
            rows[baseline] = {
                "accuracy": {"pass": accuracy_pass, "rule": accuracy_rule,
                             "paired": acc},
                "steps_to_90": {"pass": steps_pass, "faster_or_equal_seeds": faster_seeds,
                                "n": len(shared)},
                "seconds_to_90": {"pass": time_pass},
                "training_recall_loss": {"pass": loss_pass, "paired": loss},
            }
        verdicts[candidate] = {
            "per_baseline": rows,
            "memory": {
                "pass": c["state_floats"] <= STATE_LIMIT[candidate],
                "state_floats": c["state_floats"], "limit": STATE_LIMIT[candidate],
            },
            "all_pass": all(v["pass"] for r in rows.values() for v in r.values())
            and c["state_floats"] <= STATE_LIMIT[candidate],
        }

    summary = {"arms": arms, "verdicts": verdicts}
    if args.write:
        args.write.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for arm, row in arms.items():
        print(f"{arm:20s} n={len(row['seeds'])} recall={row['mean_final_recall']:.4f} "
              f"steps90={row['steps_to_90']} mean={row['mean_steps_to_90']:.0f} "
              f"t90~{row['est_seconds_to_90']:.1f}s auc={row['mean_training_recall_ce']:.3f} "
              f"params={row['parameters']} state={row['state_floats']} rss={row['peak_rss_mb']:.0f}MB")
    for candidate, verdict in verdicts.items():
        print(f"\n{candidate}: ALL PASS = {verdict['all_pass']}  memory={verdict['memory']}")
        for baseline, row in verdict["per_baseline"].items():
            print(f"  vs {baseline:20s} " + " ".join(
                f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in row.items())
                + f"  (acc d={row['accuracy']['paired']['mean']:+.4f},"
                  f" faster {row['steps_to_90']['faster_or_equal_seeds']}/{row['steps_to_90']['n']})")


if __name__ == "__main__":
    main()
