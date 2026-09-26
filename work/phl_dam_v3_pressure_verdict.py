"""Apply PREREGISTRATION_v3_round3.md part B to the write-pressure ladder.

For v3 against each opponent (PHL-DAM 004G, factorized DNC 004I, factorized
DNC + copy readout 004K), at every write level on seeds 0-4:

1. learned count >= the opponent's (learned = final training recall CE < 2.0);
2. mean recall higher and no robust paired loss (paired_verdict, 5 pp);
3. mean breakthrough step earlier (first logged recall CE < 2.0; a run that
   never breaks through is charged 710).

The ladder is won against an opponent only if all three hold at every level.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from phl_dam_report_stats import learned, paired_verdict

LEVELS = (8, 16, 20, 24, 32)
SEEDS = range(5)
NEVER = 710
CANDIDATE = ("v3", "phl_dam_004k_v3_w{w}_seed{s}.json")
OPPONENTS = {
    "phl_dam": "phl_dam_004g_stable_w{w}_seed{s}.json",
    "ntm_dnc_factorized": "phl_dam_004i_ntm_dnc_factorized_w{w}_seed{s}.json",
    "ntm_dnc_factorized_copy": "phl_dam_004k_ntm_dnc_factorized_copy_w{w}_seed{s}.json",
}


def load(outputs: Path, pattern: str, writes: int) -> dict[int, dict]:
    runs = {}
    for seed in SEEDS:
        path = outputs / pattern.format(w=writes, s=seed)
        if path.exists():
            runs[seed] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def recall(run: dict) -> float:
    return run.get("metrics", {}).get("recall", 0.0)


def breakthrough(run: dict) -> int:
    step = run.get("breakthrough_step")
    return step if step is not None else NEVER


def summary(runs: dict[int, dict]) -> dict:
    return {
        "seeds": sorted(runs),
        "learned": sum(learned(r) for r in runs.values()),
        "recall": [round(recall(runs[s]), 4) for s in sorted(runs)],
        "mean_recall": statistics.fmean(recall(r) for r in runs.values()),
        "mean_breakthrough": statistics.fmean(breakthrough(r) for r in runs.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("../outputs"))
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    result: dict = {"levels": {}, "verdicts": {}}
    won = {name: True for name in OPPONENTS}
    complete = True
    for writes in LEVELS:
        mine = load(args.outputs, CANDIDATE[1], writes)
        level = {"v3": summary(mine) if mine else None}
        for name, pattern in OPPONENTS.items():
            theirs = load(args.outputs, pattern, writes)
            shared = sorted(set(mine) & set(theirs))
            if len(shared) < len(SEEDS):
                complete = False
            if not shared:
                won[name] = False
                continue
            a = {s: mine[s] for s in shared}
            b = {s: theirs[s] for s in shared}
            sa, sb = summary(a), summary(b)
            contrast = paired_verdict([recall(a[s]) - recall(b[s]) for s in shared])
            robust_loss = contrast["robust"] and contrast["mean"] < 0
            checks = {
                "learned": sa["learned"] >= sb["learned"],
                "recall": sa["mean_recall"] > sb["mean_recall"] and not robust_loss,
                "breakthrough": sa["mean_breakthrough"] < sb["mean_breakthrough"],
            }
            if not all(checks.values()):
                won[name] = False
            level[name] = {**sb, "checks": checks, "paired_recall": contrast,
                           "n_shared": len(shared)}
        result["levels"][str(writes)] = level
    result["verdicts"] = {name: {"ladder_won": won[name], "complete": complete}
                          for name in OPPONENTS}
    if args.write:
        args.write.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for writes, level in result["levels"].items():
        v = level["v3"]
        if v:
            print(f"W={writes:>2s} v3: learned {v['learned']}/{len(v['seeds'])} "
                  f"recall {v['mean_recall']:.3f} breakthrough {v['mean_breakthrough']:.0f}")
        for name in OPPONENTS:
            row = level.get(name)
            if row:
                print(f"       vs {name:24s} learned {row['learned']}/{row['n_shared']} "
                      f"recall {row['mean_recall']:.3f} bt {row['mean_breakthrough']:.0f}  "
                      + " ".join(f"{k}={'PASS' if ok else 'FAIL'}" for k, ok in row["checks"].items())
                      + f"  (d={row['paired_recall']['mean']:+.3f}, "
                        f"robust={row['paired_recall']['robust']})")
    print(json.dumps(result["verdicts"]))


if __name__ == "__main__":
    main()
