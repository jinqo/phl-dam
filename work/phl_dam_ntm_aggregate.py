"""Aggregate the NTM/DNC baseline against PHL-DAM and the earlier baselines.

PHL-DAM's per-seed numbers are read from the frozen no-write-budget bundle
rather than rerun, exactly as the Transformer rematch reused them: rerunning the
favoured arm invites selecting a better draw.

Every contrast is paired by seed and reported through ``paired_verdict`` so its
fragility is stated next to its mean.
"""

from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from pathlib import Path

from phl_dam_report_stats import learned, paired_verdict

SEEDS = range(6)
BUNDLE = "PHL_DAM_No_Write_Budget_vs_Transformer_Reproduction.zip"
SUCCESS = 0.20   # the ">= 20% recall" success bar the rematch reports used


def phl_dam_recall(outputs: Path) -> dict[int, dict]:
    with zipfile.ZipFile(outputs / BUNDLE) as bundle:
        return {
            seed: json.loads(
                bundle.read(f"results/phl_dam_no_write_budget_seed{seed}.json")
            )["metrics"]
            for seed in SEEDS
        }


def per_seed(outputs: Path, pattern: str) -> dict[int, dict]:
    found = {}
    for seed in SEEDS:
        path = outputs / pattern.format(seed=seed)
        if path.exists():
            found[seed] = json.loads(path.read_text(encoding="utf-8"))
    return found


def describe(name: str, runs: dict[int, dict]) -> dict:
    metrics = {s: r.get("metrics", r) for s, r in runs.items()}
    recall = [metrics[s]["recall_accuracy"] for s in sorted(metrics)]
    ablated_key = next(
        (k for k in ("retrieval_disabled_accuracy", "memory_disabled_accuracy",
                     "attention_disabled_accuracy", "recurrence_disabled_accuracy")
         if k in next(iter(metrics.values()))),
        None,
    )
    configuration = next(iter(runs.values())).get("configuration", {})
    return {
        "model": name,
        "seeds": sorted(metrics),
        "parameters": configuration.get("active_parameters"),
        "recurrent_state_floats": configuration.get("recurrent_state_floats"),
        "seed_recall": recall,
        "mean_recall": statistics.fmean(recall),
        "sd_recall": statistics.stdev(recall) if len(recall) > 1 else None,
        "successful_seeds": sum(r >= SUCCESS for r in recall),
        "mean_recall_token_ce": statistics.fmean(
            metrics[s]["recall_token_ce"] for s in metrics
        ),
        "mean_memory_disabled_recall": (
            statistics.fmean(metrics[s][ablated_key] for s in metrics)
            if ablated_key else None
        ),
        "mean_distance_accuracy": {
            band: statistics.fmean(metrics[s]["distance_accuracy"][band] for s in metrics)
            for band in ("29-63", "64-95", "96-169")
        },
        "all_finite": all(metrics[s].get("finite", True) for s in metrics),
    }


PRESSURE_LEVELS = (8, 16, 20, 24, 32)
PRESSURE_ARMS = {
    # PHL-DAM with the spread-floor stability fix: the reference 004G ladder.
    "phl_dam": "phl_dam_004g_stable_w{writes}_seed{seed}.json",
    "ntm_dnc_factorized": "phl_dam_004i_ntm_dnc_factorized_w{writes}_seed{seed}.json",
    "ntm_dnc": "phl_dam_004i_ntm_dnc_w{writes}_seed{seed}.json",
}


def pressure_ladder(outputs: Path) -> dict:
    """Learned counts and paired recall contrasts at every write level."""
    ladder: dict = {"levels": {}, "contrasts": {}}
    for writes in PRESSURE_LEVELS:
        level = {}
        runs_by_arm = {}
        for arm, pattern in PRESSURE_ARMS.items():
            runs = {}
            for seed in range(5):
                path = outputs / pattern.format(writes=writes, seed=seed)
                if path.exists():
                    runs[seed] = json.loads(path.read_text(encoding="utf-8"))
            if not runs:
                continue
            runs_by_arm[arm] = runs
            recall = {s: r["metrics"].get("recall", 0.0) for s, r in runs.items()}
            level[arm] = {
                "seeds": sorted(runs),
                "learned": sum(learned(r) for r in runs.values()),
                "runs": len(runs),
                "finite": sum(bool(r.get("finite", True)) for r in runs.values()),
                "seed_recall": [recall[s] for s in sorted(recall)],
                "mean_recall": statistics.fmean(recall.values()),
                "final_recall_ce": [runs[s]["final_recall_ce"] for s in sorted(runs)],
            }
        ladder["levels"][str(writes)] = level
        if "phl_dam" in runs_by_arm:
            for arm in ("ntm_dnc_factorized", "ntm_dnc"):
                if arm not in runs_by_arm:
                    continue
                shared = sorted(set(runs_by_arm[arm]) & set(runs_by_arm["phl_dam"]))
                if len(shared) >= 2:
                    ladder["contrasts"][f"w{writes}_{arm}_minus_phl_dam"] = paired_verdict([
                        runs_by_arm[arm][s]["metrics"].get("recall", 0.0)
                        - runs_by_arm["phl_dam"][s]["metrics"].get("recall", 0.0)
                        for s in shared
                    ])
    return ladder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("../outputs"))
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    phl = phl_dam_recall(args.outputs)
    arms = {
        "ntm_dnc": per_seed(args.outputs, "phl_dam_ntm_baseline_seed{seed}.json"),
        "ntm_dnc_factorized": per_seed(args.outputs, "phl_dam_ntm_factorized_seed{seed}.json"),
        "causal_transformer": per_seed(args.outputs, "phl_dam_transformer_seed{seed}.json"),
        "selective_ssm": per_seed(args.outputs, "phl_dam_ssm_selective_seed{seed}.json"),
        "diagonal_ssm": per_seed(args.outputs, "phl_dam_ssm_baseline_seed{seed}.json"),
    }
    summary = {
        "experiment": "PHL-DAM vs NTM/DNC at matched parameters (176 tokens, 3 bindings)",
        "protocol": {
            "seeds": list(SEEDS), "training_steps": 500, "batch_size": 16,
            "eval_episodes": 2000, "learning_rate": 2e-3,
            "phl_dam_source": BUNDLE + " (frozen, not rerun)",
        },
        "models": {"phl_dam": describe("phl_dam", {s: {"metrics": m} for s, m in phl.items()})},
        "contrasts": {},
    }
    summary["models"]["phl_dam"]["parameters"] = 33_034
    summary["models"]["phl_dam"]["recurrent_state_floats"] = 456
    for name, runs in arms.items():
        if not runs:
            continue
        summary["models"][name] = describe(name, runs)
        shared = sorted(set(runs) & set(phl))
        differences = [
            phl[s]["recall_accuracy"] - runs[s]["metrics"]["recall_accuracy"] for s in shared
        ]
        summary["contrasts"][f"phl_dam_minus_{name}"] = paired_verdict(differences)

    if "ntm_dnc" in arms and "ntm_dnc_factorized" in arms:
        shared = sorted(set(arms["ntm_dnc"]) & set(arms["ntm_dnc_factorized"]))
        if shared:
            summary["contrasts"]["ntm_dnc_factorized_minus_ntm_dnc"] = paired_verdict([
                arms["ntm_dnc_factorized"][s]["metrics"]["recall_accuracy"]
                - arms["ntm_dnc"][s]["metrics"]["recall_accuracy"]
                for s in shared
            ])

    summary["write_pressure_ladder"] = pressure_ladder(args.outputs)

    text = json.dumps(summary, indent=2) + "\n"
    if args.write:
        args.write.write_text(text, encoding="utf-8")
    for name, row in summary["models"].items():
        print(f"{name:20s} params={row['parameters']} state={row['recurrent_state_floats']} "
              f"recall={row['mean_recall']:.4f}±{row['sd_recall'] or 0:.4f} "
              f"ok={row['successful_seeds']}/{len(row['seeds'])} "
              f"ablated={row['mean_memory_disabled_recall']}")
    for name, verdict in summary["contrasts"].items():
        print(f"{name:40s} mean={verdict['mean']:+.4f} robust={verdict['robust']} "
              f"wins={verdict['positive']}/{verdict['n']} warnings={verdict['warnings']}")
    print("\nwrite-pressure ladder (learned = final recall CE < 2.0)")
    for writes, level in summary["write_pressure_ladder"]["levels"].items():
        print(f"  W={writes:>2s} " + "  ".join(
            f"{arm}: {row['learned']}/{row['runs']} recall={row['mean_recall']:.3f}"
            for arm, row in level.items()))
    for name, verdict in summary["write_pressure_ladder"]["contrasts"].items():
        print(f"  {name:40s} mean={verdict['mean']:+.4f} robust={verdict['robust']} "
              f"wins={verdict['positive']}/{verdict['n']}")


if __name__ == "__main__":
    main()
