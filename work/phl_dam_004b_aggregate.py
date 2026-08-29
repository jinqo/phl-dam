"""Roll up the per-(arm, seed) PHL-DAM-004B runs into one aggregate file.

Two kinds of comparison come out of these runs and they are not equivalent:

* **Within-run** contrasts (a learned arm against random/FIFO/LRU/oracle
  evaluated under the *same* trained content weights) are exact controlled
  contrasts: only the eviction rule differs.
* **Across-run** contrasts (one learned arm against another) share the content
  architecture, protocol, seeds, streams and training budget but *not* the
  learned content weights. Each arm's content path co-adapts to its own
  eviction behaviour, so those differences are eviction policy *plus*
  co-adaptation. The residency / recall-given-resident split is what separates
  the two, and the shared LRU and oracle reference points measure how far the
  content paths drifted apart.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from phl_dam_004b_lease import ALL_ARMS, EVAL_SETTINGS, LEARNED_ARMS, POLICY_ARMS


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def collect(directory: Path, seeds: list[int], prefix: str) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for arm in LEARNED_ARMS:
        for seed in seeds:
            path = directory / f"{prefix}{arm}_seed{seed}.json"
            if path.exists():
                runs.setdefault(arm, {})[seed] = json.loads(path.read_text())
    return runs


def build(runs: dict[str, dict[int, dict]], seeds: list[int]) -> dict[str, object]:
    sample = next(iter(next(iter(runs.values())).values()))
    settings = list(sample["metrics"])
    table: dict[str, object] = {}

    for setting in settings:
        block: dict[str, object] = {}
        # Every learned run also evaluates the four fixed policies under its own
        # weights, so each fixed policy has one value per (trained arm, seed).
        for policy in POLICY_ARMS:
            per_run = {
                arm: {
                    seed: runs[arm][seed]["metrics"][setting][policy]["recall"]
                    for seed in runs.get(arm, {})
                }
                for arm in LEARNED_ARMS
                if arm in runs
            }
            flat = [value for values in per_run.values() for value in values.values()]
            block[policy] = {
                "mean_recall_over_all_runs": _mean(flat),
                "sd_recall_over_all_runs": _sd(flat),
                "recall_by_trained_arm": per_run,
                "note": (
                    "one value per trained model; spread across trained arms "
                    "measures content-path divergence, not policy quality"
                ),
            }
        for arm in LEARNED_ARMS:
            if arm not in runs:
                continue
            rows = runs[arm]
            block[arm] = {
                "mean_recall": _mean([r["metrics"][setting][arm]["recall"] for r in rows.values()]),
                "sd_recall": _sd([r["metrics"][setting][arm]["recall"] for r in rows.values()]),
                "per_seed_recall": {
                    seed: rows[seed]["metrics"][setting][arm]["recall"] for seed in rows
                },
                "mean_residency": _mean(
                    [r["metrics"][setting][arm]["residency"] for r in rows.values()]
                ),
                "mean_recall_given_resident": _mean(
                    [
                        r["metrics"][setting][arm]["recall_given_resident"]
                        for r in rows.values()
                        if r["metrics"][setting][arm]["recall_given_resident"] is not None
                    ]
                ),
                "mean_all_token_ce": _mean(
                    [r["metrics"][setting][arm]["all_token_ce"] for r in rows.values()]
                ),
                "mean_recall_token_ce": _mean(
                    [r["metrics"][setting][arm]["recall_token_ce"] for r in rows.values()]
                ),
            }

        # Paired within-run contrasts: same content weights, different rule.
        paired: dict[str, object] = {}
        for arm in LEARNED_ARMS:
            if arm not in runs:
                continue
            for reference in POLICY_ARMS:
                differences = [
                    runs[arm][seed]["metrics"][setting][arm]["recall"]
                    - runs[arm][seed]["metrics"][setting][reference]["recall"]
                    for seed in runs[arm]
                ]
                paired[f"{arm}_minus_{reference}_within_run"] = {
                    "per_seed": differences,
                    "mean": _mean(differences),
                    "seed_wins": sum(value > 0 for value in differences),
                    "seeds": len(differences),
                }
        # Across-run contrasts between learned arms: weights are not shared.
        for reference in LEARNED_ARMS:
            if reference == "phl_lease" or "phl_lease" not in runs or reference not in runs:
                continue
            shared = sorted(set(runs["phl_lease"]) & set(runs[reference]))
            differences = [
                runs["phl_lease"][seed]["metrics"][setting]["phl_lease"]["recall"]
                - runs[reference][seed]["metrics"][setting][reference]["recall"]
                for seed in shared
            ]
            paired[f"phl_lease_minus_{reference}_across_runs"] = {
                "per_seed": differences,
                "mean": _mean(differences),
                "seed_wins": sum(value > 0 for value in differences),
                "seeds": len(differences),
                "note": "independently trained; content weights are not shared",
            }
        block["paired"] = paired
        table[setting] = block

    return {
        "experiment": "PHL-DAM-004B aggregate",
        "protocol": {
            "slots": 8,
            "seeds": seeds,
            "arms": list(ALL_ARMS),
            "learned_arms": list(LEARNED_ARMS),
            "fixed_policy_arms": list(POLICY_ARMS),
            "eval_settings": settings,
            # Runs made before --scale existed were all at the full scale.
            "scale": sample["configuration"].get("scale", "full"),
            "within_run_contrasts_share_content_weights": True,
            "across_run_contrasts_share_content_weights": False,
        },
        "settings": table,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("../outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--prefix", default="phl_dam_004b_")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = collect(args.directory, args.seeds, args.prefix)
    if not runs:
        raise SystemExit("no PHL-DAM-004B run files found")
    summary = build(runs, args.seeds)
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
