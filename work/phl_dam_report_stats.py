"""Compute every statistic a PHL-DAM 004 report quotes, straight from the JSON.

Reports in this line had three defects that all shared one cause - numbers typed
into markdown by hand:

* a results table that silently mixed canonical W=12 rows with canonical W=16
  rows under one "Recall" heading;
* a run count ("one in six") that did not match the artifact set;
* an "N of twelve" tally that was off by two.

Every figure this module emits carries the evaluation setting it came from, so a
table cannot be assembled without saying which pressure produced each value.
Use ``--emit markdown`` to get tables that can be pasted verbatim.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


def load(pattern: str, directory: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(directory / pattern))):
        if "aggregate" in Path(path).name:
            continue
        rows.append(json.loads(Path(path).read_text()))
    return rows


def normalised_metrics(run: dict) -> dict[str, dict[str, dict]]:
    """Return metrics as {setting: {arm: row}} whatever shape the run used.

    PHL-DAM-004B/004C store one block per evaluation setting. PHL-DAM-004D
    evaluates a single setting and stores that row flat. Both are reported the
    same way here, and in every case the setting name travels with the numbers -
    which is the invariant that was violated when a table mixed W=12 and W=16
    rows under one heading.
    """
    metrics = run.get("metrics") or {}
    if not metrics:
        return {}
    if "recall" in metrics:
        arm = run.get("model", metrics.get("arm", "?"))
        setting = f"{metrics.get('condition', 'canonical')}_w{metrics.get('writes', '?')}"
        return {setting: {arm: metrics}}
    return metrics



def run_id(run: dict) -> str:
    """Identity that stays unique across conditions, not just across seeds.

    ``model_seedN`` collides the moment one arm is run at several write levels
    or scales, which silently merges distinct runs in any per-run listing.
    """
    config = run.get("configuration", {})
    parts = [str(run.get("model", "?"))]
    if config.get("writes") is not None:
        parts.append(f"w{config['writes']}")
    scale = config.get("scale")
    if scale and scale not in ("full",):
        parts.append(str(scale))
    parts.append(f"seed{config.get('seed', '?')}")
    return "_".join(parts)



def breakthrough_step(run: dict, threshold: float = 2.0) -> int | None:
    for record in run.get("training_history", []):
        if record.get("recall_ce", math.inf) < threshold:
            return record["step"]
    return None


def learned(run: dict, threshold: float = 2.0) -> bool:
    """Did the run END below the breakthrough threshold, not merely touch it?"""
    history = run.get("training_history", [])
    if not history:
        return False
    final = history[-1].get("recall_ce")
    return final is not None and math.isfinite(final) and final < threshold


def summarise(values: list[float]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "sd": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def paired(differences: list[float], resamples: int = 10_000, seed: int = 12345) -> dict:
    """Paired contrast with a bootstrap interval and an explicit sign tally."""
    clean = [d for d in differences if d is not None and math.isfinite(d)]
    if not clean:
        return {"n": 0}
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(statistics.fmean(sample))
    means.sort()
    return {
        "n": len(clean),
        "per_seed": clean,
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "bootstrap_95": [means[int(0.025 * resamples)], means[int(0.975 * resamples)]]
        if len(clean) > 1
        else None,
        "positive": sum(1 for d in clean if d > 0),
        "negative": sum(1 for d in clean if d < 0),
        "ties": sum(1 for d in clean if d == 0),
    }


def paired_verdict(differences: list[float], threshold: float = 0.05) -> dict:
    """A paired contrast that reports its own fragility.

    Added after a real mistake: the backbone comparison showed `ssm` beating
    `phl` by +0.184 mean recall at W=8, which was narrated as meaningful. It was
    not - the median was -0.001, the bootstrap interval spanned zero, only 2 of 5
    seeds favoured `ssm`, and dropping one seed moved the mean from 0.256 to
    0.015. A bare mean hides all four of those facts.

    This returns the mean alongside every check that would have caught it, and a
    `robust` flag that is only true when they agree. Report the flag, not the
    mean alone.
    """
    base = paired(differences)
    if base.get("n", 0) < 2:
        return {**base, "robust": False, "warnings": ["fewer than two paired samples"]}

    values = base["per_seed"]
    leave_one_out = [
        statistics.fmean(values[:i] + values[i + 1 :]) for i in range(len(values))
    ]
    low, high = base["bootstrap_95"]
    warnings = []
    if (base["mean"] > 0) != (base["median"] > 0):
        warnings.append("mean and median disagree in sign: outlier-driven")
    if low <= 0.0 <= high:
        warnings.append("bootstrap 95% interval spans zero")
    if min(base["positive"], base["negative"]) >= max(1, len(values) // 3):
        warnings.append("seed wins are split, not consistent")
    if (max(leave_one_out) > 0) != (min(leave_one_out) > 0):
        warnings.append("leave-one-seed-out flips the sign: single-seed leverage")
    if abs(base["mean"]) < threshold:
        warnings.append(f"mean below the {threshold:.0%} decision threshold")

    return {
        **base,
        "leave_one_out_means": leave_one_out,
        "leave_one_out_spread": max(leave_one_out) - min(leave_one_out),
        "exceeds_threshold": abs(base["mean"]) >= threshold,
        "warnings": warnings,
        "robust": not warnings,
    }


def describe_paired(name: str, differences: list[float], threshold: float = 0.05) -> str:
    """One-line human summary that leads with the verdict, not the mean."""
    v = paired_verdict(differences, threshold)
    if v.get("n", 0) < 2:
        return f"{name}: too few paired samples"
    verdict = "ROBUST" if v["robust"] else "NOT ROBUST"
    line = (
        f"{name}: {verdict} | mean {v['mean']:+.4f} median {v['median']:+.4f} "
        f"| +/-/= {v['positive']}/{v['negative']}/{v['ties']} "
        f"| boot95 [{v['bootstrap_95'][0]:+.3f}, {v['bootstrap_95'][1]:+.3f}]"
    )
    for w in v["warnings"]:
        line += chr(10) + "    warning: " + w
    return line


def within_run_contrast(runs: list[dict], arm: str, reference: str, setting: str) -> dict:
    """Arm minus a fixed policy evaluated under the SAME trained weights."""
    differences = []
    for run in runs:
        metrics = normalised_metrics(run).get(setting)
        if not metrics or arm not in metrics or reference not in metrics:
            continue
        differences.append(metrics[arm]["recall"] - metrics[reference]["recall"])
    return paired(differences)


def settings_of(runs: list[dict]) -> list[str]:
    seen: list[str] = []
    for run in runs:
        for key in normalised_metrics(run):
            if key not in seen:
                seen.append(key)
    return seen


def per_setting_table(runs: list[dict], arm_key: str = "model") -> dict:
    """recall / residency / CE for every arm at EVERY setting, never merged."""
    table: dict[str, dict[str, dict]] = {}
    for setting in settings_of(runs):
        table[setting] = {}
        by_arm: dict[str, list[dict]] = {}
        for run in runs:
            by_arm.setdefault(run[arm_key], []).append(run)
        for arm, arm_runs in by_arm.items():
            rows = [
                normalised_metrics(r)[setting][arm]
                for r in arm_runs
                if arm in normalised_metrics(r).get(setting, {})
            ]
            if not rows:
                continue
            table[setting][arm] = {
                "recall": summarise([r["recall"] for r in rows]),
                "residency": summarise([r["residency"] for r in rows]),
                "recall_given_resident": summarise(
                    [r["recall_given_resident"] for r in rows]
                ),
                "all_token_ce": summarise([r["all_token_ce"] for r in rows]),
                "recall_token_ce": summarise([r["recall_token_ce"] for r in rows]),
                "per_seed_recall": {
                    r["configuration"]["seed"]: normalised_metrics(r)[setting][arm]["recall"]
                    for r in arm_runs
                    if arm in normalised_metrics(r).get(setting, {})
                },
            }
    return table


def health(runs: list[dict]) -> dict:
    """Run accounting: finite, learned, breakthrough. Never silently filtered."""
    return {
        "total_runs": len(runs),
        "finite_runs": sum(1 for r in runs if r.get("finite", True)),
        "non_finite_runs": sum(1 for r in runs if not r.get("finite", True)),
        "non_finite_ids": [
            run_id(r)
            for r in runs
            if not r.get("finite", True)
        ],
        "learned_runs": sum(1 for r in runs if learned(r)),
        "breakthrough_steps": {
            run_id(r): breakthrough_step(r)
            for r in runs
        },
        "final_recall_ce": {
            run_id(r): (
                r["training_history"][-1]["recall_ce"] if r.get("training_history") else None
            )
            for r in runs
        },
    }


def controller_stats(runs: list[dict], setting: str) -> dict:
    out = {}
    for run in runs:
        arm = run.get("model", "?")
        metrics = normalised_metrics(run).get(setting)
        if not metrics or arm not in metrics:
            continue
        controller = metrics[arm].get("controller", {})
        eviction = metrics[arm].get("eviction", {})
        out[run_id(run)] = {
            "write_gate_at_binding": controller.get("mean_write_gate_at_binding"),
            "write_gate_elsewhere": controller.get("mean_write_gate_elsewhere"),
            "write_selectivity": (
                controller.get("mean_write_gate_at_binding", 0)
                - controller.get("mean_write_gate_elsewhere", 0)
                if controller.get("mean_write_gate_at_binding") is not None
                else None
            ),
            "mean_occupied_slots": controller.get("mean_occupied_slots"),
            "eviction_decisions": eviction.get("decisions"),
        }
    return out


def count_within(values: list[float], target: float, tolerance: float) -> dict:
    """The tally that was hand-written as 'ten of twelve' and was actually eight."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    inside = [v for v in clean if abs(v - target) <= tolerance]
    return {
        "target": target,
        "tolerance": tolerance,
        "within": len(inside),
        "total": len(clean),
        "phrase": f"{len(inside)} of {len(clean)}",
        "values": clean,
    }


def markdown_table(table: dict, setting: str, arms: list[str] | None = None) -> str:
    block = table.get(setting, {})
    names = arms or sorted(block)
    lines = [
        f"Evaluation setting: **{setting}** (every column below is this setting)",
        "",
        "| Arm | Recall mean | Recall median | SD | Residency | Per-seed recall |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in names:
        if arm not in block:
            continue
        row = block[arm]
        per = " / ".join(
            f"{v:.4f}" for _, v in sorted(row["per_seed_recall"].items())
        )
        lines.append(
            f"| `{arm}` | {row['recall']['mean']:.4f} | {row['recall']['median']:.4f} | "
            f"{row['recall']['sd']:.4f} | {row['residency']['mean']:.4f} | {per} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True, help="e.g. phl_dam_004bs_*.json")
    parser.add_argument("--directory", type=Path, default=Path("../outputs"))
    parser.add_argument("--setting", default=None, help="restrict tables to one setting")
    parser.add_argument("--emit", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load(args.pattern, args.directory)
    if not runs:
        raise SystemExit(f"no runs matched {args.pattern}")
    table = per_setting_table(runs)
    report = {
        "pattern": args.pattern,
        "runs": len(runs),
        "settings": settings_of(runs),
        "health": health(runs),
        "by_setting": table,
        "controller": {s: controller_stats(runs, s) for s in settings_of(runs)},
    }
    if args.emit == "markdown":
        setting = args.setting or settings_of(runs)[0]
        print(markdown_table(table, setting))
    else:
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
