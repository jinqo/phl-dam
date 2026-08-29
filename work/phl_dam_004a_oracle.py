"""PHL-DAM-004A: oracle future-relevance upper bound under memory pressure.

Policy-level simulation against an exact content store. If even an oracle that
knows the true future cannot materially beat LRU here, the benchmark cannot
fairly test the PHL temporal-lease hypothesis and the learned stage must not
be run.

Preregistered gate: oracle future relevance must beat LRU by at least
+5 percentage points mean recall at one meaningful nontrivial pressure
condition.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

import phl_dam_pressure_task as task
from phl_dam_eviction_policies import POLICY_NAMES, build_policies, simulate


ORACLE_GATE_POINTS = 0.05
TIME_SINCE_ACCESS_BINS = (("<=64", 0, 64), ("65-128", 65, 128), ("129+", 129, 10**9))


def _blank_counter() -> dict[str, list[int]]:
    return {}


def _bump(table: dict[str, list[int]], key: str, hit: bool) -> None:
    entry = table.setdefault(key, [0, 0])
    entry[0] += int(hit)
    entry[1] += 1


def _rates(table: dict[str, list[int]]) -> dict[str, object]:
    return {
        name: {
            "recall": correct / total if total else None,
            "queries": total,
        }
        for name, (correct, total) in sorted(table.items())
    }


def _time_since_bin(value: int) -> str:
    if value < 0:
        return "miss"
    for name, low, high in TIME_SINCE_ACCESS_BINS:
        if low <= value <= high:
            return name
    return "out-of-range"


def evaluate_condition(
    seed: int, writes: int, condition: str, episodes: int
) -> dict[str, object]:
    generated = task.generate_episodes(seed, episodes, writes, condition)

    peaks = sorted(episode.max_concurrent_live for episode in generated)
    class_counts = [0] * task.NUM_CLASSES
    for episode in generated:
        for item in episode.items:
            class_counts[task.NEVER if item.is_never else item.use_class] += 1
    total_items = sum(class_counts)
    class_prior = torch.tensor([count / total_items for count in class_counts])

    delays = [query.delay for episode in generated for query in episode.queries]
    policies = build_policies()
    results: dict[str, object] = {}

    for policy_index, name in enumerate(POLICY_NAMES):
        policy = policies[name]
        by_delay: dict[str, list[int]] = _blank_counter()
        by_intervening: dict[str, list[int]] = _blank_counter()
        by_occurrence: dict[str, list[int]] = _blank_counter()
        by_time_since: dict[str, list[int]] = _blank_counter()
        by_pressure_bucket: dict[str, list[int]] = _blank_counter()
        hits = 0
        queries = 0
        anchor_hits = 0
        anchor_queries = 0
        evictions = 0
        future_needed_evicted = 0
        dead_evicted = 0
        dead_retained = 0
        live_retained = 0
        wrong_protection = 0
        correct_protection = 0
        contrast_decisions = 0
        contrast_correct = 0
        occupancy_sum = 0.0
        occupancy_samples = 0
        lifetime_sum = 0
        lifetime_count = 0
        overwrites_sum = 0

        for index, episode in enumerate(generated):
            # Stable across processes: PYTHONHASHSEED randomises str hashing.
            rng = random.Random(seed * 7_919 + index * 131 + policy_index)
            trace = simulate(episode, policy, task.NUM_SLOTS, rng)
            pressure_bucket = (
                ">8 live" if episode.max_concurrent_live > task.NUM_SLOTS else "<=8 live"
            )
            for outcome in trace.outcomes:
                hits += int(outcome.hit)
                queries += 1
                overwrites_sum += outcome.evictions_since_write
                _bump(by_delay, task.delay_bin(outcome.delay), outcome.hit)
                _bump(
                    by_intervening,
                    task.intervening_bin(outcome.intervening_writes),
                    outcome.hit,
                )
                _bump(
                    by_occurrence,
                    "first-use" if outcome.occurrence == 0 else "repeat-use",
                    outcome.hit,
                )
                _bump(by_time_since, _time_since_bin(outcome.time_since_access), outcome.hit)
                _bump(by_pressure_bucket, pressure_bucket, outcome.hit)
                if outcome.is_contrast_anchor:
                    anchor_hits += int(outcome.hit)
                    anchor_queries += 1
            evictions += trace.evictions
            future_needed_evicted += trace.future_needed_evicted
            dead_evicted += trace.dead_evicted
            dead_retained += trace.dead_retained_at_end
            live_retained += trace.live_retained_at_end
            wrong_protection += trace.wrong_protection
            correct_protection += trace.correct_protection
            contrast_decisions += trace.contrast_decisions
            contrast_correct += trace.contrast_correct
            occupancy_sum += trace.occupancy_sum
            occupancy_samples += trace.occupancy_samples
            lifetime_sum += trace.lifetime_sum
            lifetime_count += trace.lifetime_count

        protections = wrong_protection + correct_protection
        results[name] = {
            "recall": hits / queries,
            "queries": queries,
            "recall_by_delay": _rates(by_delay),
            "recall_by_intervening_writes": _rates(by_intervening),
            "recall_by_query_occurrence": _rates(by_occurrence),
            "recall_by_time_since_last_access": _rates(by_time_since),
            "recall_by_live_pressure_bucket": _rates(by_pressure_bucket),
            "recall_contrast_anchor": (
                anchor_hits / anchor_queries if anchor_queries else None
            ),
            "contrast_anchor_queries": anchor_queries,
            "evictions": evictions,
            "evictions_per_episode": evictions / len(generated),
            "fraction_of_evictions_that_were_future_needed": (
                future_needed_evicted / evictions if evictions else None
            ),
            "fraction_of_evictions_that_were_dead": (
                dead_evicted / evictions if evictions else None
            ),
            "dead_retained_at_end_per_episode": dead_retained / len(generated),
            "live_retained_at_end_per_episode": live_retained / len(generated),
            "wrong_protection_rate": (
                wrong_protection / protections if protections else None
            ),
            "wrong_protection_decisions": wrong_protection,
            "protection_decisions": protections,
            "contrast_decisions": contrast_decisions,
            "contrast_correct_rate": (
                contrast_correct / contrast_decisions if contrast_decisions else None
            ),
            "mean_occupied_slots_at_write": (
                occupancy_sum / occupancy_samples if occupancy_samples else None
            ),
            "mean_memory_lifetime_tokens": (
                lifetime_sum / lifetime_count if lifetime_count else None
            ),
            "mean_overwrites_before_query": overwrites_sum / queries,
        }

    return {
        "seed": seed,
        "writes": writes,
        "query_condition": condition,
        "episodes": len(generated),
        "pressure": {
            "max_concurrent_live_mean": sum(peaks) / len(peaks),
            "max_concurrent_live_median": peaks[len(peaks) // 2],
            "max_concurrent_live_p10": peaks[int(0.10 * len(peaks))],
            "max_concurrent_live_p90": peaks[int(0.90 * len(peaks))],
            "max_concurrent_live_min": peaks[0],
            "max_concurrent_live_max": peaks[-1],
            "max_concurrent_live_histogram": {
                str(value): peaks.count(value) for value in sorted(set(peaks))
            },
            "fraction_episodes_above_8": sum(p > 8 for p in peaks) / len(peaks),
            "fraction_episodes_above_10": sum(p > 10 for p in peaks) / len(peaks),
            "fraction_episodes_above_12": sum(p > 12 for p in peaks) / len(peaks),
            "mean_queries_per_episode": sum(
                len(episode.queries) for episode in generated
            )
            / len(generated),
            "mean_live_items_per_episode": sum(
                sum(1 for item in episode.items if not item.is_never)
                for episode in generated
            )
            / len(generated),
            "contrast_episode_fraction": sum(
                episode.is_contrast for episode in generated
            )
            / len(generated),
        },
        "signal": {
            "class_prior": {
                task.CLASS_NAMES[index]: float(class_prior[index])
                for index in range(task.NUM_CLASSES)
            },
            "bayes_optimal_tag_class_accuracy": task.bayes_tag_accuracy(class_prior),
            "bayes_optimal_live_vs_never_auroc": task.bayes_live_auroc(class_prior),
            "delay_min": min(delays),
            "delay_max": max(delays),
            "delay_mean": sum(delays) / len(delays),
        },
        "policies": results,
    }


def summarise(runs: list[dict[str, object]]) -> dict[str, object]:
    per_policy: dict[str, list[float]] = {name: [] for name in POLICY_NAMES}
    for run in runs:
        for name in POLICY_NAMES:
            per_policy[name].append(run["policies"][name]["recall"])

    oracle = per_policy["oracle_future_relevance"]
    lru = per_policy["lru"]
    paired = [a - b for a, b in zip(oracle, lru)]
    belady_agreement = all(
        abs(a - b) < 1e-12
        for a, b in zip(per_policy["oracle_future_relevance"], per_policy["belady"])
    )
    return {
        "mean_recall": {
            name: statistics.fmean(values) for name, values in per_policy.items()
        },
        "sample_sd_recall": {
            name: (statistics.stdev(values) if len(values) > 1 else 0.0)
            for name, values in per_policy.items()
        },
        "per_seed_recall": per_policy,
        "oracle_minus_lru": {
            "per_seed": paired,
            "mean": statistics.fmean(paired),
            "sample_sd": statistics.stdev(paired) if len(paired) > 1 else 0.0,
            "seed_wins": sum(value > 0 for value in paired),
            "seeds": len(paired),
        },
        "oracle_minus_best_non_oracle": statistics.fmean(oracle)
        - max(
            statistics.fmean(per_policy[name])
            for name in ("random", "fifo", "lru")
        ),
        "belady_matches_oracle_future_relevance": belady_agreement,
        "gate_oracle_beats_lru_by_5pp": statistics.fmean(paired) >= ORACLE_GATE_POINTS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default="full", choices=list(task.SCALE_PROFILES))
    parser.add_argument("--writes", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument(
        "--conditions", nargs="+", default=["canonical", "spec"],
        choices=list(task.QUERY_CONDITIONS),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task.set_scale(args.scale)
    if args.writes not in task.PRESSURE_LEVELS:
        raise SystemExit(
            f"--writes {args.writes} is not a pressure level of scale "
            f"{args.scale}: {task.PRESSURE_LEVELS}"
        )
    started = time.perf_counter()
    conditions: dict[str, object] = {}
    for condition in args.conditions:
        runs = []
        for seed in args.seeds:
            run = evaluate_condition(seed, args.writes, condition, args.episodes)
            runs.append(run)
            print(
                f"writes={args.writes} condition={condition} seed={seed} "
                + " ".join(
                    f"{name}={run['policies'][name]['recall']:.4f}"
                    for name in POLICY_NAMES
                ),
                flush=True,
            )
        conditions[condition] = {"runs": runs, "summary": summarise(runs)}

    summary = {
        "experiment": "PHL-DAM-004A - Oracle future-relevance upper bound",
        "configuration": {
            "scale": task.SCALE,
            "slots": task.NUM_SLOTS,
            "writes_per_episode": args.writes,
            "seeds": args.seeds,
            "episodes_per_seed": args.episodes,
            "sequence_length": task.SEQUENCE_LENGTH,
            "delay_range": [task.MIN_DELAY, task.MAX_DELAY],
            "query_conditions": args.conditions,
            "query_budget": {
                condition: task.QUERY_BUDGET[condition][args.writes]
                for condition in args.conditions
            },
            "exact_content_store": True,
            "query_consumes_binding": False,
            "tag_primary_probability": task.TAG_PRIMARY_PROBABILITY,
            "contrast_fraction": task.CONTRAST_FRACTION,
            "access_definition": (
                "slot written, or query whose key is the key held by the slot"
            ),
            "oracle_uses_generator_truth": True,
            "learned_components": False,
            "lease_state_present": False,
            "promotion": False,
        },
        "gate": {
            "rule": "oracle future relevance minus LRU >= 5 pp mean recall",
            "threshold": ORACLE_GATE_POINTS,
            "passed_by_condition": {
                condition: conditions[condition]["summary"]["gate_oracle_beats_lru_by_5pp"]
                for condition in args.conditions
            },
        },
        "conditions": conditions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                condition: conditions[condition]["summary"]["mean_recall"]
                for condition in args.conditions
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
