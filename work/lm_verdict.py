"""Apply PREREGISTRATION_LM_round1.md to the LM runs, and time the arms.

    python3 lm_verdict.py --timing      # round-robin step-time benchmark
    python3 lm_verdict.py               # criteria 1-4, pass or fail
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import lm_hybrid as lm

ARMS = ("window", "window_sml", "window_sml_local", "full")
SEEDS = (0, 1, 2)
OUT = Path(__file__).resolve().parent.parent / "outputs"


def timing(rounds: int = 12) -> dict:
    torch.set_num_threads(1)
    x = torch.randint(0, 256, (32, 129), generator=torch.Generator().manual_seed(0))
    models, opts = {}, {}
    for arm in ARMS:
        kw = {"memory_layer": 3, "compile_step": True} if arm.startswith("window_sml") else {}
        torch.manual_seed(0)
        models[arm] = lm.LM(arm, 128, 16, **kw)
        opts[arm] = torch.optim.AdamW(models[arm].parameters())
    times = {arm: [] for arm in ARMS}
    for r in range(rounds + 2):
        for arm in ARMS:
            t0 = time.perf_counter()
            loss = F.cross_entropy(models[arm](x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
            opts[arm].zero_grad(); loss.backward(); opts[arm].step()
            if r >= 2:
                times[arm].append(time.perf_counter() - t0)
    return {arm: statistics.median(v) for arm, v in times.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timing", action="store_true")
    a = p.parse_args()
    if a.timing:
        result = timing()
        (OUT / "lm_round1_timing.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=1))
        return

    runs = {arm: {s: json.loads((OUT / f"lm_round1_{arm}_seed{s}.json").read_text())
                  for s in SEEDS if (OUT / f"lm_round1_{arm}_seed{s}.json").exists()}
            for arm in ARMS}
    def metric(arm, s, key):
        return runs[arm][s]["final"][key]
    complete = all(len(runs[arm]) == len(SEEDS) for arm in ARMS)
    seeds = [s for s in SEEDS if all(s in runs[arm] for arm in ARMS)]
    rare, bpb = "rare_word_repeat_bits_per_byte", "bits_per_byte"
    table = {arm: {
        "bits_per_byte": [round(metric(arm, s, bpb), 4) for s in seeds],
        "rare_word_bits_per_byte": [round(metric(arm, s, rare), 4) for s in seeds],
        "passkey_digit_accuracy": [round(metric(arm, s, "passkey_digit_accuracy"), 3) for s in seeds],
        "passkey_key_accuracy": [metric(arm, s, "passkey_key_accuracy") for s in seeds],
    } for arm in ARMS}
    c1 = (all(metric("window_sml", s, rare) < metric("window", s, rare) for s in seeds)
          and all(metric("window_sml", s, rare) < metric("window_sml_local", s, rare) for s in seeds))
    c2 = all(metric("window_sml", s, bpb) <= metric("window", s, bpb) for s in seeds)
    c3 = (statistics.fmean(metric("window_sml", s, rare) for s in seeds)
          <= statistics.fmean(metric("full", s, rare) for s in seeds))
    timing_path = OUT / "lm_round1_timing.json"
    c4 = None
    if timing_path.exists():
        t = json.loads(timing_path.read_text())
        sample = runs["window_sml"][seeds[0]]
        state_sml = sample["kv_cache_floats"] + sample["memory_state_floats"]
        state_full = runs["full"][seeds[0]]["kv_cache_floats"]
        c4 = {"step_time_ratio": t["window_sml"] / t["window"],
              "state_floats": state_sml, "full_kv_cache_floats": state_full,
              "pass": t["window_sml"] / t["window"] <= 1.6 and state_sml < state_full}
    verdict = {"complete": complete, "seeds": seeds, "table": table,
               "criteria": {"1_long_range_benefit": c1, "2_no_harm_to_text": c2,
                            "3_matches_full_attention_on_long_range": c3,
                            "4_affordable": c4},
               "worth_using": bool(complete and c1 and c2 and c3 and c4 and c4["pass"])}
    (OUT / "lm_round1_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    for arm, row in table.items():
        print(f"{arm:18s} " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(json.dumps(verdict["criteria"], indent=1))
    print("WORTH USING:", verdict["worth_using"], "(complete)" if complete else "(incomplete)")


if __name__ == "__main__":
    main()
