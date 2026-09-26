# PHL-DAM v2 — what made the architecture win, and what it owes to generic mechanisms

## Verdict

**On the 176-token associative-recall benchmark, PHL-DAM v2 (`v2_TC`) passes
every preregistered criterion against every baseline, on fresh seeds.** At
32,147 parameters and **68 floats of inference state** — less than any
baseline, including the SSMs' 96 — it reaches 99.99% recall, hits 90% recall in
a mean **55 steps / 14.0 s** (factorized DNC 115 steps / 59.6 s; original
PHL-DAM 272 steps and one seed that never gets there; Transformer and SSMs
never, even at 10× budget), and carries a training recall-loss area of
**0.166** (factorized DNC 0.511, original PHL-DAM 1.136, Transformer 1.839).

Three qualifications belong in the same paragraph:

1. **The speed comes mostly from a generic mechanism.** The copy readout that
   delivers most of the gain transfers to the DNC: a factorized DNC given the
   same readout reaches 90% in 30 steps (v2_TC: 55) with a lower recall-loss
   area (0.067 vs 0.166). v2_TC is still slightly faster in wall-clock (14.0 s
   vs 14.8 s, because each step is ~2× cheaper) and uses 5.8× less state (68 vs
   392 floats). This control was run and is reported with equal weight.
2. **The PHL horizon lattice is removed in v2.** Dropping it made learning
   faster; v2 is PHL-DAM's slot memory, not its temporal backbone.
3. **Under write pressure v2 does not beat the factorized DNC.** On the 004
   ladder it breaks through ~4× sooner (step 50 vs 200–250) but ends at lower
   recall (best dev variant 58–72% vs 77–80% at W=20), and it suffers gradient
   spikes of 10²–10⁸ that the DNC (max ~10) does not. This is dev-seed evidence
   only; no v2 pressure ladder was confirmed.

## How the benchmark gap was located

| Step | Finding |
|---|---|
| NTM/DNC baseline | Plain DNC 49.5%: PHL-DAM wins by +50 pp, 6/6 seeds |
| Role-factorized DNC control | Given PHL-DAM's hardwired key/value wiring, the DNC ties PHL-DAM (99.99%). The original advantage was the wiring |
| Instrumented plateau | PHL-DAM's write gate collapses from 4.8% to 1.3% open in 25 steps; sigmoid saturation leaves it 30–50× less gradient than the DNC's gate |
| `normalized_values` | Read values ÷ occupancy (a slot's write-weighted average, independent of gate magnitude). Gate becomes selective ~100 steps sooner |
| Remove PHL | Faster still; state 456 → 392 floats |
| Component diagnostic | The remaining plateau waits on the output layer learning to *decode* retrieved values (18% → 57% over 60 steps) |
| `copy_readout` | Score the retrieved value against every token's value projection; a clean retrieval decodes correctly from initialisation. Breakout moves to step ~30–50 |
| Compact memory | 4 slots × (8+8) dims suffice for 3 bindings: 68 state floats |

Ideas tested and rejected on dev seeds: opening the write gate (becomes
anti-selective), direct readout, softer read temperature, scale-free merge
threshold, key biases, occupancy decay (FIFO), and a DNC-style free gate
(collapses under pressure).

## Round 1 (seeds 0–5) — the honest miss

Preregistered in `work/PREREGISTRATION_v2_confirmation.md`. `v2_S` (normalized
values, no PHL, d_model 76) tied the factorized DNC: mean steps-to-90% 117 vs
117, faster-or-equal on 3/6 seeds, recall-loss area 0.509 vs 0.511. Failed.
`v2_T` (compact) collapsed on seed 5. Failed. The round also exposed a flaw in
the preregistered seconds criterion (a never-converging arm was charged 510
steps and so looked fast); it was reported as written and fixed for round 2.

## Round 2 (fresh seeds 6–11) — confirmation

Preregistered in `work/PREREGISTRATION_v2_round2.md` before any seed-6–11 run;
candidates chosen on dev seeds 100–105 only.

| Arm | Recall | Steps to 90% | Mean | s to 90% | Recall-loss area | State |
|---|---:|---|---:|---:|---:|---:|
| **v2_TC** | **99.99%** | 50 50 50 60 70 50 | **55** | **14.0** | **0.166** | **68** |
| v2_SC | 99.67% | 60 50 50 40 50 50 | 50 | 14.8 | 0.129 | 392 |
| factorized DNC | 100.00% | 150 120 100 120 80 120 | 115 | 59.6 | 0.511 | 392 |
| PHL-DAM (original) | 84.73% | 220 310 200 never 240 150 | 272 | ∞ | 1.136 | 456 |
| DNC (plain) | 50.39% | never | — | ∞ | 1.789 | 392 |
| Transformer | 47.08% | never | — | ∞ | 1.839 | 16,896 |
| SSM selective | 27.70% | never | — | ∞ | 2.161 | 96 |
| SSM diagonal | 26.56% | never | — | ∞ | 2.196 | 96 |
| *control: factorized DNC + copy readout* | 100.00% | 30 ×6 | 30 | 14.8 | 0.067 | 392 |

Seconds = mean steps × median s/step from a round-robin benchmark (v2_TC 0.254,
v2_SC 0.296, factorized DNC 0.518, copy DNC 0.492, original PHL-DAM 0.594).

**Criteria (per `phl_dam_v2_confirmation.py --round 2`):**

- `v2_TC`: **all pass** — accuracy, steps (faster on 6/6 seeds against every
  baseline), seconds, recall-loss area (robust), memory (68 < 96).
- `v2_SC`: all pass except accuracy against the original PHL-DAM: +14.9 pp,
  but carried by the one seed where PHL-DAM collapsed, so `paired_verdict`
  flags it not robust.

## Write-pressure ladder

The factorized DNC has no write-pressure cliff at all:

| Writes | PHL-DAM (004G) learned | factorized DNC learned | factorized DNC recall |
|---:|---:|---:|---:|
| 8 | 4/5 | 5/5 | 91.4% |
| 16 | 3/5 | 5/5 | 79.3% |
| 20 | 0/5 | 5/5 | 73.9% |
| 24 | 0/5 | 5/5 | 71.3% |
| 32 | 1/5 | 4/4 (1 pending) | 67.2% |

The "cliff between 16 and 20 writes" reported for PHL-DAM is therefore a
property of PHL-DAM's write/allocation mechanics, not of the task.

v2 on the pressure task (dev seeds 100–101, W=20; `outputs/v2_dev_screen/pressure_dev/`):

| Variant | Recall s100 / s101 | Breakthrough | Max grad norm |
|---|---|---:|---:|
| factorized DNC | 77.2% / 79.8% | 200 / 250 | ~11 |
| v2 (normalized, copy, no PHL) | 59.5% / 59.2% | 50 | 10² |
| + retention head | 24.9% / 67.6% | 50 | 10⁵–10⁷ |
| + retention, bounded read (floor 0.05, prior ε 0.05) | 72.2% / 58.3% | 50 | 10⁴–10⁶ |
| + free gate | 10.5% / 25.6% | 50–75 | 10⁶–10⁸ |

A minimal port to the original pressure model (normalized values only,
`--normalized-values --backbone none`, seeds 0–4) learned 4/5 at W=16 (PHL-DAM
3/5), mean recall 55.7% vs 50.0%.

**Open problem:** some path in v2's slot recurrence is unbounded under
eviction; the two 1/x singularities fixed so far (value normalisation and the
log-occupancy read prior) reduce but do not remove the spikes.

## Reproduction

```
cd work
python3 phl_dam_learning_curve.py --arm phl_dam_v2 --seed 6 --probe-every 10 \
  --options '{"fast": true, "normalized_values": true, "use_phl": false,
              "d_model": 84, "num_slots": 4, "d_key": 8, "d_value": 8,
              "copy_readout": true}'
python3 phl_dam_v2_confirmation.py --round 2
python3 phl_dam_timing_benchmark.py
```

Raw per-seed curves: `phl_dam_v2r2_*.json` (round 2), `phl_dam_v2conf_*.json`
(round 1), dev screens in `v2_dev_screen/`. Environment: Python 3.11, torch
2.14 CPU; a cross-environment reproduction of the published PHL-DAM seed 0
matched to 1/6,000 queries.
