> **SUPERSEDED by `PHL_DAM_004BS_Supervised_Lease_Report_v2.md`.** This version
> (a) said "ten of twelve" runs were within 0.005 of the Bayes-optimal AUROC when
> the count is eight, (b) generalised a three-seed result into a claim about
> temporal relevance in general, and (c) said two controllers beat LRU
> "consistently" when that holds only at the highest canonical pressure. Read v2.

# PHL-DAM-004B-S — Supervised-timing lease: does PHL transport add anything?

## Verdict

Given a timing predictor that demonstrably works and a transport operator calibrated to the task's own delays, PHL transport does not beat holding the same prediction static. The timing head reaches AUROC 0.85 against a Bayes ceiling of 0.849, so the relevance signal is decoded essentially optimally — the excuse available to PHL-DAM-004B is gone. Against its parameter-identical static twin the transported lease scores −11.9 pp mean recall at the highest pressure and −14.4 pp at moderate pressure. On the single seed where both arms trained cleanly it wins by 1.2 pp, far short of the 5 pp gate; on the seed where they diverge it loses by 36.8 pp after collapsing into memory thrashing. Transport is neutral at best and destructive at worst. The lease/transport approach in its current form should be killed.

## Research question

Two questions, in order:

1. **Can the timing signal be learned at all** when supervision is supplied equally to every learned arm? (PHL-DAM-004B left this unresolved: the head never decoded the cue, so transport was never given a working input.)
2. **Given a working prediction, does transporting it through the PHL horizon lattice beat holding it static?** This is the direct transport attribution, and it is the only comparison in this research line that can isolate PHL.

## What changed from PHL-DAM-004B

Two corrections, both made before any 004B-S result existed:

**Scale-derived lease horizons.** `LEASE_BIN_EDGES` had been hardcoded to the full-scale delay ranges (0–31 / 32–56 / 57–96 / 97–160 / 161–256) and was never rebuilt when the compact profile was active, whose delays are 29–104. The transport operator's per-token rates are `1 / bin width`, so mass walked the lattice roughly 2.5× too slowly — only 8.8% of a "far" lease reached the "due" horizon by the time its query actually arrived. Horizons are now derived from `task.FIRST_USE_DELAY` at the active scale. A test asserts that far-horizon mass peaks at "due" as the far delay elapses, at both scales. **This defect penalised `phl_lease` and nothing else, so the 004B compact numbers for that arm were a lower bound.** 004B-S is the fair test.

**Equal timing supervision.** All four learned arms carry an identical `lease_head` trained with the same cross-entropy against the true next-use horizon, weight 1.0. Confirmed equal: every arm starts at `timing_ce = 1.8826` on a given seed. For `content_only` and `learned_utility` the head is trained but deliberately not wired into their eviction score, so the auxiliary gradient reaches every arm's shared representation without converting the controls into lease arms. A test asserts the recall loss alone leaves those heads untouched. Labels are training-only; a test asserts that overwriting them does not change any forward pass.

## Frozen protocol

Compact scale, 176 tokens, 8 slots, writes 8/12/16, delays 29–104, 800 updates, batch 16, lr 2e-3, AdamW, write-budget weight 0, promotion off, seeds 0–2, 512 evaluation episodes per setting, `timing-weight` 1.0 for every learned arm. No arm receives future information at evaluation; the oracle is evaluation-only.

## Gate 1 — was the timing signal actually learned? **PASS**

Measured at write positions against generator truth, used for measurement only.

| Arm | seed 0 | seed 1 | seed 2 | Bayes ceiling |
|---|---:|---:|---:|---:|
| `content_only` | .629 / .840 | .643 / .847 | .628 / .848 | **.690 / .849** |
| `static_priority` | .528 / .834 | .641 / .846 | .643 / .851 | .690 / .849 |
| `phl_lease` | .426 / .740 | .640 / .846 | .633 / .851 | .690 / .849 |
| `learned_utility` | .633 / .843 | .641 / .846 | .643 / .849 | .690 / .849 |

*(class accuracy / live-vs-never AUROC)*

Ten of twelve runs sit within 0.005 of the Bayes-optimal AUROC. The relevance signal is decoded about as well as the cue permits. The single low run, `phl_lease` seed 0 at 0.740, degraded alongside that run's late collapse — its head was damaged by the training instability, not the reverse.

## Gate 2 — does transport beat static? **FAIL**

`phl_lease` minus `static_priority`, recall, per seed:

| Setting | seed 0 | seed 1 | seed 2 | Mean | Required |
|---|---:|---:|---:|---:|---:|
| canonical W=12 | **−42.06 pp** | +0.07 pp | −1.14 pp | **−14.38 pp** | ≥ +5 pp |
| canonical W=16 | **−36.82 pp** | +0.03 pp | +1.19 pp | **−11.86 pp** | ≥ +5 pp |

The naive tally reads "2/3 seed wins", and that is misleading — seed 1 is a tie at chance where neither arm learned (both 9.6% recall, 0.000 residency), so it carries no information. Reading the seeds by what actually happened:

| Seed | Status | `phl_lease` | `static_priority` | Difference |
|---|---|---:|---:|---:|
| 0 | phl collapsed | 22.55% / .113 | 59.37% / .571 | **−36.82 pp** |
| 1 | neither learned | 9.62% / .000 | 9.59% / .000 | +0.03 pp |
| 2 | both learned | 68.82% / .619 | 67.63% / .631 | **+1.19 pp** |

*(recall / residency, canonical W=16)*

On the one seed where both arms trained cleanly, transport is worth **+1.2 pp** — a quarter of the gate threshold, and its residency is actually *lower* (.619 vs .631). On the seed where the arms diverge, transport loses by 36.8 pp.

## Why seed 0 collapsed — the mechanism

| Metric, canonical W=16, seed 0 | `phl_lease` | `static_priority` |
|---|---:|---:|
| Write gate at bindings | 0.909 | 0.625 |
| Write gate elsewhere | **0.221** | 0.022 |
| Eviction decisions | **10,676** | 1,177 |
| Residency | 0.113 | 0.571 |
| Final training recall CE | 2.411 (collapsed) | 1.374 |

`phl_lease` seed 0 reached recall CE 0.517 mid-training — the best value any arm in this experiment attained — and then regressed to 2.411 by step 800. It ends up writing at non-binding positions ten times as often as its static twin and making nine times as many eviction decisions, with a fifth of the residency. The transported lease drove the controller into memory thrashing. This is the same signature as PHL-DAM-004B, where `phl_lease` was the only arm to fail on a majority of seeds.

## Measurement limitation

`residency` counts only writes whose gate crossed the 0.5 commitment threshold, so it is a **lower bound** on true content residency. This is visible in `learned_utility` seed 0, which scores 62.81% recall against 43.63% measured residency — content written at gate values just below threshold is retrievable but unrecorded, and that run logs zero eviction decisions for the same reason.

The bias does not weaken the transport comparison; it strengthens it. `phl_lease` seed 0 has the **highest** write gate of any run (0.909), so the under-counting runs in its favour, and its residency is still five times lower than its static twin's.

## Attribution

**Did temporal PHL transport help?** No. With the timing signal decoded at the Bayes ceiling and the horizons correctly calibrated, transport is worth +1.2 pp on the clean seed and −36.8 pp on the unstable one. Both are decisive against a +5 pp gate, in opposite ways.

**Was the 004B failure just an unlearnable lease?** Partly, and it does not rescue the hypothesis. Supervision fixed the predictor completely (AUROC 0.74→0.85) and lifted both lease arms substantially — `static_priority` went from 36.79% to 59.37%/67.63% recall on its learning seeds. The gain belongs to *having* a relevance estimate, not to transporting it.

**Is the corrected transport operator now doing the right thing?** Yes, and it still does not help. Far-horizon mass now peaks at "due" exactly as the far delay elapses, verified by test at both scales. The mechanism works as designed; the design does not buy anything.

## Failure analysis

Only 2 of 3 seeds produced a learning run for any arm — seed 1 failed for all four arms identically, which is a seed-level content-path failure rather than an arm effect. That leaves the central contrast resting on two informative seeds, one of which is a collapse. Three seeds cannot resolve a 5 pp effect, and this report does not claim to have measured one precisely; it claims the observed effect is not a ≥5 pp advantage, which two seeds disagreeing in sign by 38 pp comfortably establish.

## Scope

**Establishes.** With timing supervision sufficient to decode the cue at the Bayes ceiling, and horizons calibrated to the task, PHL-transported temporal relevance does not improve retention over a parameter-identical static priority at 8 slots and 12–16 writes on this benchmark, and can destabilise the write controller badly enough to collapse training.

**Does not establish.** That temporal relevance is useless in general, or that no transport scheme could work — only that this operator, in this architecture, at this scale, does not earn its complexity. It says nothing about the brief's full-scale protocol (still a null on content-path grounds), nor about PHL-DAM versus Transformers, attention, scaling, reasoning, or asymptotic efficiency.

## Decision

**KILL LEASE HYPOTHESIS.** Gate 1 passes, Gate 2 fails, and the failure is not attributable to an unlearnable predictor, a mis-calibrated operator, or a missing signal — all three were fixed and the result did not move in the hypothesis's favour. Per the preregistered decision rule, the current lease/transport approach should not be carried forward, scaled, or added to the PHL-DAM backbone.

The finding that survives is not PHL's: a learned eviction score from purely local features (`content_only`, 57 parameters) and a static learned priority both beat LRU consistently. Memory management in this architecture benefits from learning, and specifically not from temporal transport.

## Reproduction

```
cd C:\Users\liamp\Documents\Codex\2026-08-28\tes\work
py -3.14 -m unittest -v
py -3.14 phl_dam_004b_lease.py --arm ARM --scale compact --seed N --steps 800 \
    --timing-weight 1.0 --eval-episodes 512 --output ..\outputs\phl_dam_004bs_ARM_seedN.json
py -3.14 phl_dam_004b_aggregate.py --prefix phl_dam_004bs_ --seeds 0 1 2 \
    --output ..\outputs\phl_dam_004bs_aggregate.json
```

ARM ∈ {content_only, static_priority, phl_lease, learned_utility}; N ∈ {0,1,2}.
Environment: Python 3.14.0, torch 2.13.0+cpu, CPU only, one thread per process.
