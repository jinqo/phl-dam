# PHL-DAM-004D — Write-pressure ladder at fixed sequence length

## Verdict

**The write-pressure hypothesis is not established, and this experiment cannot
establish it, because a pervasive gradient instability contaminates every level
including the lowest-pressure baseline.** Seven of twenty-five runs diverged to
non-finite parameters, two of them at the W=8 baseline, and the baseline itself
learned on only 2 of 5 seeds. The learned-rate across the ladder — 2/3, 1/4, 0/3,
1/4, 0/4 among finite runs — trends downward but is not monotone and rests on
three to four usable runs per level. What the experiment *does* deliver is a
clean mechanistic signature of the failure and evidence on its temporal ordering:
runs that fail lose write selectivity and usually **invert** it, and gradient
explosion precedes selectivity loss far more often than the reverse. Write
pressure remains the leading candidate but must be re-tested after the
optimisation instability is fixed.

## Research question

Why does the write/content controller stop learning under high slot
over-subscription? PHL-DAM-004C could only report an *association*, because write
count and sequence length are confounded by construction in those profiles. 004D
removes that confound.

## Frozen protocol

The `pressure` scale profile pins everything except write count.

| Quantity | Value |
|---|---:|
| Sequence length | **456, fixed at every level** |
| Write region end | 190, fixed |
| Delay range | **32–256, fixed** |
| Query budget | 6–9, fixed |
| Slots | 8 |
| Write levels | 8 / 16 / 20 / 24 / 32 |
| Seeds | 0–4 (5 seeds, identical set at every level) |
| Steps | 700, batch 16, lr 2e-3, AdamW |
| Arm | `content_only` (no lease, no timing supervision) |

Measured invariance across the ladder (asserted by test): queries per episode
6.69 → 7.27, live items 5.45 → 5.91, peak concurrent live 4.58 → 5.02, last write
position 193 — all effectively constant, while **never-queried distractor writes
rise 2.55 → 26.09**. Live items stay below slot capacity at every level, so
pressure comes purely from distractor traffic competing for the same eight slots.

**Preregistered before any comparison:** a seed is content-path-informative iff
that seed's **W=8 baseline** run ends below recall CE 2.0. Decided by the
baseline level alone, never by the condition under test.

## Result 1 — the instability that dominates everything

| Level | Runs | Diverged | Learned (all) | Learned (finite only) |
|---|---:|---:|---:|---:|
| W=8 | 5 | **2** | 2/5 | 2/3 |
| W=16 | 5 | 1 | 1/5 | 1/4 |
| W=20 | 5 | **2** | 0/5 | 0/3 |
| W=24 | 5 | 1 | 1/5 | 1/4 |
| W=32 | 5 | 1 | 0/5 | 0/4 |
| **Total** | **25** | **7 (28%)** | **4/25** | **4/18** |

Gradient norms are extreme throughout: 17 of 25 runs cross 1e3, and 13 cross 1e9,
with maxima up to **1.46e19**. Crossings usually happen at **step 1–3**, before
any controller behaviour could have developed.

Max gradient norm separates outcomes sharply:

| Outcome | n | Median max gradient norm |
|---|---:|---:|
| Learned | 4 | 7.75e6 |
| Did not learn | 21 | **1.56e11** |

Because the baseline is itself affected — 2 of 5 W=8 runs diverged and only 2 of
5 learned — the ladder cannot be read as a clean dose-response curve. This is the
central limitation of the experiment and it is not repairable by reanalysis.

## Result 2 — the failure signature is loss and inversion of write selectivity

Write selectivity is `write_gate_at_binding − write_gate_elsewhere`, measured on
a held-out probe batch disjoint from training.

| Outcome | n | Mean selectivity | Median | Runs with **negative** selectivity |
|---|---:|---:|---:|---:|
| Learned | 4 | **+0.666** | +0.696 | **0 of 4** |
| Did not learn | 14 | −0.015 | −0.028 | **11 of 14** |

This is **failure mode A** from the hypothesis, and it is stark: failing
controllers do not merely lose selectivity, they *invert* it — writing more
readily at non-binding positions than at the actual bindings. Examples at the end
of training: W=24 s4 gate 0.052 at bindings versus 0.550 elsewhere
(selectivity −0.498); W=32 s4 0.038 versus 0.348 (−0.310); W=16 s4 0.039 versus
0.316 (−0.277).

Failure mode B (high gate everywhere with random allocation) is **not** the
dominant pattern here. Allocation entropy at bindings stays well below the
ln 8 = 2.079 ceiling in every run (range 0.06–1.76), so the allocator does not
become uniform; the write gate fails first.

The four learned runs show the expected healthy pattern — W=8 s4: gate 0.997 at
bindings versus 0.011 elsewhere, recall 99.84%, residency 99.92%.

## Result 3 — temporal ordering: gradient first, then gate

Comparing, per run, the first step where the gradient norm exceeds 1e3 against
the first step where selectivity goes non-positive (25-step telemetry
resolution):

| Which came first | Runs |
|---|---:|
| **Gradient explosion first** | **11** |
| Same telemetry step (ambiguous) | 4 |
| Selectivity loss first | 2 |
| Only gradient event occurred | 1 |
| Only selectivity event occurred | 4 |
| Neither | 3 |

Among the 17 runs where both events occur, gradient-first outnumbers
selectivity-first **11 to 2**, with 4 ambiguous. This favours

> gradient explosion → gate failure → memory thrashing

over the reverse ordering. Two caveats keep this from being conclusive: the
25-step telemetry interval cannot resolve events inside the same window, and 11
of the gradient crossings occur at step 1–3, so early gradients may reflect
initialisation transients rather than a developing pathology.

## Result 4 — what the ladder does and does not show

Restricted to the two preregistered informative seeds (0 and 4):

| Level | Learned / total | Learned / finite | Detail |
|---|---:|---:|---|
| W=8 | 2/2 | 2/2 | s0 learn, s4 learn |
| W=16 | 0/2 | 0/1 | s0 diverged, s4 fail |
| W=20 | 0/2 | 0/1 | s0 diverged, s4 fail |
| W=24 | 0/2 | 0/2 | s0 fail, s4 fail |
| W=32 | 0/2 | 0/2 | s0 fail, s4 fail |

**The W=8 row is tautological** — informativeness is defined by it and carries no
evidential weight. The non-tautological observation is that the same two seeds
that learn at W=8 fail at *every* higher level, 0 of 6 finite runs. That is
consistent with write pressure mattering.

It is also weak: two seeds, and no localisation of a transition — performance
does not degrade gradually but fails immediately at W=16, the first step up. A
sharp cliff between W=8 and W=16 is equally consistent with "any pressure above
1× capacity breaks this controller" and with "the gradient instability happens to
bite harder once more write events exist". The experiment cannot separate those.

## Diagnostics summary

| Level | Mean selectivity | Gate at binding | Gate elsewhere | Allocation entropy at binding | Mean occupied slots |
|---|---:|---:|---:|---:|---:|
| W=8 | +0.382 | 0.407 | 0.026 | 0.815 | 5.03 |
| W=16 | +0.063 | 0.185 | 0.122 | 0.861 | 6.82 |
| W=20 | +0.115 | 0.363 | 0.248 | 0.798 | 6.94 |
| W=24 | +0.067 | 0.374 | 0.307 | 0.606 | 7.17 |
| W=32 | **−0.085** | 0.164 | 0.249 | 0.935 | 6.92 |

Mean selectivity falls from +0.382 at W=8 to negative at W=32, and mean occupied
slots rises 5.03 → 6.92 as distractor traffic fills memory. These are averages
over few runs, mixing learners and non-learners, and should be read as
descriptive.

Evaluation-time eviction counts are 0 in every run, because the residency ledger
only records writes whose gate exceeds 0.5 and most runs never sustain that. This
is the same lower-bound limitation documented in PHL-DAM-004B-S, and it makes
eviction-based diagnostics uninformative in this experiment.

## Scope

**Establishes.** At a genuinely fixed sequence length, delay distribution and
query budget, the controller's failure signature is loss and frequently inversion
of write selectivity, not allocator randomisation. Gradient explosion precedes
selectivity loss in the large majority of runs where both occur. A severe
optimisation instability affects this configuration at every write level,
including the 1×-capacity baseline.

**Does not establish.** That write count causes the collapse. The ladder is not
monotone, the baseline is compromised, only 2 of 5 seeds are informative, and
28% of runs diverged. It does not identify the source of the gradient explosion.
It does not rule out write pressure — the leading candidate is untouched — but it
cannot confirm it.

## Decision

**Fix the optimisation instability first, then re-run this ladder.** Testing
write pressure on a configuration where the baseline diverges 40% of the time
cannot produce a clean answer no matter how many seeds are added.

The evidence points at the gradient path, not the task: crossings at step 1–3,
maxima to 1e19, and a clean separation between learned (median 7.75e6) and failed
(median 1.56e11) runs. The single highest-value next step is to find and bound
whatever produces those gradients — the slot-key normalisation and the
straight-through allocation are the two structures already known to have
pathological Jacobians in this model.

## Recommended next steps

1. **Locate the gradient source.** Per-module gradient norms across the first ten
   steps, at W=8 where the task is easy, to identify which path produces 1e11.
2. **Bound it, then re-run 004D unchanged.** Candidates already suggested by
   earlier defects: normalisation epsilon, straight-through scaling, controller
   warm-up, or gradient normalisation around the controller. Change one thing.
3. **Only then** interpret the write-pressure ladder, with 8–10 seeds.
4. Do not resume lease work; see `PHL_DAM_Lease_Pure_Transport_Design.md`.

## Reproduction

```
cd C:\Users\liamp\Documents\Codex\2026-08-28\tes\work
py -3.14 -m unittest -v
for W in 8 16 20 24 32; for S in 0 1 2 3 4:
py -3.14 phl_dam_004d_write_pressure.py --writes W --seed S --steps 700 \
    --eval-episodes 192 --output ..\outputs\phl_dam_004d_wW_seedS.json

py -3.14 phl_dam_report_stats.py --pattern "phl_dam_004d_*.json"
```

Outputs: `phl_dam_004d_w{8,16,20,24,32}_seed{0..4}.json` (25 files).
Environment: Python 3.14.0, torch 2.13.0+cpu, CPU only, one thread per process.
