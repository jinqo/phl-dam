# PHL-DAM-004B-S (v2, corrected) — Supervised-timing lease: does PHL transport add anything?

> Supersedes `PHL_DAM_004BS_Supervised_Lease_Report.md`. That version stated an
> AUROC tally that was off by two, generalised a negative result beyond what
> three seeds support, and claimed two controllers beat LRU "consistently" when
> that holds only at the highest pressure. All three are corrected here. The run
> artifacts are unchanged and were not re-run.

## Verdict

**The current PHL lease-transport implementation fails the preregistered
adoption gate and provides no evidence that its added complexity is beneficial.
It should not be carried into the main architecture in its current form.**

This is an engineering decision, not a proof that temporal relevance is useless.
The timing predictor demonstrably worked — AUROC 0.85 against a Bayes ceiling of
0.849 — so the failure cannot be blamed on an unlearnable signal, and the
transport operator was calibrated to the task's own delays, so it cannot be
blamed on mis-calibration either. But with three seeds, of which only one is
informative for this contrast, the data do not statistically establish that
transport is harmful. They establish that it did not earn adoption.

## Research question

1. **Can the timing signal be learned at all** when supervision is supplied
   equally to every learned arm? PHL-DAM-004B left this unresolved: the head
   never decoded the cue, so transport was never given a working input.
2. **Given a working prediction, does transporting it through the PHL horizon
   lattice beat holding it static?** This is the only comparison in this line
   that isolates PHL.

## What changed from PHL-DAM-004B

Two corrections, both made before any 004B-S result existed.

**Scale-derived lease horizons.** `LEASE_BIN_EDGES` had been hardcoded to the
full-scale delay ranges and was never rebuilt for the compact profile, whose
delays are 29–104. Transport rates are `1 / bin width`, so mass walked the
lattice roughly 2.5× too slowly — only 8.8% of a "far" lease reached "due" by the
time its query arrived. Horizons are now derived from `task.FIRST_USE_DELAY` at
the active scale, and a test asserts far-horizon mass peaks at "due" as the far
delay elapses, at both scales. This defect penalised `phl_lease` and nothing
else, so the 004B compact numbers for that arm were a lower bound.

**Equal timing supervision.** All four learned arms carry an identical
`lease_head` trained with the same cross-entropy against the true next-use
horizon, weight 1.0 — verified equal, every arm starting at `timing_ce = 1.8826`
on a given seed. For `content_only` and `learned_utility` the head is trained but
deliberately not wired into their eviction score, so the auxiliary gradient
reaches every arm's shared representation without converting the controls into
lease arms. Tests assert that the recall loss alone leaves those heads untouched
and that overwriting the labels changes no forward pass.

## Frozen protocol

Compact scale, 176 tokens, 8 slots, writes 8/12/16, delays 29–104, 800 updates,
batch 16, lr 2e-3, AdamW, write-budget weight 0, promotion off, seeds 0–2, 512
evaluation episodes per setting, timing weight 1.0 for every learned arm. No arm
receives future information at evaluation; the oracle is evaluation-only.

## Gate 1 — was the timing signal actually learned? **PASS**

Live-vs-never AUROC at write positions, measured against generator truth for
diagnosis only. Bayes ceiling computed from the realised class prior: **0.8490**.

| Arm | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| `content_only` | 0.8399 | 0.8468 | 0.8475 |
| `static_priority` | 0.8336 | 0.8463 | 0.8508 |
| `phl_lease` | **0.7405** | 0.8463 | 0.8515 |
| `learned_utility` | 0.8429 | 0.8463 | 0.8487 |

**Correction.** v1 said "ten of twelve runs sit within 0.005 of the
Bayes-optimal AUROC". The count is **8 of 12** (computed by
`phl_dam_report_stats.count_within`, not by hand). The four outside are
`content_only` s0 (0.0091 away), `learned_utility` s0 (0.0061),
`static_priority` s0 (0.0154) and `phl_lease` s0 (0.1085). All 12 exceed 0.74 and
11 of 12 exceed 0.83, so the signal is decoded well; the precise tally was
simply wrong.

## Gate 2 — does transport beat static? **FAIL (adoption gate)**

`phl_lease` minus `static_priority`, recall, with the full statistics rather than
a bare mean.

| Setting | Mean | Median | SD | Per seed (0/1/2) | +/−/tie | Bootstrap 95% |
|---|---:|---:|---:|---|---:|---|
| canonical W=8 | −11.45 pp | −1.46 pp | 18.54 | −33.5 / −0.0 / −1.5 | 0/3/0 | [−32.8, −0.0] |
| canonical W=12 | −14.38 pp | −1.14 pp | 23.98 | −42.1 / +0.1 / −1.1 | 1/2/0 | [−42.1, +0.1] |
| canonical W=16 | −11.86 pp | +0.03 pp | 21.62 | −36.8 / +0.0 / +1.2 | 2/1/0 | [−36.8, +1.2] |
| spec W=12 | −16.10 pp | −1.24 pp | 26.79 | −47.0 / −0.0 / −1.2 | 0/3/0 | [−47.0, −0.0] |

Required to pass: ≥ +5 pp. Observed: negative at every setting, and nowhere near
+5 pp at any seed. **The gate fails.**

At the two canonical pressure settings the bootstrap interval spans zero, so the
mean is dominated by one collapsed seed and this is **not** a statistically
established negative effect. It is a failed adoption criterion.

### Reading the seeds by what actually happened

A "2/3 seed wins" tally at W=16 is misleading, because seed 1 is a tie at chance
where neither arm learned. Preregistered informativeness — both compared arms
ending below recall CE 2.0 — leaves exactly one informative seed:

| Seed | `phl_lease` learned | `static_priority` learned | Informative | W=16 recall (phl / static) | Diff |
|---|---:|---:|---:|---:|---:|
| 0 | **no** (collapsed, CE 2.411) | yes (CE 1.374) | no | 22.55% / 59.37% | −36.8 pp |
| 1 | no (CE 2.301) | no (CE 2.302) | no | 9.62% / 9.59% | +0.03 pp |
| 2 | yes (CE 1.049) | yes (CE 1.187) | **yes** | 68.82% / 67.63% | **+1.19 pp** |

On the single informative seed transport is worth **+1.19 pp**, roughly a quarter
of the gate threshold, and its residency is slightly *lower* (.619 vs .631). One
seed cannot establish an effect of either sign; it can and does fail to
demonstrate the ≥5 pp benefit required for adoption.

## Why seed 0 collapsed — the mechanism

| Metric, canonical W=16, seed 0 | `phl_lease` | `static_priority` |
|---|---:|---:|
| Write gate at bindings | 0.909 | 0.625 |
| Write gate elsewhere | **0.221** | 0.022 |
| Eviction decisions | **10,676** | 1,177 |
| Residency | 0.113 | 0.571 |
| Final training recall CE | 2.411 (collapsed) | 1.374 |

`phl_lease` seed 0 reached recall CE 0.517 mid-training — the best value any arm
in this experiment attained — then regressed to 2.411 by step 800, ending up
writing at non-binding positions ten times as often as its static twin and making
nine times as many eviction decisions with a fifth of the residency. This is a
single observed instance of transport coinciding with a training collapse, and
the same signature appeared in PHL-DAM-004B. It is a documented risk, not a
demonstrated rate.

## Learned controllers versus LRU — corrected claim

**Correction.** v1 said `content_only` and `static_priority` "both beat LRU
consistently". Checked at every setting, that is false for these supervised runs.

Within-run contrast (identical content weights, only the eviction rule differs):

| Arm | Setting | Mean | Median | +/−/tie | Bootstrap 95% |
|---|---|---:|---:|---:|---|
| `content_only` | canonical W=8 | +0.90 pp | −0.10 pp | 1/2/0 | [−12.0, +14.8] |
| `content_only` | canonical W=12 | +1.12 pp | −0.25 pp | 1/2/0 | [−7.1, +10.7] |
| **`content_only`** | **canonical W=16** | **+3.95 pp** | +1.92 pp | **3/0/0** | **[+0.3, +9.7]** |
| `content_only` | spec W=12 | +0.97 pp | +0.08 pp | 2/1/0 | [−11.1, +13.9] |
| `static_priority` | canonical W=8 | +4.29 pp | 0.00 pp | 1/1/1 | [−2.1, +15.0] |
| `static_priority` | canonical W=12 | +4.59 pp | +0.11 pp | 2/1/0 | [−0.0, +13.7] |
| **`static_priority`** | **canonical W=16** | **+2.54 pp** | +1.11 pp | **3/0/0** | **[+0.0, +6.5]** |
| `static_priority` | spec W=12 | +6.06 pp | 0.00 pp | 1/1/1 | [−0.6, +18.8] |

Accurate statement: **at the highest canonical pressure, the content-only and
static-priority controllers outperform their within-run LRU controls across the
tested seeds. Results at lower pressures are mixed**, with intervals spanning
zero and sign tallies of 1/2 or 1/1/1.

For completeness, in the *unsupervised* PHL-DAM-004B runs `content_only` did beat
LRU 3/3 at all four settings (+9.0, +7.9, +4.5, +9.2 pp), while
`static_priority` was mixed there too (2/0/1, 3/0/0, 2/1/0, 1/1/1). The stronger
"consistent" language belongs to `content_only` in 004B only, and even there
three seeds is thin.

## Attribution

**Did temporal PHL transport help?** No. With the signal decoded near the Bayes
ceiling and horizons correctly calibrated, transport is worth +1.19 pp on the one
informative seed — far below the adoption threshold — and coincided with a
training collapse on another.

**Was the 004B failure just an unlearnable lease?** Partly, and it does not
rescue the hypothesis. Supervision fixed the predictor and lifted both lease
arms substantially: `static_priority` went from 36.79% (004B) to 59.37% and
67.63% on its learning seeds. That gain belongs to *having* a relevance
estimate, not to transporting it.

## Scope

**Establishes.** The current lease-transport implementation, in this
architecture, at this scale, on this benchmark, fails its preregistered adoption
gate at every evaluation setting, with a working timing predictor and a
correctly calibrated operator.

**Does not establish.** That temporal relevance is useless, that transport is
harmful in general, or that no transport scheme could work. The negative mean is
driven by one collapsed seed; bootstrap intervals at the canonical pressures span
zero. Three seeds with one informative seed cannot resolve a 5 pp effect.

## Decision

**Do not adopt the current PHL temporal lease transport mechanism.** Do not carry
it into the main architecture, scale it, or continue developing it on this
evidence.

If leases are ever revisited, the correct next step is the pure-transport
attribution design in `PHL_DAM_Lease_Pure_Transport_Design.md`: train and freeze
one content backbone and one timing predictor, duplicate the checkpoint, and
allow only the transport mechanism to differ. That removes the training-path
divergence which dominates the present result and asks the causal question
directly.

## Reproduction

```
cd C:\Users\liamp\Documents\Codex\2026-08-28\tes\work
py -3.14 -m unittest -v
py -3.14 phl_dam_004b_lease.py --arm ARM --scale compact --seed N --steps 800 \
    --timing-weight 1.0 --eval-episodes 512 --output ..\outputs\phl_dam_004bs_ARM_seedN.json
py -3.14 phl_dam_004b_aggregate.py --prefix phl_dam_004bs_ --seeds 0 1 2 \
    --output ..\outputs\phl_dam_004bs_aggregate.json

# Regenerate every figure in this report from the artifacts:
py -3.14 phl_dam_report_stats.py --pattern "phl_dam_004bs_*.json" --setting canonical_w16
```

ARM ∈ {content_only, static_priority, phl_lease, learned_utility}; N ∈ {0,1,2}.
Environment: Python 3.14.0, torch 2.13.0+cpu, CPU only, one thread per process.
