# PHL-DAM-004C (v2, corrected) — What is associated with the full-scale collapse?

> Supersedes `PHL_DAM_004C_Scale_Collapse_Report.md`. That version overstated the
> causal claim, compared two different evaluation pressures in a single "Recall"
> column, and miscounted the diverged runs. All three are corrected here; the
> underlying run artifacts are unchanged and were not re-run.

## Verdict

The full-scale collapse is most strongly associated with the high write-load /
slot-over-subscription regime. Sequence length up to approximately 451 tokens is
insufficient by itself to reproduce the collapse — at 8/12/16 writes the model
learns at every length tested, needing only more updates as length grows — while
extending the full-scale training budget to 1400 updates does not rescue it.
**This experiment does not prove that write count alone is causal.** Write count
and sequence length are confounded by construction in these profiles, and the
controlled write-count experiment that could establish causation is
PHL-DAM-004D, which had not been run when these data were collected.

## Research question

PHL-DAM-004B produced a null at the brief's full scale: no arm reached its recall
breakthrough, every policy sat at chance, and eviction quality was unmeasurable.
The scale ladder showed the identical model reaching 99.18% recall at 176 tokens.
Which factor is responsible — sequence length, write count, or training budget?

## Method

Two interventions, both probing with `content_only` (the arm that learned most
reliably in 004B, so the measurement concerns the content path rather than the
lease).

**Temporal dilation.** Four profiles stretch positions and delays by 1.0×, 1.5×,
2.0× and 2.6× from the compact base while holding the task fixed: same write
counts, same query budgets, same class mixture, same cue strength. A test
asserts the invariance — peak concurrent live items varies by less than 0.8
across profiles and queries per episode by less than 0.8, while sequence length
more than doubles.

| Profile | Sequence length | Delay range | Peak live items | Queries/episode |
|---|---:|---:|---:|---:|
| `dilate10` | 183 | 29–104 | 7.42 | 8.9 |
| `dilate15` | 267 | 44–156 | 7.14 | 8.8 |
| `dilate20` | 351 | 58–208 | 7.31 | 8.9 |
| `dilate26` | 451 | 75–270 | 7.26 | 8.8 |

**Budget extension.** The full scale (456 tokens, writes 16/24/32, delays
32–256) rerun at 1400 updates instead of 600, two seeds.

## Results — all recall figures at canonical W=16

**Correction.** The v1 table reported dilation rows at canonical W=12 and
full-scale rows at canonical W=16 under one undifferentiated "Recall" heading,
because the two profile families expose different evaluation settings and the
selector silently fell back. Canonical W=16 is the only setting both families
share, so it is the comparison column here. Canonical W=12 is shown separately
and is blank for full-scale runs, which do not evaluate it.

| Run | Length | Train writes | Steps | Breakthrough | Final recall CE | **W=16 recall** | **W=16 residency** | W=12 recall | Finite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dilate10` s0 | 183 | 8/12/16 | 800 | **200** | 2.076 | **29.96%** | .282 | 33.72% | yes |
| `dilate15` s0 | 267 | 8/12/16 | 800 | **350** | 1.293 | **57.82%** | .375 | 70.71% | yes |
| `dilate20` s0 | 351 | 8/12/16 | 800 | **350** | 1.680 | **64.66%** | .624 | 78.45% | yes |
| `dilate26` s0 | 451 | 8/12/16 | 800 | **550** | 2.003 | **37.14%** | .331 | 44.54% | yes |
| **full s0** | **456** | **16/24/32** | **1400** | **none** | 2.323 | **12.08%** | .004 | — | yes |
| **full s1** | **456** | **16/24/32** | **1400** | **none** | 2.308 | **10.59%** | .159 | — | yes |
| `dilate10` s1 | 183 | 8/12/16 | 800 | none | **nan** | 0.00% | .000 | 0.00% | **no** |
| `dilate15` s1 | 267 | 8/12/16 | 800 | none | 2.309 | 10.66% | .054 | 9.01% | yes |
| `dilate20` s1 | 351 | 8/12/16 | 800 | none | 2.273 | 9.35% | .193 | 8.49% | yes |
| `dilate26` s1 | 451 | 8/12/16 | 800 | none | 2.295 | 10.66% | .000 | 10.67% | yes |

The headline same-setting contrast: **`dilate26` seed 0 reaches 37.14% at
canonical W=16 while full-scale seed 0 reaches 12.08%**, five tokens apart in
sequence length.

### Sequence length is not sufficient to cause the collapse

At 8/12/16 writes the model learns at every length from 183 to 451 tokens.
Breakthrough step rises with length (200 → 350 → 350 → 550), so length raises the
cost of learning without preventing it. A 451-token episode — within five tokens
of full scale — reaches 37.14% recall at the same evaluation pressure where full
scale reaches 12.08%.

### Training budget is not the explanation

Full scale at 1400 updates — more than twice the 600 that produced the original
null, and well past the step-550 breakthrough observed at comparable length —
still sits at chance on both seeds (recall CE 2.323 and 2.308). This rules out
the budget-shortfall interpretation.

### What remains: write load, as an association

The remaining difference between `dilate26` (451 tokens, learns by step 550) and
full scale (456 tokens, fails at 1400) is the write count: 8/12/16 versus
16/24/32, i.e. 1–2× the eight slots versus 2–4×.

**This is an association, not a demonstrated cause.** Write count and sequence
length are confounded by construction: 32 write events at four tokens each
occupy 128 tokens, and a 256-token delay must fit after them, so high write
counts *require* long sequences and the complementary cell — short sequence, 32
writes — is geometrically impossible in these profiles. What the data establish
is that length up to 451 tokens does not by itself reproduce the collapse, and
that the collapse coincides with doubled write load. Isolating write count
requires a profile in which it can vary at a genuinely fixed sequence length,
which is what PHL-DAM-004D was built to provide.

## Run accounting

**Correction.** v1 said the divergence was "one run in six". The artifact set
contains **10** 004C runs: **8** temporal-dilation runs (four profiles × two
seeds) and **2** full-scale budget runs.

| Quantity | Count |
|---|---:|
| Total 004C runs | 10 |
| Temporal-dilation runs | 8 |
| Full-scale budget runs | 2 |
| Non-finite runs | **1** (`dilate10` seed 1) |
| Non-finite share | **1 of 8 dilation runs; 1 of 10 total** |
| Runs reaching breakthrough | 4 |

## Two failures worth recording

**A run diverged to NaN.** `dilate10` seed 1 shows gradient norms climbing
2.4e7 → 1.4e4 → 1.6e5 → 1.3e7 → 1.7e11 before the parameters went non-finite;
the artifact records `"finite": false`. No reported result depends on it. The
test suite at the time asserted finite gradients on a fresh model for a single
step, so it structurally could not catch a divergence developing over 175
updates. PHL-DAM-004D adds per-step finiteness checks, gradient-threshold
crossing records, and a long-horizon training test that trains far enough to
catch delayed instability.

**Seed 1 fails systematically.** It produced no breakthrough at `dilate15`,
`dilate20`, `dilate26` or full scale, and in PHL-DAM-004B-S it failed for all
four arms identically. This is a seed-level property of the initialisation and
stream, not an arm, scale or length effect. With three seeds, roughly one in
three being uninformative materially reduces the effective sample of every
comparison in this line — which is why PHL-DAM-004D uses five seeds and a
breakthrough criterion frozen before any comparison.

## Scope

**Establishes.** Temporal dilation from 183 to 451 tokens, with the task held
statistically fixed, does not prevent the finite-capacity PHL-DAM from learning
content recall; it raises the required budget. The full-scale configuration does
not learn within 1400 updates, so its null is not a budget artefact. The
collapse coincides with a doubling of write load.

**Does not establish.** That write count is causal, sole or sufficient — it
cannot be separated from the sequence length it requires in these profiles. It
does not identify the mechanism by which over-subscription would block
optimisation. It says nothing about the lease hypothesis, settled separately at
compact scale, and nothing about PHL-DAM versus any other architecture.

## Consequence for PHL-DAM-004B

PHL-DAM-004B described its full-scale null as reflecting "the content
controller's reach at 456 tokens under this budget". The budget half of that is
now refuted and the length half is not supported: 451 tokens is within reach at
lower write load. The defensible statement is that the null is associated with
**2–4× slot over-subscription**, which at these delay ranges forces a ~456-token
sequence. Neither the 004B nor the 004B-S verdict changes — both rest on
compact-scale comparisons where every arm had ample budget.

## Reproduction

```
cd C:\Users\liamp\Documents\Codex\2026-08-28\tes\work
py -3.14 -m unittest test_phl_dam_pressure_task -v
py -3.14 phl_dam_004b_lease.py --arm content_only --scale dilate10|dilate15|dilate20|dilate26 \
    --seed N --steps 800 --eval-episodes 256 --output ..\outputs\phl_dam_004c_SCALE_seedN.json
py -3.14 phl_dam_004b_lease.py --arm content_only --scale full --seed N --steps 1400 \
    --eval-episodes 256 --output ..\outputs\phl_dam_004c_budget_full_seedN.json

# Regenerate every figure in this report from the artifacts:
py -3.14 phl_dam_report_stats.py --pattern "phl_dam_004c_*.json" --setting canonical_w16
```

Environment: Python 3.14.0, torch 2.13.0+cpu, CPU only, one thread per process.
