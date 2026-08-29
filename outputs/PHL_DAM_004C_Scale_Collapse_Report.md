> **SUPERSEDED by `PHL_DAM_004C_Scale_Collapse_Report_v2.md`.** This version
> (a) overstated the causal claim about write count, (b) reported dilation rows
> at canonical W=12 alongside full-scale rows at canonical W=16 under one
> undifferentiated "Recall" column, and (c) said the divergence was "one run in
> six" when the artifact set holds 8 dilation runs and 10 runs in total. Read v2.

# PHL-DAM-004C — What blocks learning at the full 004 scale?

## Verdict

The full-scale content-path failure is caused by the **write count**, not by sequence length and not by training budget. Stretching the task in time from 183 to 451 tokens, holding writes and every other statistic fixed, leaves the model learning at every length tested — the required budget grows (breakthrough at step 200, 350, 350, 550) but learning still happens. Running the full scale for 1400 updates, 2.3× the original budget, still produces no breakthrough on either seed. The difference between the two is the write count: 8/12/16 writes (1–2× the eight slots) learns; 16/24/32 writes (2–4× capacity) does not. An important caveat limits how cleanly this can be attributed: write count and sequence length are confounded **by construction** in this task family, because 32 four-token write events plus a 256-token delay cannot fit in a short sequence. Length is therefore exonerated as an independent cause, but write count cannot be isolated from the length it requires.

## Research question

PHL-DAM-004B produced a null at the brief's full scale: no arm reached its recall breakthrough, every policy sat at chance, and eviction quality was unmeasurable. The scale ladder showed the identical model reaching 99.18% recall at 176 tokens. Which factor is responsible — sequence length, write count, or simply an insufficient training budget?

## Method

Two controlled interventions, both using `content_only` as the probe (the arm that learned most reliably in 004B, so the measurement is about the content path rather than the lease).

**Temporal dilation.** Four scale profiles stretch positions and delays by 1.0×, 1.5×, 2.0× and 2.6× from the compact base while holding the task itself fixed: same write counts, same query budgets, same class mixture, same cue strength. A test asserts the invariance — peak concurrent live items stays within 0.8 across profiles and queries per episode within 0.8, while the sequence length more than doubles.

| Profile | Sequence length | Delay range | Peak live items | Queries/episode |
|---|---:|---:|---:|---:|
| `dilate10` | 183 | 29–104 | 7.42 | 8.9 |
| `dilate15` | 267 | 44–156 | 7.14 | 8.8 |
| `dilate20` | 351 | 58–208 | 7.31 | 8.9 |
| `dilate26` | 451 | 75–270 | 7.26 | 8.8 |

**Budget extension.** The full scale (456 tokens, writes 16/24/32, delays 32–256) rerun at 1400 updates instead of 600, two seeds.

## Results

| Run | Length | Writes | Steps | Breakthrough | Final recall CE | Recall | Residency | Finite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dilate10` s0 | 183 | 8/12/16 | 800 | **step 200** | 2.076 | 33.72% | .312 | yes |
| `dilate15` s0 | 267 | 8/12/16 | 800 | **step 350** | 1.293 | 70.71% | .485 | yes |
| `dilate20` s0 | 351 | 8/12/16 | 800 | **step 350** | 1.680 | 78.45% | .761 | yes |
| `dilate26` s0 | 451 | 8/12/16 | 800 | **step 550** | 2.003 | 44.54% | .431 | yes |
| **full s0** | **456** | **16/24/32** | **1400** | **none** | 2.323 | 12.08% | .004 | yes |
| **full s1** | **456** | **16/24/32** | **1400** | **none** | 2.308 | 10.59% | .159 | yes |
| `dilate10` s1 | 183 | 8/12/16 | 800 | none | **nan** | — | — | **no** |
| `dilate15` s1 | 267 | 8/12/16 | 800 | none | 2.309 | 9.01% | .065 | yes |
| `dilate20` s1 | 351 | 8/12/16 | 800 | none | 2.273 | 8.49% | .229 | yes |
| `dilate26` s1 | 451 | 8/12/16 | 800 | none | 2.295 | 10.67% | .000 | yes |

### Length is not the barrier

At 8/12/16 writes the model learns at every length from 183 to 451 tokens. The breakthrough step rises with length (200 → 350 → 350 → 550), so length does raise the cost of learning, but it does not prevent it. A 451-token episode — within five tokens of the full scale's 456 — reaches 44.54% recall.

### Budget is not the explanation

The full scale at 1400 updates, more than twice the 600 that produced the original null and well beyond the step-550 breakthrough seen at comparable length, still sits at chance on both seeds (recall CE 2.323 and 2.308). This rules out the interpretation I had provisionally drawn from the dilation trend alone, that the 004B null was simply a budget shortfall. It was not.

### What remains: the write count

The only substantive difference between `dilate26` (451 tokens, learns by step 550) and the full scale (456 tokens, fails at 1400) is the write count: 8/12/16 versus 16/24/32, i.e. 1–2× the eight slots versus 2–4×.

**This attribution cannot be made airtight, and it would be wrong to present it as such.** Write count and sequence length are confounded by construction: 32 write events at four tokens each occupy 128 tokens, and a 256-token delay must then fit after them, so high write counts *require* long sequences. The complementary cell — short sequence, 32 writes — is geometrically impossible in this task family. What the data establish is that length up to 451 tokens is not sufficient to cause the collapse, and that the collapse appears when the write count doubles. Write count is the remaining candidate, not a demonstrated cause.

## Two failures worth recording

**A run diverged to NaN.** `dilate10` seed 1 shows gradient norms climbing 2.4e7 → 1.4e4 → 1.6e5 → 1.3e7 → 1.7e11 before the parameters went non-finite; the artifact records `"finite": false`. No reported result depends on it. The test suite asserts finite gradients on a fresh model for a single step, so it structurally cannot catch a divergence developing over 175 updates — a real limitation of the tests, not a false pass. One run in six here; all 12 PHL-DAM-004B and all 12 004B-S runs were finite.

**Seed 1 fails systematically.** It produced no breakthrough at `dilate15`, `dilate20`, `dilate26` or full scale, and in PHL-DAM-004B-S it failed for all four arms identically. This is a seed-level property of the initialisation and stream, not an arm, scale or length effect. Any experiment in this line using three seeds should expect roughly one to be uninformative, which materially reduces the effective sample size of every comparison reported so far.

## Scope

**Establishes.** Temporal dilation from 183 to 451 tokens, with the task held statistically fixed, does not prevent the finite-capacity PHL-DAM from learning content recall; it raises the required budget. The full-scale configuration does not learn within 1400 updates, so its null is not a budget artefact. The distinguishing factor between the two is write count.

**Does not establish.** That write count is the sole or sufficient cause — it cannot be separated from the sequence length it requires. It does not identify the mechanism by which over-subscription blocks optimisation. It says nothing about the lease hypothesis, which was settled at compact scale in PHL-DAM-004B-S, and nothing about PHL-DAM versus any other architecture.

## Consequence for the earlier reports

PHL-DAM-004B described its full-scale null as reflecting "the content controller's reach at 456 tokens under this budget". That phrasing survives the budget test but is imprecise about length: 451 tokens is within reach at lower write counts. The accurate statement is that the null reflects the controller's reach at **2–4× slot over-subscription**, which at these delay ranges forces a ~456-token sequence. Neither the 004B nor the 004B-S verdict changes — both rest on compact-scale comparisons where every arm had ample budget.

## Recommended next steps

1. Establish what fails at high over-subscription: instrument the write gate and allocation entropy across the 8/12/16 → 16/24/32 boundary at fixed length, e.g. writes 16, 20, 24 at 451 tokens, to find where learning stops.
2. Add a divergence guard and a long-horizon finiteness test; the current suite cannot detect training-time divergence.
3. Use five or more seeds in this line, given that roughly one seed in three is uninformative.

## Reproduction

```
cd C:\Users\liamp\Documents\Codex\2026-08-28\tes\work
py -3.14 -m unittest test_phl_dam_pressure_task -v
py -3.14 phl_dam_004b_lease.py --arm content_only --scale dilate10|dilate15|dilate20|dilate26 \
    --seed N --steps 800 --eval-episodes 256 --output ..\outputs\phl_dam_004c_SCALE_seedN.json
py -3.14 phl_dam_004b_lease.py --arm content_only --scale full --seed N --steps 1400 \
    --eval-episodes 256 --output ..\outputs\phl_dam_004c_budget_full_seedN.json
```

Environment: Python 3.14.0, torch 2.13.0+cpu, CPU only, one thread per process.
