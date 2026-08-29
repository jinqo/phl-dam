# PHL-DAM Stage C — Three-Seed Content-Only Screen

**Verdict: PASS**

The unchanged learned-controller configuration passed every preregistered
Stage C gate across seeds 0, 1, and 2.

## Gate results

| Criterion | Required | Observed | Verdict |
|---|---:|---:|---:|
| Mean recall | ≥30% | **95.24%** | PASS |
| Lowest seed | >15% | **86.40%** | PASS |
| Finite/stable | 3/3 | **3/3** | PASS |

Temporal leases remained completely disabled: there is no lease state, lease
score, lease transport, or promotion path in any run.

## Per-seed results

Each seed used 500 training steps and was evaluated on 2,000 new episodes / 
6,000 recall queries.

| Seed | Recall | Retrieval disabled | Address top-1 | All-token CE | Recall CE |
|---:|---:|---:|---:|---:|---:|
| 0 | 86.40% | 10.18% | 92.47% | 0.2474 | 0.3748 |
| 1 | 99.62% | 10.20% | 99.10% | 0.2295 | 0.0196 |
| 2 | 99.72% | 10.30% | 99.97% | 0.2290 | 0.0154 |
| **Mean** | **95.24%** | **10.23%** | **97.18%** | **0.2353** | **0.1366** |

The average retrieval-dependent gain is **+85.02 percentage points**. Every
retrieval-disabled control remains at the ten-value chance rate, so the
three-seed result depends on DAM rather than the PHL backbone alone.

## Recall by distance

| Delay | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| 29–63 | 85.70% | 99.65% | 99.55% | **94.97%** |
| 64–95 | 86.90% | 99.40% | 99.80% | **95.37%** |
| 96–169 | 86.60% | 99.80% | 99.80% | **95.40%** |

There is no degradation with longer delay in this bounded content-only task.

## Stability interpretation

Final recall is robust, but optimization timing varies substantially:

- seed 1 discovered the controller behavior around steps 150–200;
- seed 2 broke through around steps 200–250;
- seed 0 did not break through until roughly steps 400–450.

Thus the acceptance gate passes, while sample-efficiency variance remains a
real risk to track in later rematches. Seed 2 also retained a slightly noisier
write policy (`mean write gate elsewhere = 0.00367`) than seeds 0 and 1, though
its held-out recall was unaffected.

## Scope and next decision

This screen establishes repeatable learned content recall on the current
synthetic protocol. It does not establish that PHL contributes beyond DAM,
because DAM-only and alternative-backbone controls have not yet been run on
this exact implementation. It also does not establish lease utility.

Per the research order, the next defensible work is the content-only attribution
set—especially DAM-only and the fast-weight/delta baseline—before enabling or
claiming value from temporal leases.

## Verification

- unchanged Stage B implementation and hyperparameters
- focused tests: 6/6 passed before the seed runs
- 6,000 held-out episodes / 18,000 recall queries total
- 0 NaN / Inf across all runs
- exact raw metrics and training histories retained per seed
