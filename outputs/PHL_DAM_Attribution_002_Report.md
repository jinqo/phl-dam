# PHL-DAM Attribution-002 — Parameter-Matched DAM-only

**Verdict: the PHL+DAM robustness advantage survives parameter matching.**

The matched PHL-off model has **33,098 trainable parameters**, only 64 more
than PHL+DAM's 33,034—a **0.19% difference**. It preserves the same eight
explicit slots, 24-dimensional keys/values, controller layer shapes, streams,
objective, seeds, and 500-step budget. Leases and promotion remain absent.

The added capacity is a generic residual `64 → 64 → 64` GELU FFN on the local
context path. The matched model contains zero PHL parameters and zero PHL
recurrent state.

## Aggregate comparison

| Model | Parameters | Recall (mean ± sample SD) | Seeds ≥20% | Minimum | All-token CE | State |
|---|---:|---:|---:|---:|---:|---:|
| **PHL+DAM** | 33,034 | **95.24% ± 7.66 pp** | **3/3** | **86.40%** | **0.2353** | 456 floats |
| DAM-only-matched | 33,098 | 69.96% ± 51.91 pp | 2/3 | 10.02% | 0.2665 | **392 floats** |

PHL+DAM leads mean recall by **25.29 percentage points** and all-token CE by
**0.0312 nats**, despite having slightly fewer parameters. DAM-only-matched
retains a 14.04% recurrent-state advantage.

## Paired seed results

| Seed | PHL+DAM | DAM-only-matched | Matched retrieval disabled |
|---:|---:|---:|---:|
| 0 | 86.40% | 10.02% | 10.02% |
| 1 | 99.62% | 99.95% | 8.10% |
| 2 | 99.72% | 99.90% | 10.30% |

The matched model failed seed 0 because its learned controller did not select
bindings: mean write strength was `0.00387` at binding positions versus
`0.01717` elsewhere, mean query read strength was only `0.0972`, and address
top-1 was 40.25%. Successful seeds reached effectively perfect recall and
100% address top-1.

This is therefore an optimization/discovery difference rather than a memory
capacity ceiling: generic parameter count did not remove the one-seed collapse,
while PHL+DAM learned in all three seeds.

## Interpretation

The earlier PHL+DAM advantage cannot be explained simply by having 8,256 more
parameters than the original DAM-only model. A slightly larger PHL-off control
still achieved only 2/3 successful seeds under the same budget.

This strengthens evidence that the PHL path—or the way its computation is
inserted—improves controller-learning robustness. It does **not** yet prove
that temporal prediction is the causal feature: this test uses one generic
capacity placement and only three initializations. A wider-seed confirmation
or a controller-width-matched alternative would distinguish a repeatable PHL
inductive-bias effect from residual architecture/initialization sensitivity.

## Verification

- 33,098 parameters: within 1% of the 33,034 target
- PHL completely off; no PHL parameters or state
- unchanged 8 slots, `d_key=24`, `d_value=24`
- same streams, objective, seeds, and training/evaluation budgets
- leases off; promotion off
- 13/13 focused tests passed before training
- 6,000 held-out episodes / 18,000 recall queries for the matched control
- 0 NaN / Inf
