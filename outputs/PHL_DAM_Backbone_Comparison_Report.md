# PHL+DAM vs DAM-only vs Fast-weight/Delta

**Verdict: PHL+DAM wins this fixed-budget three-seed screen, primarily through
controller-learning robustness.**

All models used the same three-binding protocol, token sequences, objective,
500-step budget, seeds, and 2,000-episode evaluation. Temporal leases and
promotion remained absent.

## Aggregate results

| Model | Recall (mean ± sample SD) | Seeds ≥20% | Minimum | All-token CE | Retrieval disabled |
|---|---:|---:|---:|---:|---:|
| **PHL+DAM** | **95.24% ± 7.66 pp** | **3/3** | **86.40%** | **0.2353** | 10.23% |
| DAM-only | 70.12% ± 51.54 pp | 2/3 | 10.60% | 0.2612 | 10.30% |
| Fast-weight/delta | 70.10% ± 51.76 pp | 2/3 | 10.33% | 0.2500 | 10.16% |

PHL+DAM leads mean recall by **25.13 percentage points** over DAM-only and
**25.14 points** over fast-weight. It is also the only model to learn the
controller in all three seeds under the fixed 500-step budget.

## Per-seed recall

| Seed | PHL+DAM | DAM-only | Fast-weight/delta |
|---:|---:|---:|---:|
| 0 | 86.40% | 99.83% | 100.00% |
| 1 | 99.62% | 10.60% | 99.97% |
| 2 | 99.72% | 99.92% | 10.33% |

When DAM-only or fast-weight discovers the controller, it reaches essentially
perfect recall. Their lower means come from one chance-level seed each, not a
lower learned capacity ceiling. The failed DAM-only seed closes its read gate
at queries; the failed fast-weight seed neither selects binding writes nor
opens query reads. These are controller-discovery failures.

## Size and recurrent state

| Model | Trainable parameters | Recurrent state / sequence |
|---|---:|---:|
| PHL+DAM | 33,034 | 456 floats |
| DAM-only | 24,778 | **392 floats** |
| Fast-weight/delta | 24,779 | 576 floats |

PHL+DAM has 33.32% more parameters than DAM-only. DAM-only uses 14.04% less
recurrent state than PHL+DAM, while this 24×24 fast-weight matrix uses 26.32%
more. Consequently the present result is evidence for a possible PHL
optimization-robustness contribution, but it does not isolate that contribution
from PHL's extra parameters.

Fast-weight does not earn a clear advantage here: it matches DAM-only mean
recall and seed success, has nearly identical trainable parameter count, and
uses more recurrent state. Its successful seeds do show that the delta rule is
a fully capable primitive on this task.

## Distance behavior

| Model | 29–63 | 64–95 | 96–169 |
|---|---:|---:|---:|
| PHL+DAM | **94.97%** | **95.37%** | **95.40%** |
| DAM-only | 69.92% | 70.02% | 70.42% |
| Fast-weight/delta | 70.13% | 69.77% | 70.40% |

The flat profiles show that failures are not caused by longer delays; failed
seeds remain at chance in every bin.

## What this establishes

- Explicit-slot DAM and delta-rule memory are both capable of near-perfect
  learned recall.
- PHL+DAM is materially more robust across these three initializations under a
  fixed training budget.
- PHL+DAM also has the best mean all-token CE in this synthetic protocol.
- Retrieval-disabled controls are chance-level for every model, confirming
  that successful recall is memory-dependent.
- All nine runs are finite; no lease mechanism was present.

## What it does not establish

This is not yet clean proof of a uniquely PHL-specific mechanism benefit. The
PHL model has 8,256 additional trainable parameters, and only three seeds were
tested. The strongest next attribution control would match parameter count or
controller capacity while keeping PHL disabled, followed by SSM+DAM under the
same protocol. Runtime is not compared because PHL and baseline runs were not
executed under an isolated matched-throughput setup.

## Verification

- 11/11 focused original-plus-comparison tests passed
- default PHL path retained exact deterministic step-1 equivalence after the
  PHL-off ablation switch was added
- 18,000 held-out queries per model; 54,000 total
- 0 NaN / Inf
- raw metrics and complete training histories retained for all nine runs
