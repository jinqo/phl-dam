# PHL-DAM Attribution-003 — Wider-Seed Confirmation

**Verdict: CONFIRMED descriptively across six seeds.**

PHL+DAM learned the content-addressed controller in **6/6 seeds**. The
parameter-matched, PHL-off DAM learned it in **3/6 seeds** under the same
streams, objective, slots, dimensions, and 500-step budget. Temporal leases
and promotion remained absent.

## Six-seed aggregate

| Model | Parameters | Recall (mean ± sample SD) | Successes | Minimum | All-token CE | Retrieval disabled |
|---|---:|---:|---:|---:|---:|---:|
| **PHL+DAM** | 33,034 | **97.42% ± 5.40 pp** | **6/6** | **86.40%** | **0.2332** | 10.09% |
| DAM-only-matched | 33,098 | 54.80% ± 49.27 pp | 3/6 | 9.23% | 0.2715 | 9.68% |

PHL+DAM leads mean recall by **42.62 percentage points** and all-token CE by
**0.0383 nats**, despite having 64 fewer parameters. All 12 runs are finite.

## Paired seed results

| Seed | PHL+DAM | DAM-only-matched | Outcome |
|---:|---:|---:|---|
| 0 | 86.40% | 10.02% | PHL-only success |
| 1 | 99.62% | 99.95% | both succeed |
| 2 | 99.72% | 99.90% | both succeed |
| 3 | 99.33% | 99.47% | both succeed |
| 4 | 99.45% | 10.22% | PHL-only success |
| 5 | 99.98% | 9.23% | PHL-only success |

There are three discordant pairs, all favoring PHL+DAM, and none favoring the
matched control. When both models learn, their final recall differs by less
than 0.34 percentage points. The effect is therefore controller-discovery
probability, not a higher learned recall ceiling.

## Failure diagnosis

The matched model's failed seeds are 0, 4, and 5. Each shows the same pattern:

- binding write strength below filler write strength;
- query read gate suppressed;
- address top-1 near or below chance;
- retrieval-disabled accuracy identical to normal chance-level accuracy.

All six PHL runs learn selective binding writes and query reads. Recall remains
flat across delay bins, so the difference is not caused by the longest delay.

## Statistical caution

This is a stronger descriptive confirmation than the three-seed screen, but
six paired seeds are still a small, bimodal sample. The mean paired advantage
is 42.62 points; an approximate paired t interval is wide and includes zero
(-6.90 to 92.14 points). The evidence supports an engineering/research decision
to retain PHL for the next phase, not a broad statistical claim.

## Before-leases decision

The content-only PHL+DAM baseline is now stable enough to freeze for temporal
lease experiments. Leases should not be tested on the current three-write,
eight-slot task because no eviction pressure exists. The lease phase should
use constrained memory—such as 8 slots with 16–32 writes—and compare:

1. content-only retrieval;
2. content + transported lease;
3. content + random lease;
4. content + static learned priority;
5. ideally LRU and oracle-future-relevance controls.

This attribution result does not itself show lease utility.

## Verification

- unchanged models and hyperparameters; no tuning during seeds 3–5
- 13/13 focused tests passed before the wider run
- 12,000 held-out episodes / 36,000 queries per model
- 24,000 episodes / 72,000 queries total
- 0 NaN / Inf
- leases off; promotion off
