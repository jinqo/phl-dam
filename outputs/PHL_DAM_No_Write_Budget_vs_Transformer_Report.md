# PHL-DAM without write-budget penalty vs causal Transformer

## Verdict

The PHL-DAM advantage survives complete removal of the write-budget penalty. With the coefficient set to exactly zero, PHL-DAM achieves **99.55% mean recall** versus **47.08%** for the parameter-matched causal Transformer and wins all six paired seeds.

The penalty is therefore not responsible for PHL-DAM's advantage in this experiment. Its measurable role is policy shaping: without it, the controller writes more often outside binding locations, but still learns the task.

## Controlled intervention

Exactly one training term changed:

```text
original PHL-DAM loss = all-token CE + recall-token CE + 0.05 × write-budget penalty
ablated PHL-DAM loss  = all-token CE + recall-token CE
```

The architecture, 33,034 parameters, eight slots, key/value dimensions, streams, seeds 0–5, optimizer, learning rate, 500 updates, batch size 16, and 2,000 evaluation episodes were preserved. Leases and promotion remained off. The Transformer is the same frozen, verified 33,074-parameter causal baseline from the clean rematch.

## Main results

| Metric | PHL-DAM, penalty = 0 | Transformer | Difference |
|---|---:|---:|---:|
| Parameters | 33,034 | 33,074 | −40 |
| Recall, mean ± sample SD | **99.55% ± 1.07%** | 47.08% ± 4.47% | **+52.47 pp** |
| Recall range | 97.37–100% | 40.55–54.27% | — |
| Paired seed wins | **6/6** | 0/6 | — |
| All-token CE | **0.2337** | 0.2520 | **0.0183 nats better** |
| Recall-token CE | **0.0158** | 1.2264 | **1.2106 nats better** |
| Memory-disabled recall | 9.99% | 8.88% | Both near chance |
| Length-176 inference state | 456 floats | 16,896 floats | 97.30% less raw state |

The paired recall advantages are +51.48, +45.73, +51.52, +54.32, +59.45, and +52.33 percentage points. The mean is +52.47 pp, with an approximate paired 95% t interval of [+47.79, +57.15] pp.

### Recall by delay

| Write-to-query delay | PHL-DAM, penalty = 0 | Transformer |
|---|---:|---:|
| 29–63 | 99.49% | 40.28% |
| 64–95 | 99.59% | 46.32% |
| 96–169 | 99.58% | 54.63% |

Disabling DAM retrieval reduces zero-penalty PHL-DAM from 99.55% to 9.99%, confirming that the memory path—not a shortcut—carries the result.

## What removing the penalty changed

Compared with the earlier regularized PHL-DAM runs:

| Metric | Penalty 0.05 | Penalty 0 |
|---|---:|---:|
| Mean recall | 97.42% | 99.55% |
| Mean all-token CE | 0.2332 | 0.2337 |
| Mean recall-token CE | 0.0767 | 0.0158 |
| Mean write gate away from bindings | 0.12% | 4.22% |

The recall point estimate improves by 2.14 pp, largely because seed 0 rises from 86.40% to 97.37%. This should **not** be interpreted as proof that removing the penalty improves recall: the approximate paired interval for that change is [−2.41, +6.68] pp with only six seeds.

The unambiguous effect is reduced write sparsity. Average off-binding write strength grows about 34-fold. Some successful seeds also show large final minibatch write-budget diagnostics, yet attain essentially perfect recall. Thus the penalty enforces a cleaner write policy but is not necessary for content-memory learning under this protocol.

## Gate 3 audit

| Requirement | Observation | Result |
|---|---:|---:|
| All-token CE within 0.01 nats of conventional baseline | Better by 0.0183 nats | Pass |
| Recall within 5 pp | Better by 52.47 pp | Pass |
| At least one meaningful advantage | Recall and CE advantages both qualify | Pass |

**Overall Gate 3: pass.** The 97.30% raw state reduction is reported descriptively, not counted as a matched-recall result.

## Reproduction notes

All six PHL-DAM ablation runs were newly trained with `--write-budget-weight 0`. The Transformer outputs are the frozen deterministic seed 0–5 results from the immediately preceding clean rematch, preventing favorable rerun selection. All raw files report finite models, the zero coefficient, identical budgets, and leases/promotion disabled.

No wall-clock comparison is made because the PHL-DAM jobs ran concurrently and the Transformer jobs were executed under different scheduling conditions.
