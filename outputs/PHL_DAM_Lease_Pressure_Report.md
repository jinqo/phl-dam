# PHL-DAM Lease-001 — predicting when memory will matter

## Answer

**Yes, in this controlled retention pilot.** A learned PHL lease predicts future-use horizons from causal context and preserves the right memories under 4× write pressure substantially better than static priority, randomized timing, FIFO, or random eviction.

The transported lease reaches **87.11% retention**, compared with **79.97%** for a parameter-identical static learned priority. It wins all six paired seeds and finishes only **0.11 percentage points below an oracle** that sees the true next-use time.

This is deliberately a retention-only result. Key/value storage and lookup are exact, so content-addressing errors cannot masquerade as eviction errors. The learned policy receives no future query metadata at evaluation.

## Frozen protocol

| Item | Setting |
|---|---:|
| Memory capacity | 8 slots |
| Writes per episode | 32 |
| Pressure | 4× slots |
| Future-use classes | near 5–9, medium 14–22, far 32–48, never |
| Evaluation | 2,000 episodes per seed; 322,420 total queries |
| Seeds | 0–5 |
| Predictor | 548 parameters |
| Predictor input | noisy causal context cue at write time |
| Future labels at evaluation | hidden |
| Promotion | off |

The predictor is trained on separate episodes to classify future-use horizons. Its distribution is expanded over delay support and transported forward with a fixed countdown operator. Eviction favors memories with greater remaining near-term relevance. The static control uses the exact same predictor and parameters but collapses its output to a time-invariant probability that the item will ever be queried.

## Results

| Retention policy | Mean hit rate ± sample SD |
|---|---:|
| Random eviction | 65.72% ± 0.22% |
| FIFO | 69.01% ± 0.25% |
| Randomized lease timing | 77.12% ± 0.66% |
| Static learned priority | 79.97% ± 0.39% |
| **PHL transported lease** | **87.11% ± 0.13%** |
| Oracle next use | 87.22% ± 0.13% |

PHL's paired advantages are:

| Comparison | Mean advantage | Approximate paired 95% interval | Seed wins |
|---|---:|---:|---:|
| PHL − static learned | **+7.14 pp** | [+6.72, +7.56] | 6/6 |
| PHL − randomized lease | **+9.99 pp** | [+9.31, +10.67] | 6/6 |
| PHL − FIFO | **+18.10 pp** | [+17.92, +18.29] | 6/6 |
| Oracle − PHL | +0.11 pp | [+0.10, +0.12] | 6/6 |

The delay predictor achieves **99.946% top-1 accuracy**, mean cross-entropy 0.00519, and expected calibration error 0.00394. That confirms the model can decode the synthetic causal cue and turn it into calibrated future-use timing.

## Gate audit

| Preregistered gate | Observed | Result |
|---|---:|---:|
| Delay prediction ≥85% | 99.95% | Pass |
| PHL over static ≥5 pp | +7.14 pp; 6/6 seeds | Pass |
| PHL over best non-oracle control ≥5 pp | +7.14 pp; 6/6 seeds | Pass |
| Oracle over best FIFO/random/randomized heuristic ≥10 pp | +10.10 pp aggregate; 4/6 individual seeds | Borderline pass |

The oracle feasibility gate clears its aggregate threshold by only 0.10 pp and does not clear it on every seed. This supports continuing the lease line, but not claiming a large oracle headroom.

## Interpretation

The comparison against static learned priority is the important attribution. Both policies know whether a memory is likely to matter and use the same predictor. Only PHL transports *when* that relevance becomes imminent. The +7.14 pp paired gain therefore isolates useful temporal ranking rather than extra parameters or better identification of never-used distractors.

The near-oracle result also means this particular cue-rich task is close to solved. It is evidence that the mechanism can work, not yet that it handles realistic uncertainty.

## Limits and next falsification

This pilot uses supervised delay-class training, synthetic cues, exact content storage, one-shot queries, and query-consumes-binding semantics. It does not establish end-to-end lease learning from predictive loss or performance on natural sequences.

The next experiment should make cues ambiguous or absent, add a persistent/unknown-delay lease state, and require the learned policy to avoid false precision. Only after that should transported leases be integrated into the full learned PHL-DAM slot controller.
