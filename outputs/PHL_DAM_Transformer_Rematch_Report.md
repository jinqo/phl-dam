# PHL-DAM vs causal Transformer — clean six-seed rematch

## Verdict

PHL-DAM wins the fixed 500-step rematch on every paired seed. At essentially identical parameter count, it achieves **97.42% mean recall** versus **47.08%** for the Transformer, while also producing lower all-token cross-entropy and using 97.30% less inference-time memory state at sequence length 176.

This is a result about learning speed and state efficiency on this synthetic associative-recall protocol. It is not evidence of universal superiority over Transformers.

## Matched contract

| Item | PHL-DAM | Causal Transformer |
|---|---:|---:|
| Trainable parameters | 33,034 | 33,074 |
| Parameter mismatch | — | +40 (+0.12%) |
| Seeds | 0–5 | 0–5 |
| Training budget | 500 steps, batch 16 | 500 steps, batch 16 |
| Evaluation | 2,000 episodes / 6,000 queries per seed | Same |
| Sequence/task streams | Frozen generator streams | Same |
| Primary predictive objective | All-token CE + marked recall-token CE | Same |
| Runtime state at length 176 | 456 floats | 16,896 KV-cache floats |
| Leases / promotion | Off / off | Off / off |

The Transformer is a one-layer pre-norm causal decoder with `d_model=48`, four attention heads, FFN width 195, fixed sinusoidal positions, and no dropout. A future-token mutation test verifies that logits at earlier positions do not change.

One asymmetry is disclosed rather than hidden: PHL-DAM retains its preregistered `0.05 ×` position-agnostic write-budget penalty. The Transformer has no write controller, so that architecture-specific term is inapplicable. The common predictive loss, data, seeds, optimizer family, learning rate, batch size, and update count are matched.

## Results

| Metric | PHL-DAM | Transformer | PHL-DAM difference |
|---|---:|---:|---:|
| Recall, mean ± sample SD | 97.42% ± 5.40% | 47.08% ± 4.47% | +50.34 pp |
| Recall range | 86.40–99.98% | 40.55–54.27% | — |
| Successful seeds at ≥20% recall | 6/6 | 6/6 | — |
| All-token CE | 0.2332 | 0.2520 | −0.0188 nats |
| Recall-token CE | 0.0767 | 1.2264 | −1.1497 nats |
| Memory-disabled recall | 10.09% | 8.88% | Near chance for both |
| Length-176 inference state | 456 floats | 16,896 floats | 97.30% less |

Paired recall differences (PHL-DAM minus Transformer) are +40.52, +45.35, +51.28, +53.65, +58.90, and +52.32 percentage points. Their mean is +50.34 pp; the approximate paired 95% t interval is [+43.53, +57.14] pp. With only six seeds, this interval is descriptive evidence rather than a substitute for broader replication.

### Recall by delay

| Write-to-query delay | PHL-DAM | Transformer |
|---|---:|---:|
| 29–63 | 97.32% | 40.28% |
| 64–95 | 97.47% | 46.32% |
| 96–169 | 97.47% | 54.63% |

The PHL-DAM advantage is not confined to a single distance bin. Disabling DAM retrieval drops its recall by 87.33 pp; disabling Transformer attention drops recall by 38.20 pp. Both ablations fall near the 10-value chance rate, confirming that the compared memory pathways carry the useful signal.

## Preregistered Gate 3 audit

The research design requires PHL-DAM to be within 0.01 nats of the best conventional model on all-token CE, within five percentage points on recall, and show at least one specified advantage.

| Gate | Observed | Result |
|---|---:|---:|
| All-token CE within 0.01 nats | PHL-DAM is better by 0.0188 | Pass |
| Recall within 5 pp | PHL-DAM is better by 50.34 pp | Pass |
| ≥5 pp recall advantage | +50.34 pp | Pass |
| ≥0.01-nat CE advantage | +0.0188 nats | Pass |
| ≥20% less runtime state at matched recall | Raw state is 97.30% less, but recall was not matched | Not separately tested |

**Overall Gate 3: pass.** The recall and CE advantages each independently satisfy the “one meaningful advantage” requirement. The state comparison is reported descriptively, not counted as a matched-recall gate. No threshold was moved after observing results.

## Reproduction notes

The PHL-DAM side uses the already-frozen deterministic seed 0–5 outputs from the preceding wider-seed confirmation; the Transformer side was run against the same generator and protocol. Reusing those immutable PHL-DAM outputs avoids selecting a more favorable rerun. All 12 raw run files, both model implementations, tests, aggregate, and this report are included in the reproduction archive.

No wall-clock speed conclusion is reported because the Transformer jobs were run concurrently and the PHL-DAM timings came from earlier execution conditions.
