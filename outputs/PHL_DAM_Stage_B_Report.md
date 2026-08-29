# PHL-DAM Stage B — Learned One-Seed Pilot

**Verdict: PASS**

The learned-controller, content-only pilot reached **86.40% held-out recall**,
exceeding the preregistered Stage B gate of `recall >= 20%`.

## Test boundary

- seed 0 only
- 3 randomized key/value bindings per sequence
- 10 possible values and 32 possible keys
- 8 soft-allocated slots, `d_key=24`, `d_value=24`
- simplified PHL backbone: `d_model=64`, 4 horizons × 16 dimensions
- sequence length 176; recall delays 29–169 tokens
- WRITE and QUERY markers are ordinary input tokens
- no WRITE/QUERY positions, slot labels, or controller targets passed to the model
- all-token next-token CE + marked recall-token CE
- after an observed filler-write failure, a `0.05` position-agnostic penalty
  encourages total write mass to equal the known three-binding capacity; it
  does not reveal binding positions
- learned soft allocation; hard allocation disabled
- **no temporal lease state, lease score, or promotion path**
- 500 training steps, batch size 16, learning rate 0.002
- 2,000 held-out episodes / 6,000 recall queries
- PyTorch 2.13.0+cpu, Python 3.14.0, CPU execution

The local binding path factors the causal `WRITE key value` roles before the
learned projections: the preceding token supplies the candidate key and the
current token supplies the candidate value. The write gate still learns
whether that local context is a binding.

## Results

| Metric | Result |
|---|---:|
| Recall accuracy | **86.40%** |
| Stage B gate | **PASS (≥20%)** |
| Retrieval-disabled accuracy | 10.18% |
| Gain attributable to retrieval | +76.22 pp |
| Correct-address top-1 | 92.47% |
| All-token CE | 0.2474 nats |
| Recall-token CE | 0.3748 nats |
| NaN / Inf | 0 |

Recall by distance:

| Delay | Queries | Recall |
|---|---:|---:|
| 29–63 | 2,000 | 85.70% |
| 64–95 | 2,000 | 86.90% |
| 96–169 | 2,000 | 86.60% |

Controller diagnostics:

| Diagnostic | Binding/query | Elsewhere |
|---|---:|---:|
| Mean write gate | 0.99834 | 0.000136 |
| Mean read gate | 0.99924 | 0.05940 |

The retrieval-disabled control is at the ten-value chance rate. The recall
gain therefore comes from the DAM read path rather than the PHL backbone or a
stable key-to-value shortcut.

## Learning behavior

Learning was sharply delayed. Recall CE stayed near chance through step 400,
fell to 1.343 at step 450, and reached 0.458 at step 500. This late phase
transition is a seed-stability risk and is the main reason not to infer Stage C
success from this pilot.

Two bounded controller fixes followed documented probe failures:

1. The read gate was initialized open enough to expose the memory path to
   early recall gradients.
2. Candidate key/value roles were locally factored and the small write-budget
   penalty was added after the controller spread writes across filler.

No temporal mechanism was added.

## Verification

Six focused tests cover the sequence protocol and delay bins, token-only model
input, finite forward state, complete absence of lease parameters/state,
nonzero gradients through the learned controller/address/readout path, and the
retrieval ablation. The source also passes `compileall`.

## Interpretation and next gate

Stage B supports proceeding to the preregistered three-seed screen. It does not
establish seed robustness, DAM-only versus PHL-DAM attribution, or a temporal
lease advantage. In particular, leases should remain disabled until the
content-only three-seed result satisfies Stage C.

## Reproduce

From the extracted bundle:

```powershell
python -m unittest discover -s . -p 'test_*.py' -v
python .\phl_dam_stage_b.py --seed 0 --steps 500 --batch-size 16 --eval-episodes 2000 --output .\phl_dam_stage_b_results.json
```

The exact metrics and training history are in
`phl_dam_stage_b_results.json`.
