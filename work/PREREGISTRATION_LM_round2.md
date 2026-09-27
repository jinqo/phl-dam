# Slot memory vs a selective SSM layer in a Transformer LM (fixed before running)

Round 1 (`PREREGISTRATION_LM_round1.md`, seeds 0–2) found the memory lowers
bits/byte on every seed but failed "worth using" on two grounds: the rare-word
long-range metric turned out invalid (the window-only model scores best on
it), and CPU wall-clock at 1.69× the baseline step time.

Round 2 asks the question an architect actually faces: **if I add one cheap
recurrent sublayer to a windowed Transformer, is SlotMemory a better choice
than the standard one, a Mamba-style selective SSM?**

## Arms (parameter-matched, ~839K parameters, `work/lm_hybrid.py`)

* `window` — 4-layer Transformer, attention window 16 (stacked reach 60);
* `window_sml` — the same + SlotMemory in block 3 (16 slots × 64, copy readout,
  compiled step), FFN of that block shrunk to match parameters;
* `window_ssm` — the same + a selective SSM sublayer in block 3 (64 channels,
  Mamba discretisation h ← exp(−Δ·A)h + Δ·x, A init 1..16, Δ init
  log-uniform [1e-3, 1e-1], SiLU output gate), FFN shrunk to match;
* `full` — full causal attention over the 128-byte sequence.

The SSM was not tuned beyond one sanity run (dev seed 100, 1,000 steps). The
memory configuration is round 1's, unchanged; none of the dev variants tried
since (write penalty, surprise gate, values from embeddings, 64 slots, copy
auxiliary loss) is used, because none helped in dev.

## Protocol

Seeds 3, 4, 5 (never used). Otherwise identical to round 1: 4,000 steps,
batch 32, sequence 128, lr 3e-3 (warmup + cosine), AdamW(0.9, 0.95, wd 0.1),
clip 1.0, 25% passkey crops in training, 1 CPU thread per run. Evaluation:
bits/byte on 96 natural crops of WikiText-2 valid; 192 passkey crops.

## Criteria — the memory is "worth using" only if all hold

1. **Helps text.** Bits/byte of `window_sml` < `window` on 3/3 seeds.
2. **Beats the standard alternative.** Bits/byte of `window_sml` <
   `window_ssm` on 3/3 seeds.
3. **As good as a bigger attention span.** Mean bits/byte of `window_sml` ≤
   mean of `full`.
4. **Affordable.** Counted forward+backward FLOPs of `window_sml` ≤ 1.10×
   `window` (measured before this round: 1.039×), and its inference state
   (KV window + memory) below `full`'s KV cache.

### Change of cost criterion, and why

Round 1 judged cost by CPU wall-clock. That measures a Python loop of 128
small kernel launches per sequence on one CPU thread, not the architecture:
the same recurrence is a single fused scan on an accelerator, and the SSM arm
runs the same kind of loop. FLOPs are hardware-independent. CPU seconds per
step are still recorded in every output and reported next to the verdict.

The rare-word metric is dropped as a criterion (shown invalid in round 1); it
is still reported. Passkey accuracy is reported, not a criterion (no arm
learns it reliably in 4,000 steps). Every criterion is reported pass or fail.
