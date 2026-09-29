# Slot memory at 4× the context (fixed before running)

Round 2 (`PREREGISTRATION_LM_round2.md`, 128-byte sequences, window 16)
passed 4/4. Round 3 asks whether that holds when the context and the window
both grow 4×, and — new — whether the gain needs memory *beyond* the window.

## Dev findings since round 2 (seeds 100–101 only)

* Memory on top of **full** attention at 128 bytes: no gain (1.7631 vs
  1.7639; 1.7487 vs 1.7432). The benefit comes from window + memory, which
  beat full attention on both dev seeds.
* With a passkey in *every* training crop the memory still does not learn it
  (digit accuracy 0.097); write gate 0.094 at the stated digits vs 0.089
  elsewhere, read attention on the digits' content ~1%.
* Hard replacement writes (salience > 0.5 with a straight-through gradient,
  replacing one whole slot) made text modelling worse (1.78–2.23 vs 1.735)
  with and without the gate's gradient detached from the trunk, and did not
  learn the passkey. Not used.

The memory configuration is round 2's, unchanged.

## Arms (parameter-matched as in round 2, ~839K parameters)

* `window` — attention window 64 (stacked reach 252);
* `window_sml` — the same + SlotMemory in block 3 (16 slots × 64);
* `window_sml_local` — control: the same memory wiped every 64 positions, so
  it carries nothing beyond the attention window;
* `full` — full causal attention over 512 bytes.

## Protocol

Seeds 6, 7, 8 (never used). Sequence 512, batch 8 (the same 4,096 bytes per
step as rounds 1–2), 4,000 steps, lr 3e-3 (warmup + cosine), AdamW(0.9,
0.95, wd 0.1), clip 1.0, 25% passkey crops (key stated in the first 20
bytes, asked at the end, ~490 bytes later — beyond the window arms' reach),
1 CPU thread per run. No dev run at this length beyond a 20-step timing check.

## Criteria — "holds at 4× context" only if all hold

1. **Helps text.** Bits/byte of `window_sml` < `window` on 3/3 seeds.
2. **The gain uses memory beyond the window.** Bits/byte of `window_sml` <
   `window_sml_local` on 3/3 seeds.
3. **As good as full attention.** Mean bits/byte of `window_sml` ≤ mean of
   `full`.
4. **Affordable.** Counted forward+backward FLOPs of `window_sml` ≤ 1.10×
   `window` at this length, and its inference state (64-position KV window +
   memory) below `full`'s 512-position KV cache.

Passkey accuracy and CPU seconds per step are reported, not criteria. Every
criterion is reported pass or fail.
