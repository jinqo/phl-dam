# Slot memory at 512 bytes with a 16-byte window (fixed before running)

Round 3 (512 bytes, window 64, seeds 6–8) failed 1/4. Dev diagnosis since
(seeds 100–101 only):

* With a 64-byte window the memory's write gate climbs from 0.09 to 0.80–0.96
  by step 500 (it was 0.19 in a 128-byte, window-16 model), so the memory
  becomes a few-position recency cache that duplicates the attention window.
* A lower learning rate (1e-3) and truncated backprop through the scan
  (every 128 positions) did not fix it (write gate 0.96 and 0.99).
* With a **16-byte window** at 512 bytes the write gate stays low (0.04–0.27)
  and the memory helps on both dev seeds: 1.7515 / 1.7523 bits/byte vs
  1.7760 / 1.7822 for window-16 alone. Window-16 alone also beats window-64
  alone at this scale (1.776 vs 1.843, seed 100).

So round 4 tests round 2's configuration — window 16, unchanged memory — at
4× the context.

## Arms (parameter-matched, ~839K parameters)

* `window` — attention window 16 (stacked reach 60);
* `window_sml` — the same + SlotMemory in block 3 (16 slots × 64);
* `window_sml_local` — control: the same memory wiped every 16 positions;
* `full` — full causal attention over 512 bytes.

## Protocol

Seeds 9, 10, 11 (never used). Sequence 512, batch 8, 4,000 steps, lr 3e-3
(warmup + cosine), AdamW(0.9, 0.95, wd 0.1), clip 1.0, 25% passkey crops (key
asked ~490 bytes after it is stated), 1 CPU thread per run.

## Criteria — "holds at 4× context" only if all hold

1. **Helps text.** Bits/byte of `window_sml` < `window` on 3/3 seeds.
2. **The gain uses memory beyond the window.** Bits/byte of `window_sml` <
   `window_sml_local` on 3/3 seeds.
3. **As good as full attention.** Mean bits/byte of `window_sml` ≤ mean of
   `full`.
4. **Affordable.** Counted forward+backward FLOPs of `window_sml` ≤ 1.10×
   `window` at this length, and its inference state (16-position KV window +
   memory) below `full`'s 512-position KV cache.

Passkey accuracy and CPU seconds per step are reported, not criteria. Every
criterion is reported pass or fail.
