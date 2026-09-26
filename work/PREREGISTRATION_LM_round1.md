# Slot memory in a Transformer LM — "worth using" confirmation (fixed before running)

Question: is the SlotMemory layer worth adding to a Transformer language model?
Proxy on real text (WikiText-2 bytes), `work/lm_hybrid.py`.

Dev work (seeds 100–101 only) chose the configuration: memory in block 3 of
4 (dev rare-word bits/byte at step 3,500: layer 3 3.31, layer 2 3.37,
layer 1 3.43; 32 slots no better), 16 slots × 64 dims, compiled scan step
(1.49× the baseline's step time, down from 2.0×).

## Arms (parameter-matched, ~839K parameters)

* `window` — 4-layer Transformer, attention window 16 bytes (stacked reach 60);
* `window_sml` — the same + SlotMemory in block 3, copy readout, FFN of that
  block shrunk to match parameters;
* `window_sml_local` — control: the same memory wiped every 16 positions, so
  it carries nothing beyond the attention window;
* `full` — full causal attention over the 128-byte sequence.

## Protocol

Seeds 0, 1, 2 (never used). 4,000 steps, batch 32, sequence 128, lr 3e-3
(warmup + cosine), AdamW(0.9, 0.95, wd 0.1), clip 1.0, 25% passkey crops in
training for every arm. Evaluation: 96 natural crops of WikiText-2 valid
(bits/byte); rare-word long-range repeats from valid + test (152 crops, 842
positions: characters 2+ of a word occurring ≤ 5 times in train, whose previous
occurrence is beyond the 60-byte reach and not within it); 192 passkey crops.

## Criteria — the memory is "worth using" only if all hold

1. **Long-range benefit.** Rare-word bits/byte of `window_sml` lower than
   `window` on 3/3 seeds and lower than `window_sml_local` on 3/3 seeds.
2. **No harm to text.** Overall bits/byte of `window_sml` ≤ `window` on 3/3
   seeds.
3. **Matches a bigger context on long-range recall.** Mean rare-word bits/byte
   of `window_sml` ≤ that of `full`.
4. **Affordable.** Round-robin step time of `window_sml` ≤ 1.6× `window`, and
   its inference state (KV window + memory) below `full`'s KV cache.

Passkey accuracy is reported but is not a criterion: no arm learned it in dev.
Every criterion is reported pass or fail.
