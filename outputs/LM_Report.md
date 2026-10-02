# SlotMemory in a Transformer language model — real-text results

The synthetic results (`PHL_DAM_v2_Report.md`) say nothing about language. This
report tests the v4 memory as one extra sublayer in a small Transformer LM on
real text: byte-level WikiText-2, `work/lm_hybrid.py`, module
`work/slot_memory.py`. Every number below comes from preregistered runs on
seeds that dev work never touched.

> **Update — rounds 3 and 4 (512 bytes):** the memory still beats window-only attention with a 16-byte window (round 4, 3/3 seeds), but **a control whose memory is wiped every 16 bytes matches it** (1.7524 vs 1.7510 mean). The gain is local capacity, not long-range memory. With a 64-byte window (round 3) the memory degenerates and does not help. See the round-4 and round-3 sections.

## Verdict — round 2 (`work/PREREGISTRATION_LM_round2.md`, seeds 3–5): worth using, **4/4 criteria pass**

Question: when you add one cheap recurrent sublayer to a windowed Transformer,
is SlotMemory a better choice than the standard one, a Mamba-style selective
SSM? All arms have ~839K parameters (matched within 0.03%). Training was 4,000
steps at batch 32 × 128 bytes, and the attention window is 16 (stacked reach
60). The memory and SSM sublayers sit in block 3 of 4.

| Arm | Bits/byte, seed 3 | Seed 4 | Seed 5 | Mean | FLOPs/step | CPU s/step |
|---|---:|---:|---:|---:|---:|---:|
| **window + SlotMemory** | **1.7380** | **1.7332** | **1.7445** | **1.7386** | 1.039× | 0.750 |
| full attention (128) | 1.7438 | 1.7497 | 1.7653 | 1.7529 | 1.000× | 0.443 |
| window only | 1.7651 | 1.7720 | 1.7694 | 1.7688 | 1.000× | 0.447 |
| window + selective SSM | 1.7748 | 1.7681 | 1.7681 | 1.7703 | 1.000× | 0.505 |

1. **Helps text** — below `window` on 3/3 seeds (−0.030 bits/byte mean): **pass**.
2. **Beats the standard alternative** — below `window_ssm` on 3/3 seeds
   (−0.032 mean): **pass**. The SSM sublayer, at the same parameter count, did
   no better than the window baseline.
3. **As good as a bigger attention span** — mean 1.7386 ≤ full attention's
   1.7529: **pass**.
4. **Affordable** — 1.039× counted forward+backward FLOPs (limit 1.10×); the
   inference state is 18,448 floats (16-byte KV window plus 16 slots × 129),
   against a 131,072-float KV cache for full attention: **pass**.

Six fresh seeds over two rounds give the same ordering. Memory is below
window-only on 6/6 seeds, with pooled means of 1.7424 vs 1.7653 bits/byte.
Memory is also below full attention on 5/6 seeds, with pooled means of
1.7424 vs 1.7539. The one exception is round-1 seed 2, at 1.7524 vs 1.7513.

## What this does *not* show — read before adopting

* **Scale.** 839K parameters, 4,000 steps, 128-byte sequences, bytes not
  tokens. The effect is small (≈1.7% lower bits/byte than window-only). Whether it
  survives at 150M parameters and 2k+ token contexts is untested. The
  prompt `IMPLEMENTATION_PROMPT_150M.md` is written to find out.
* **Not all of the gain is long-range memory.** In round 1, a control whose
  memory was wiped every 16 bytes (so it carries nothing beyond the attention
  window) recovered about half of the gain. Its mean was 1.7533, against
  1.7617 for window-only and 1.7461 for the full memory. The full memory
  beat the control on 3/3 seeds, but by 0.002–0.017. Part of the benefit is
  a cheap within-window mechanism, not recall from far away.
* **Passkey recall is not learned reliably.** A 5-digit key stated once, then
  asked for ~90–105 bytes later (beyond the 60-byte attention reach), was learned in only 1 of 6 memory runs
  (round-1 seed 1), and full attention also learned it in 2 of 6. In the LM
  the learned write gate is *anti*-selective: it writes less at the stated
  digits than at ordinary text. Five dev fixes did not make it reliable:
  a write-sparsity penalty, a surprise-gated write, values from token
  embeddings, 64 slots, and a copy auxiliary loss.
* **CPU wall-clock is 1.68× the baseline.** That figure is dominated by a
  Python loop of 128 small kernel launches on one thread; the SSM's loop costs
  1.13×. Round 1 judged cost by wall-clock and failed (1.69× against a 1.6
  limit). Round 2 switched to counted FLOPs, with the reason written in the
  preregistration before any round-2 run. At scale the scan needs a fused
  kernel (see the implementation prompt, §3).
* **The FLOP count covers matrix products only.** The elementwise work of both
  recurrences (sort, cumprod and softmax over 16 slots; the SSM's scan) is not
  counted. It is small (tens of kFLOPs per token against ~5 MFLOPs), but it is
  not zero.
* **Full attention costs the same FLOPs here** because the window is
  implemented as masked attention at length 128. The memory's cost advantage
  over full attention is the constant-size inference state, not training
  compute at this length.
* **The SSM baseline was not tuned** beyond one 1,000-step sanity run. A
  stronger SSM (larger state, several layers, Mamba's convolution) might do
  better. The claim is only: at equal parameters and placement, this SSM did not help
  and the memory did.

## Round 4 (`work/PREREGISTRATION_LM_round4.md`, seeds 9–11): 512 bytes, window 16 — **3/4; the gain is not long-range**

Round 3's failure was diagnosed on dev seeds as the 64-byte window driving
the write gate to 0.8–0.96. Round 4 kept round 2's 16-byte window at 512
bytes.

| Arm | Bits/byte, seed 9 | Seed 10 | Seed 11 | Mean |
|---|---:|---:|---:|---:|
| window only | 1.7749 | 1.7777 | 1.7732 | 1.7753 |
| window + SlotMemory | 1.7529 | **1.7443** | 1.7557 | **1.7510** |
| memory wiped every 16 (control) | **1.7513** | 1.7552 | **1.7507** | 1.7524 |
| full attention (512) | 1.8919 | 2.0468 | 1.9369 | 1.9585 |

1. Helps text (< window, 3/3): **pass** (−0.024 mean).
2. Gain needs memory beyond the window (< control, 3/3): **fail** — 1/3,
   means 1.7510 vs 1.7524.
3. As good as full attention: **pass** (full attention trains badly at 512
   bytes at this scale).
4. Affordable: **pass** (1.035× FLOPs; 18,448-float state vs 524,288).

**This is the decisive result of the LM study.** A memory that is wiped every
16 bytes — so it can hold nothing the attention window cannot already see —
gives the same gain as the full memory. At this scale the layer helps
because it adds cheap *local* capacity (an extra recurrent mixing path over
the last few positions), not because it carries information across long
distances. Round 1 already hinted at this (the control kept about half the
gain at 128 bytes); at 512 bytes it keeps all of it. Passkey recall stays at
chance for every arm except full attention on some seeds.

## Round 3 (`work/PREREGISTRATION_LM_round3.md`, seeds 6–8): does it hold at 4× the context? **No — 1/4**

Same models, context 512 bytes, window 64 (stacked reach 252), batch 8 (the
same bytes per step), 4,000 steps, lr 3e-3; a new control wipes the memory
every 64 bytes.

| Arm | Bits/byte, seed 6 | Seed 7 | Seed 8 | Mean |
|---|---:|---:|---:|---:|
| window only | 1.8676 | 1.8530 | 1.8510 | **1.8572** |
| window + SlotMemory | **1.8534** | 1.9075 | 2.5596 | 2.1068 |
| memory wiped every 64 (control) | 2.3151 | 2.4308 | 2.4574 | 2.4011 |
| full attention (512) | 1.8980 | 2.0299 | 1.8977 | 1.9419 |

1. Helps text (< window, 3/3): **fail** (1/3).
2. Gain needs memory beyond the window (< control, 3/3): **fail** as
   preregistered (2/3; the control itself trained badly, see below).
3. As good as full attention (mean): **fail**.
4. Affordable: **pass** (1.035× FLOPs; 67,600-float state vs a 524,288-float
   KV cache).

What happened: at 512 bytes the memory arms often train badly. Seed 8
stalls from the start (3.14 bits/byte at step 1,000 vs 2.49 for window-only)
and never recovers; every run of the reset control ends 0.45–0.6 worse
than window-only. The models do use the memory: switching it off at
evaluation is worse still (e.g. seed 6: 1.853 → 2.056), so the trunk has
co-adapted to a memory that helps less than attention alone would have.
Full attention was also unstable on one seed (2.03), and window-only was
the best arm, so at this scale and learning rate the longer context is hard
for every arm — but hardest for the memory. **The round-2 result does not
transfer to 4× the context as-is.** This is the main open problem before
adopting the layer at a 2k context; diagnosis is in progress (dev seeds).

## Round 1 (`work/PREREGISTRATION_LM_round1.md`, seeds 0–2): not worth using, 1/4

| Arm | Bits/byte, seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| window + SlotMemory | 1.7376 | 1.7484 | 1.7524 |
| window + memory wiped every 16 (control) | 1.7550 | 1.7503 | 1.7546 |
| full attention | 1.7615 | 1.7516 | 1.7513 |
| window only | 1.7649 | 1.7583 | 1.7618 |

Criterion 2 passed: memory was below window-only on 3/3 seeds. The rest
failed:

* **Criteria 1 and 3** used a rare-word long-range metric, bits/byte on
  characters 2+ of rare words whose previous occurrence lies beyond the
  60-byte reach. The metric proved invalid, because the window-only model
  scored best on it. It is kept in the outputs but was dropped as a
  criterion.
* **Criterion 4** failed on CPU step time: 1.69×, against a 1.6× limit.

## Reproduction

```
cd work
python3 -m unittest test_slot_memory      # causality for every arm, parameter matching, reach
python3 lm_hybrid.py --arm window_sml --seed 3 --steps 4000 --length 128 --window 16 \
    --lr 3e-3 --eval-every 500 --passkey-fraction 0.25 --threads 1 \
    --memory-layer 3 --compile-step --output ../outputs/lm_round2_window_sml_seed3.json
python3 lm_verdict.py --round 2           # -> outputs/lm_round2_verdict.json
```

The other arms are `window`, `window_ssm` (add `--memory-layer 3`) and `full`.
WikiText-2 is downloaded to `data_cache/` on first use.
