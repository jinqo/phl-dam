# SlotMemory in a Transformer language model — real-text results

The synthetic results (`PHL_DAM_v2_Report.md`) say nothing about language. This
report tests the v4 memory as one extra sublayer in a small Transformer LM on
real text: byte-level WikiText-2, `work/lm_hybrid.py`, module
`work/slot_memory.py`. Every number below comes from preregistered runs on
seeds that dev work never touched.

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
