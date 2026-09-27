# Prompt: add the SlotMemory layer (PHL-DAM v4) to a ~150M-parameter model

Paste everything below the line into Claude Code, run inside the repository of
the 150M model. It is self-contained. The complete, tested reference code is
in section 3. The source study is `jinqo/phl-dam` (`outputs/LM_Report.md`,
`outputs/PHL_DAM_v2_Report.md`, `work/slot_memory_reference.py`); read it if
you can reach it, but you do not need it.

---

You are adding an external slot-memory layer ("SlotMemory") to my
~150M-parameter language model. Test whether it helps, and report honestly
either way. Work in stages, and stop at each point marked **STOP** to show me
what you found before continuing. Do not skip the stops.

## 1. What the layer is, in one paragraph

SlotMemory is a residual sublayer that carries a small recurrent state per
sequence: `N` slots, each holding a key, a value and an occupancy. At every
position, left to right, it **writes then reads**.

* **Write.** The candidate key comes from the *previous* position's hidden
  state and the value from the *current* one, so it stores "what followed
  this context". A learned write gate chooses how much to store, and a
  DNC-style address chooses where: the least-used slot, or a slot with a
  similar key.
* **Read.** It queries with the current position through the *same* key
  projection, and reads occupancy-normalised values.
* **Output.** The result goes back into the residual stream, and also
  straight to the LM head as a copy readout, which scores the read against
  every vocabulary item's value projection.

The state is constant-size (e.g. 16 slots × 129 floats), however long the
context.

## 2. What the evidence does and does not say — read before designing

It is a hypothesis for my model, not a known result. The source study is
small and preregistered, with fresh seeds for every confirmation.

**Synthetic recall** (~30K parameters, 8 slots): store 3 key→value bindings,
then retrieve them 29–169 tokens later.

* It reached 100% recall in 12 training steps. A parameter-matched
  Transformer reached 47.6% and a selective SSM 27.9%.
* Under write pressure (8–32 bindings into 8 slots) it had the highest recall
  at every level.
* Most of that came from **generic** mechanisms, not from anything unique:
  the copy readout and DNC write addressing. A DNC given the same copy
  readout came close.

**Real text** (`outputs/LM_Report.md`): one sublayer in block 3 of a 4-layer,
~839K-parameter byte-level Transformer. The data was WikiText-2 with an
attention window of 16 (stacked reach 60), 128-byte sequences and 4,000
steps. All arms were parameter-matched, and the result was the preregistered
round 2, seeds 3–5.

| Arm | Bits/byte (mean of 3) | FLOPs | Inference state (floats) |
|---|---:|---:|---:|
| window + SlotMemory | **1.7386** (best on 3/3) | 1.039× | 18,448 |
| full attention over 128 | 1.7529 | 1.000× | 131,072 |
| window only | 1.7688 | 1.000× | 16,384 |
| window + Mamba-style selective SSM | 1.7703 | 1.000× | 16,448 |

Across 6 fresh seeds (two rounds), it beat window-only 6/6 and full
attention 5/6.

**Caveats that apply to you:**

1. **Small effect, tiny scale.** The gain is ≈1.3–1.7% bits/byte, at 839K
   parameters and 128 bytes. It may shrink or vanish at 150M with a 2k
   context and full attention.
2. **Not all long-range.** A control whose memory is wiped every 16 bytes
   kept about half the gain. The layer is partly just a cheap extra
   within-window mechanism.
3. **It does not reliably learn to store a stated fact.** Passkey recall (a
   5-digit key stated once, asked ~100 bytes later) was learned in 1/6
   memory runs, versus 2/6 for full attention. The cause is the learned
   write gate: it is *anti*-selective, writing less at the stated digits
   than at ordinary text, so the key is diluted to <0.5% of any slot. Five
   fixes were tried in dev and none worked (section 3.3).
4. **Wall-clock.** A Python loop over positions cost 1.68× the step time on
   CPU even though FLOPs were +3.9%. At scale the scan needs a fused kernel
   (section 4.4).
5. **Weak comparison.** The SSM baseline was untuned.

What it plausibly buys a 150M model:

* recall beyond the attention window with a constant-size state;
* a windowed-attention + memory model that matches full attention at a
  fraction of the KV cache;
* possibly a small perplexity gain.

Design the evaluation to detect those, and to detect harm.

## 3. The module — implement exactly this

### 3.1 Reference code (tested; port it, do not redesign it)

This is `work/slot_memory_reference.py` from the study. It matches, bit for
bit, the module used in every LM run, and its `step()` matches the parallel
forward (both are unit-tested there).

```python
"""SlotMemory reference implementation (the code embedded in IMPLEMENTATION_PROMPT_150M.md).

Same maths and parameter names as work/slot_memory.py (the module every LM run
used), without the dev-only options that did not help (surprise-gated write,
values from token embeddings), plus a streaming API for generation.
test_slot_memory.py checks it matches slot_memory.py exactly and that
step-by-step decoding matches the parallel forward.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def unit(x: Tensor, eps: float = 0.05) -> Tensor:
    """Bounded unit-normalisation. F.normalize on a zero vector has Jacobian ~1e12."""
    return x / (x.pow(2).sum(-1, keepdim=True) + eps * eps).sqrt()


def memory_step(keys: Tensor, values: Tensor, occ: Tensor, k: Tensor, v: Tensor,
                write_gate: Tensor, alloc_gate: Tensor, sharpness: Tensor, q: Tensor,
                read_temperature: float, floor: float):
    """One write-then-read step of the slot recurrence. All tensors fp32.

    keys [B,N,dk], values [B,N,dv], occ [B,N]; k, q [B,dk]; v [B,dv];
    write_gate, alloc_gate, sharpness [B].
    """
    # DNC allocation: least-occupied slots first (sort indices carry no gradient)
    sorted_occ, order = occ.sort(dim=-1)
    exclusive = torch.cat([torch.ones_like(sorted_occ[:, :1]),
                           sorted_occ[:, :-1]], dim=-1).cumprod(dim=-1)
    alloc = torch.zeros_like(occ).scatter(1, order, (1.0 - sorted_occ) * exclusive)
    content = torch.softmax(sharpness[:, None] * torch.einsum("bd,bnd->bn", k, keys), -1)
    g = alloc_gate[:, None]
    write = write_gate[:, None] * (g * alloc + (1.0 - g) * content)
    remain = 1.0 - write
    keys = unit(remain[..., None] * keys + write[..., None] * k[:, None])
    values = remain[..., None] * values + write[..., None] * v[:, None]
    occ = remain * occ + write
    # read after write; occupancy prior; values normalised by occupancy
    score = torch.einsum("bd,bnd->bn", q, keys) / read_temperature
    attn = torch.softmax(score + 0.25 * torch.log(occ + 1e-6), dim=-1)
    retrieved = torch.einsum("bn,bnv->bv", attn, values / occ.clamp_min(floor)[..., None])
    entropy = -(attn * attn.clamp_min(1e-12).log()).sum(-1)
    return keys, values, occ, retrieved, entropy


class SlotMemory(nn.Module):
    """External slot memory as a residual sublayer.

    forward(h) -> (delta, m): add `delta` to the residual stream; pass `m` to
    copy_logits() at the LM head. `h` is the pre-normed hidden state at the
    insertion point, [B, T, d_model].
    """

    def __init__(self, d_model: int, num_slots: int = 16, d_key: int = 64,
                 d_value: int = 64, read_temperature: float = 0.1,
                 write_gate_bias: float = -3.0, read_gate_bias: float = 1.0,
                 copy_scale: float = 2.0, occupancy_floor: float = 1e-3,
                 compile_step: bool = False) -> None:
        super().__init__()
        self.num_slots, self.d_key, self.d_value = num_slots, d_key, d_value
        self.read_temperature, self.occupancy_floor = read_temperature, occupancy_floor
        self.key = nn.Linear(d_model, d_key, bias=False)        # tied key/query
        self.value = nn.Linear(d_model, d_value, bias=False)
        self.gates = nn.Linear(d_model, 3)                       # write, alloc, sharpness
        self.read_gate = nn.Linear(d_model + d_value + 1, 1)
        self.out = nn.Linear(d_value, d_model, bias=False)
        self.copy_scale = nn.Parameter(torch.tensor(float(copy_scale)))
        self.write_gate_bias, self.read_gate_bias = write_gate_bias, read_gate_bias
        self.step_fn = torch.compile(memory_step, dynamic=False) if compile_step else memory_step
        self.reset_gate_biases()

    def reset_gate_biases(self) -> None:
        """Call AFTER any model-wide init loop - a generic init zeroes these biases."""
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([self.write_gate_bias, 0.0, 0.0]))
            self.read_gate.bias.fill_(self.read_gate_bias)

    def state_floats(self) -> int:
        return self.num_slots * (self.d_key + self.d_value + 1)

    # -- projections, parallel over positions ---------------------------------
    def _project(self, h: Tensor, previous: Tensor):
        cand_keys = unit(self.key(previous)).float()             # key from h_{t-1}
        queries = unit(self.key(h)).float()                      # query from h_t (tied W_k)
        cand_values = self.value(h).float()                      # value from h_t
        gates = self.gates(h).float()
        return (cand_keys, queries, cand_values, torch.sigmoid(gates[..., 0]),
                torch.sigmoid(gates[..., 1]), F.softplus(gates[..., 2]) + 1.0)

    def _read_out(self, h: Tensor, r: Tensor, entropy: Tensor) -> tuple[Tensor, Tensor]:
        r = r.to(h.dtype)
        conf = (1.0 - entropy / math.log(self.num_slots)).to(h.dtype)
        rho = torch.sigmoid(self.read_gate(torch.cat([h, r, conf[..., None]], -1)))
        m = rho * r
        return self.out(m), m

    def init_state(self, batch: int, device=None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(keys, values, occupancy, previous hidden) - all zero at a document start."""
        z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)
        return (z(batch, self.num_slots, self.d_key), z(batch, self.num_slots, self.d_value),
                z(batch, self.num_slots), None)

    # -- training / prefill ----------------------------------------------------
    def forward(self, h: Tensor, reset: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """reset: bool [B, T], True where a new document starts (state zeroed before t)."""
        batch, length, _ = h.shape
        previous = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        if reset is not None:                    # no key taken from across a boundary
            previous = previous.masked_fill(reset[..., None], 0.0)
        k, q, v, w, g, beta = self._project(h, previous)
        keys, values, occ, _ = self.init_state(batch, h.device)
        retrieved, entropy = [], []
        for t in range(length):
            if reset is not None:
                keep = (~reset[:, t]).float()
                keys, values = keys * keep[:, None, None], values * keep[:, None, None]
                occ = occ * keep[:, None]
            keys, values, occ, r_t, e_t = self.step_fn(
                keys, values, occ, k[:, t], v[:, t], w[:, t], g[:, t], beta[:, t], q[:, t],
                self.read_temperature, self.occupancy_floor)
            retrieved.append(r_t)
            entropy.append(e_t)
        return self._read_out(h, torch.stack(retrieved, 1), torch.stack(entropy, 1))

    # -- generation: one position at a time, constant-size state --------------
    def step(self, h_t: Tensor, state) -> tuple[Tensor, Tensor, tuple]:
        """h_t [B, d_model] -> (delta [B, d_model], m [B, d_value], new state)."""
        keys, values, occ, previous = state
        if previous is None:
            previous = torch.zeros_like(h_t)
        k, q, v, w, g, beta = self._project(h_t, previous)
        keys, values, occ, r, e = memory_step(keys, values, occ, k, v, w, g, beta, q,
                                              self.read_temperature, self.occupancy_floor)
        delta, m = self._read_out(h_t, r, e)
        return delta, m, (keys, values, occ, h_t)

    def copy_logits(self, m: Tensor, embedding: Tensor) -> Tensor:
        """Pointer-style readout: m scored against every vocabulary item's value, W_v E."""
        return self.copy_scale * m @ self.value(embedding).T
```

The maths, per position t (all state fp32):

```
k_t = unit(W_k h_{t-1})   q_t = unit(W_k h_t)   v_t = W_v h_t          # W_k tied
w_t = sigmoid(.)  (bias -3)   g_t = sigmoid(.)   beta_t = softplus(.) + 1
alloc   = DNC allocation over ascending occupancy
content = softmax(beta_t * K k_t)
write   = w_t * (g_t * alloc + (1 - g_t) * content)                       # [N]
K <- unit((1-write) K + write k_t);  U <- (1-write) U + write v_t;  o <- (1-write) o + write
a_t = softmax(K q_t / 0.1 + 0.25 log(o + 1e-6))
r_t = sum_n a_t[n] U[n] / max(o[n], 1e-3)
m_t = sigmoid(W_r [h_t, r_t, 1 - H(a_t)/log N] + b_r) * r_t              # b_r = +1
residual += W_o m_t;   logits += s * m_t (W_v E)^T                       # s init 2.0
```

### 3.2 Why each piece is there (each was measured; keep all of them)

* **Occupancy-normalised values** (`U / o`). With a barely-open write gate,
  raw values are tiny. The gate then collapsed and starved itself of
  gradient.
* **Copy readout and tied `W_k`.** Together they remove two waits at
  initialisation: the head learning to decode retrievals, and key/query
  alignment. Time to learn recall halved.
* **DNC write addressing.** The original allocation smeared overflow writes
  across all slots, so the gate learned to under-write and 21% of needed
  items were never stored.
* **Bounded `unit()`**, not `F.normalize`. See section 5.
* **Write-gate bias −3 and read-gate bias +1.** A model-wide init loop will
  overwrite these, so call `reset_gate_biases()` after it. This bug happened
  in the study.

### 3.3 Tried and not adopted — do not start with these

Keep each behind a flag, default off, for ablations only:

* **Address-based erase gate**, `keep = (1-write)(1 - e_t·address)`. It
  looked good on 2 dev seeds, then failed its preregistered confirmation.
* **Write-sparsity penalty.** The gate shut everywhere instead of becoming
  selective.
* **Surprise-gated write**, adding the model's own next-token loss to the
  write logit. The gate stayed anti-selective, and the learned weight
  shrank.
* **Values from token embeddings** instead of hidden states. No gain.
* **64 slots instead of 16.** No gain.
* **Auxiliary copy loss on long-range repeats.** Passkey was learned on 1
  of 3 dev seeds, and bits/byte got worse by 0.04–0.05.

## 4. Integration at ~150M

### 4.1 Placement and sizes

* **One SlotMemory sublayer**, in one block at ≈ 3/4 depth (block
  `round(0.75·L)`, 0-indexed from the input). In the study, block 3 of 4 beat
  blocks 1 and 2.
* Insert it **after the attention sublayer and before the FFN**, pre-normed
  with its own norm like the other sublayers:

  ```python
  x = x + attn(norm1(x))
  delta, m = memory(norm_m(x), reset=doc_start)   # only in the chosen block
  x = x + delta
  x = x + ffn(norm2(x))
  ```

* **Sizes to start:** `num_slots=16`, `d_key=d_value=64`. Later, and only as
  a measured ablation, try 32–64 slots, several independent memory heads
  (separate module instances whose `delta`s are summed and `m`s
  concatenated for the copy readout), or 2 memory layers.

### 4.2 Copy readout at the head

Carry `m` from the memory block to the head and add
`memory.copy_logits(m, E)` to the logits. `E` is the token-embedding
matrix, or the input embedding if the head is untied.

At 150M the vocabulary is large (e.g. 50k), so `value(E)` is a `[V, 64]`
projection computed once per forward pass. The extra logits cost
`B·T·64·V` multiply-adds, about the same as a 64-wide slice of the LM head.
If the head uses a fused or chunked cross-entropy, fold the copy term into
it rather than materialising a second `[B, T, V]` tensor.

### 4.3 Parameter matching

The layer adds `d·(dk + 2·dv) + 4·d + dv + 6` parameters (150,598 at
d=768 with dk = dv = 64). Remove the same count from the FFN of the
block that holds it:

```python
ffn_hidden_new = ffn_hidden - round(extra / (2 * d_model + 1))
```

Use `(2·d + 1)` for a biased GELU FFN; for SwiGLU use `3·d`. Assert that the
two models' counts match within 0.1%, and report both.

### 4.4 Speed: the scan is sequential over T

In order:

1. **Keep every projection outside the loop.** The reference already does
   this: only `memory_step` runs per position.
2. **`torch.compile` the step** (`compile_step=True`). This was about 2×
   faster on CPU in the study.
3. **On GPU at T = 2k, write a fused scan kernel** in Triton or CUDA:
   * one program per (batch, head), holding K, U and o (N×129 floats) in
     registers or SRAM for all T positions;
   * the forward saves per-position inputs, not states;
   * the backward recomputes the states in chunks (e.g. 64 positions), or
     stores checkpoints every C positions.

   Verify the kernel against `memory_step` to 1e-5 in forward and backward
   before using it.

Measure tokens/sec for the unmodified model and for the model with memory,
before and after each step. Report the overhead honestly. Also report counted
FLOPs: `torch.utils.flop_counter.FlopCounterMode` counts matrix products
only, so count the scan's elementwise operations by hand. They are
O(N·(dk+dv)) per token, small next to the matrix products, but not zero.

### 4.5 Precision, documents, generation

* **Precision.** Keep K, U, o, the divisions and the logs in fp32, even if the
  model runs in bf16. The reference casts the projections to fp32.
* **Documents.** Pass `reset` (True at each document's first token) whenever
  documents are packed into one sequence. A slot must never carry anything
  across documents, and no key is taken from the previous document's last
  token.
* **Generation.**
  * Prefill with `forward()`, keeping the final state; or run `step()` over
    the prompt, which is simpler and exact.
  * Then call `step()` once per new token, with the state carried alongside
    the KV cache.
  * The memory state is constant-size, so with windowed attention the whole
    inference state is bounded.

## 5. Numerical stability — non-negotiable

Each of these produced gradient spikes of 10³–10¹² in the study.

1. **`F.normalize` on an exactly zero key** (position 0, a reset, a padded
   token) has a Jacobian of ~1e12. Use `unit()` with eps 0.05 for keys,
   candidate keys and queries.
2. **Division by occupancy.** Floor it at 1e-3. Raise the floor to 0.05 if you
   add anything that creates low-occupancy slots.
3. **`log(o + 1e-6)` in the read prior.** Its derivative is 1/o. Raise the
   epsilon if spikes appear there.
4. **Gradient clipping.** Keep it on, and log the global grad norm every
   step. On any spike above 1e3, hook `grad_in`/`grad_out` per op to find
   which op amplified it before changing anything else.
5. **Initialisation.** The model's init loop runs over all Linear layers, so
   call `memory.reset_gate_biases()` after it.

## 6. Tests to write before any training run

* **Causality.** Changing any token after position t leaves all logits at
  positions ≤ t bit-identical, with the copy readout on and memory in the
  chosen block.
* **Off-switch.** With the memory disabled (`delta = 0`, copy term 0), the
  output equals the model without it, given the same weights elsewhere.
* **Document reset.** Two documents packed with `reset` give the second one
  the same memory outputs as running it alone.
* **Streaming equals parallel.** `step()` over a sequence matches
  `forward()` to 1e-5. So does your fused kernel, forward and backward.
* **Retrieval at initialisation.** Write one binding with the gate forced
  open, then query its key. The copy logits must rank the stored token first.
* **Finite gradients** over a full-length sequence from a zero state, in the
  training precision.
* **Accounting.** Parameter counts and state floats are printed and asserted.

## 7. Inspect my model first — **STOP after this step**

Read the codebase and report back, without changing anything:

1. **Architecture:** `d_model`, layers, heads, FFN type and width, norm type
   and placement, positional scheme, vocabulary size, tied embeddings,
   attention type (full, windowed, or other) and context length.
2. **Data:** how documents are packed, and where boundaries are marked.
3. **Training loop:** optimizer, LR schedule, clipping, precision,
   distributed setup, `torch.compile`, current tokens/sec and FLOPs per
   token.
4. **Evaluation:** existing evaluations, and where long-context or
   retrieval evaluations can go.
5. **Generation:** the inference path, if any: KV cache handling and
   sampling loop.

Then propose:

* the insertion block;
* sizes;
* the parameter-matching plan;
* the expected parameter, FLOP and throughput cost;
* whether to use a windowed-attention variant for the main comparison
  (section 8).

Wait for my go-ahead.

## 8. Experiments — preregister, then run

Before any comparison run, write a preregistration file containing:

* arms;
* seeds (dev seeds for tuning, fresh seeds for confirmation);
* data and token budget;
* metrics;
* the exact pass/fail rule for each criterion.

Commit it before running. Tune only on dev seeds. Report every criterion as
pass or fail, including the ones that fail. If a criterion turns out to be
flawed, say so and fix it in the *next* preregistration, never
retroactively.

### Stage A — reproduce the study on my stack (small, cheap). **STOP after.**

Build a ~10–20M-parameter version of my model with my tokenizer and data.
Use 4 parameter-matched arms, the same token budget, and ≥3 fresh seeds:

* `base`: my model;
* `base + SlotMemory`;
* `base + SlotMemory, reset every W tokens`: the local control, where W is
  the attention window, or 64 if attention is full;
* `base + selective SSM sublayer`: the same placement and parameter count,
  with a Mamba-style discretisation.

If my model uses full attention, also run windowed attention with and
without memory. Window + memory versus full attention is the pairing where
the memory saves inference state.

Measure:

* validation loss (bits/token);
* passkey or needle accuracy beyond the attention window;
* multi-query associative recall (MQAR-style) accuracy;
* tokens/sec;
* FLOPs.

Expected from the study: memory ≤ base on validation loss on every seed,
better than the SSM arm, and better than the local control. If Stage A does
not reproduce that, stop and report. Do not scale up a result that does not
reproduce.

### Stage B — 150M

Run parameter-matched `base` against `base + SlotMemory`, with identical
data order, optimizer, schedule and token budget, and as many seeds as the
budget allows. State the number of seeds and the uncertainty it leaves.
Add windowed attention + memory if Stage A favoured it.

Suggested criteria, all fixed in advance:

1. validation loss of memory < base on every seed;
2. no regression beyond a threshold you state in advance on my existing
   downstream evaluations;
3. passkey or needle retrieval at depths beyond the attention window is not
   worse than base;
4. training throughput ≥ 0.85× base after the fused kernel, and counted
   FLOPs ≤ 1.10× base;
5. inference state reported against the KV cache.

### Ablations (Stage A, and Stage B if affordable)

* memory off;
* no copy readout;
* untied `W_k`;
* raw (unnormalised) values;
* slot count 16 / 32 / 64;
* insertion depth 1/2 vs 3/4;
* erase gate on.

## 9. The open problem worth your effort, after the above

The weak point is **write selectivity**. The learned write gate stores
ordinary text more than rare, important tokens, so a fact stated once is not
reliably recalled. If Stage A reproduces the problem (passkey near chance
for the memory arm), that is the most valuable thing to work on. Measure it
directly: log the mean write gate at the stated key versus elsewhere, and
the key's share of its best slot at question time. Try ideas one at a time,
on dev seeds, against that measurement. Promising directions that were *not*
tried in the study:

* a write gate conditioned on attention entropy or the residual norm;
* a separate small "salience" head trained with a stop-gradient target;
* top-k hard writes with a straight-through estimator;
* curriculum training with a higher passkey fraction early.

## 10. Deliverables

1. The module (ported from section 3.1), config flags for every component
   and ablation, and the tests from section 6.
2. The preregistrations, raw per-run results (JSON), and an aggregation or
   verdict script.
3. A report that leads with the verdict, including "no benefit" or "hurts"
   if that is what happens. Then give the numbers per seed, the throughput
   and FLOP cost, the state size, and what remains untested.

Do not overstate results. If the memory only helps on synthetic recall and
not on my model's real objective, say exactly that. If it helps only because
it adds within-window capacity (it matches the local control), say that too.
