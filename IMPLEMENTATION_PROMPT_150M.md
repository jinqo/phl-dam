# Prompt: add the PHL-DAM v4 slot-memory layer to a ~150M-parameter model

Copy everything below the line into Claude (Claude Code, run inside the
repository of the 150M model). It is self-contained; the reference
implementation lives in `jinqo/phl-dam` (`work/phl_dam_v2.py`,
`PHLDAMv2._fast_forward`), which Claude should read if it can reach it.

---

You are adding an external slot-memory layer ("SML", from the PHL-DAM v4
architecture) to my ~150M-parameter model, testing whether it helps, and
reporting honestly either way. Work in stages and stop at each checkpoint
marked **STOP** to show me what you found before continuing.

## 0. What the evidence does and does not say — read before designing

The design comes from a small, preregistered study (repo `jinqo/phl-dam`,
`outputs/PHL_DAM_v2_Report.md`). Treat it as a hypothesis for my model, not a
known result:

* It was validated only on **synthetic key→value associative recall**
  (176-token episodes with 3 bindings; and a 456-token "write pressure" task
  with 8–32 bindings competing for 8 slots), at **~30–40K parameters**, 8
  slots, CPU training. Never on natural language, never at scale.
* There it beat a parameter-matched Transformer, two SSMs, and two DNC
  variants on accuracy, steps to 90% recall (10–30 steps vs 27–250+) and
  training recall loss, with 392 floats of recurrent state.
* Most of the gain came from **generic mechanisms**: a copy readout and the
  DNC's write addressing. A DNC given the same copy readout came close.
* The toy hard-wired the binding roles (key = previous token, value =
  current token). In a language model those roles are only an inductive bias
  — the same pattern induction heads learn — and may or may not help.
* A 150M Transformer already does in-context associative recall with
  attention. The plausible benefits here are **constant-size memory for
  recall beyond the attention window, cheaper long-context inference, and
  faster acquisition of recall** — not lower short-context perplexity.
  Design the evaluation to detect those, and to detect harm.

## 1. Inspect my model first — **STOP after this step**

Read the codebase and report back, without changing anything:

1. Architecture: Transformer / recurrent / hybrid; `d_model`, layers, heads,
   FFN width, norm type and placement, positional scheme, vocabulary size,
   whether input and output embeddings are tied.
2. Context length, how documents are packed into sequences, and where
   document boundaries are marked (EOS/BOS tokens, attention masks, position
   resets).
3. Training loop: optimizer, LR schedule, gradient clipping, mixed precision,
   distributed setup, `torch.compile` use, current tokens/sec.
4. Existing evaluations and where long-context or retrieval evals could go.

Then propose where the SML goes (section 3), its sizes, the expected
parameter and throughput cost, and how you will keep the comparison
parameter-matched. Wait for my go-ahead.

## 2. The module — implement exactly this

Per sequence the SML keeps a recurrent state of `N` slots:
`keys K ∈ R^{N×dk}`, `values U ∈ R^{N×dv}`, `occupancy o ∈ R^N`, all zero at
the start of every document (reset at document boundaries, not only at
sequence starts). It reads the hidden stream `h_t ∈ R^d` at its insertion
point and processes positions left to right; at each step it **writes, then
reads**.

Projections and gates (all computed for every position in parallel before
the scan; only the slot recurrence is sequential):

```
unit(x)  = x / sqrt(|x|^2 + 0.05^2)                 # NOT F.normalize — see 4.1
c_t      = context features: h_t (optionally a small MLP over h_{t-1}, h_t)
k_t      = unit(W_k h_{t-1})        # candidate key, from the PREVIOUS position
q_t      = unit(W_k h_t)            # query: SAME projection W_k (tied)
v_t      = W_v h_t                  # candidate value, from the current position
w_t      = sigmoid(W_w c_t + b_w)   # write gate,      b_w initialised to -3
g_t      = sigmoid(W_g c_t)         # allocation gate (DNC)
beta_t   = softplus(W_b c_t) + 1    # write-address sharpness (DNC)
```

`W_k` and `W_v` have no bias. `h_{-1}` is zero, so `k_0 = unit(0) = 0`; this
is harmless only because `unit` is bounded (section 4.1).

Write (DNC write addressing, occupancy as usage):

```
alloc_t   = DNC allocation over ascending occupancy:
            sort o ascending -> o_(1..N);  a_(j) = (1 - o_(j)) * prod_{i<j} o_(i);
            scatter back to slot order.   (gradient flows through the sorted
            values; the sort indices are constants)
content_t = softmax(beta_t * (K @ k_t))
address_t = g_t * alloc_t + (1 - g_t) * content_t          # (N,)
write_t   = w_t * address_t                                # (N,)
keep_t    = 1 - write_t                                    # (N,)   [erase: see 2.1]
K <- unit(keep_t[:,None] * K + write_t[:,None] * k_t)
U <-       keep_t[:,None] * U + write_t[:,None] * v_t
o <-       keep_t * o + write_t
```

Read (after the write, so the value written at t is readable at t):

```
score_t  = (K @ q_t) / 0.1 + 0.25 * log(o + 1e-6)
a_t      = softmax(score_t)
r_t      = sum_n a_t[n] * U[n] / max(o[n], 1e-3)     # occupancy-normalised values
conf_t   = 1 - H(a_t) / log N
rho_t    = sigmoid(W_r [c_t, r_t, conf_t] + b_r),  b_r initialised to +1
m_t      = rho_t * r_t
```

Outputs, two paths:

```
h_t      <- h_t + W_o m_t                            # back into the residual stream
logits_t <- logits_t + s * (m_t @ (W_v E)^T)         # copy readout
```

`E` is the token-embedding matrix (vocab × d); `W_v E` projects every
vocabulary item into value space once per forward pass; `s` is a learned
scalar initialised to **2.0**. The copy readout lets a clean retrieval of
token v score token v highest from initialisation. If the SML sits in a
middle layer, feed `m_t` of the chosen SML layer(s) to the copy term at the
LM head (store it alongside the hidden state).

Why each piece is there (each was measured, see the report):

* **Normalised values** (`U / o`): with a barely-open write gate, raw values
  are tiny; the model's write gate collapsed and starved itself of gradient.
  The ratio is the slot's write-weighted average, independent of gate size.
* **Copy readout + tied `W_k`**: the plateau before learning waited on the
  output layer learning to decode retrieved values and on key/query
  alignment; these remove both at initialisation (breakthrough 20 → 10 steps).
* **DNC write addressing**: the original PHL-DAM allocation smeared every
  overflow write across all slots, so the write gate learned to under-write
  (21% of needed items never stored). Usage-sorted allocation fixed it.
* **No PHL horizon lattice**: removing it made learning faster.

### 2.1 Erase gate — tested, not adopted (keep it as an ablation flag, off)

An address-based erase, `keep_t = (1 - write_t) * (1 - e_t * address_t)` with
`e_t = sigmoid(W_e c_t + b_e)`, `b_e = -2`, lets a fully addressed slot be
replaced outright instead of blended. It was motivated by a real diagnostic
(the model recalled its most recently written bindings worst under heavy
write pressure), looked good on two dev seeds, and then failed its
preregistered confirmation: +1.3 pp at 24 writes (not robust), +0.0 pp at 32,
and breakthrough one step later. Implement it behind a flag for the
ablation, default off. The underlying weakness — recent bindings blended into
occupied slots under overflow — is still open; if your model shows it,
investigate there.

## 3. Integration at ~150M — sizes and placement

Start small and measurable, then scale:

* **One SML layer first**, inserted as an extra residual sublayer after the
  attention sublayer of a middle block (e.g. block L/2), pre-normed like the
  model's other sublayers. Later try 2–4 layers or several independent
  memory heads per layer (separate `W_k, W_v`, gates and state).
* **Sizes to start**: `N = 16–64` slots per head, `dk = dv = 64`,
  `d = d_model`. State per head is `N·(dk+dv+1)` floats — e.g. 64 slots ×
  129 = 8,256 floats per sequence, versus the KV cache's
  `2·layers·d_model·T`.
* **Parameter matching**: the SML adds roughly `d·(2·dk + dv) + d·dv + small
  gates`. Remove the same count elsewhere (e.g. shrink one block's FFN
  width) so the comparison with the unmodified model is fair. Report both
  counts.
* **The scan is sequential over T.** A Python loop over 2k+ tokens will be
  slow. In order: (1) keep all projections outside the loop (as above);
  (2) `torch.compile` the step function; (3) if throughput is still the
  bottleneck, write a Triton/CUDA kernel for the scan (state is tiny; one
  program per (batch, head) holding K, U, o in SRAM), or process in chunks.
  Measure tokens/sec before and after and report the overhead honestly.
* **Mixed precision**: keep the scan state (K, U, o) and the divisions in
  fp32 even if the rest of the model runs in bf16.
* **Documents**: reset the state at every document boundary inside packed
  sequences; a slot must never carry a binding across documents.

## 4. Numerical stability — non-negotiable

These bugs happened in the reference implementation; each produced gradient
spikes of 10^3–10^12:

1. `F.normalize` on an exactly zero key divides by 1e-12 → Jacobian 1e12.
   Use `unit(x) = x / sqrt(|x|^2 + 0.05^2)` for keys, candidate keys and
   queries.
2. Dividing by occupancy: floor it (`max(o, 1e-3)`; raise to 0.05 if you add
   anything that creates low-occupancy slots).
3. `log(o + eps)` in the read prior: its derivative is 1/o; keep eps ≥ 1e-6
   and raise it if spikes appear.
4. Keep gradient clipping on; log the per-step global grad norm and, on any
   spike > 1e3, record which op amplified it (hook `grad_in/grad_out` per op;
   the reference repo's `diag3.py` approach).

## 5. Tests to write before any training run

* **Causality, per query**: for random positions t, changing any token > t
  leaves all logits ≤ t bit-identical (test with the copy readout on).
* **Off-switch equivalence**: with the SML disabled (or `W_o = 0`, `s = 0`,
  `rho = 0`), the model's outputs equal the unmodified model's exactly.
* **Document reset**: two documents packed in one sequence give the second
  document the same SML outputs as running it alone.
* **Retrieval sanity at initialisation**: write a single binding with the
  gate forced open, query its key, check the copy logits rank the stored
  value first.
* **Finite gradients** over a full-length sequence from a zero state, in the
  training precision.
* **Allocation gradient**: the usage allocation has nonzero gradient with
  respect to occupancy.
* **Parameter and state accounting** printed and asserted.

## 6. Experiments — preregister, then run

Write a short preregistration file before any comparison run: candidates,
baselines, seeds, metrics, and the exact pass/fail rule. Tune only on dev
seeds / dev data; confirm on held-out seeds / data. Report every criterion,
pass or fail.

**Stage A — synthetic, small and cheap (days, not weeks).** A ~10–20M
version of my model with and without the SML, several seeds each, on:
multi-query associative recall (MQAR-style, many key–value pairs, queries
after long delays), passkey / needle retrieval beyond the attention window,
and the write-pressure setting (more bindings than slots, with a cue
predicting which will be queried). Metrics: recall accuracy, steps to 90%
recall, training recall loss, tokens/sec. This checks the port reproduces
the reference behaviour before spending 150M-scale compute. **STOP** and
show me the results.

**Stage B — 150M.** Parameter-matched: unmodified model vs model + SML,
identical data order, optimizer, schedule and token budget. Metrics:
validation perplexity (must not regress beyond a threshold you state in
advance), long-context retrieval (needle/passkey at several depths and
lengths, including beyond the attention window), in-context key–value
lookup, any downstream evals my repo already has, and throughput. Seeds: as
many as the budget allows (state the number and the uncertainty it leaves).

**Ablations** (Stage A, and Stage B if affordable): SML off; no copy
readout; untied `W_k`; raw (unnormalised) values; Stage-B-style allocation
instead of DNC addressing; erase gate on; slot count N.

## 7. Deliverables

1. The SML module, config flags for every component above, and the tests.
2. The preregistration(s), raw per-run results, and an aggregation script.
3. A report that leads with the verdict — including "no benefit" or "hurts
   perplexity" if that is what happens — then the numbers, the throughput
   cost, and what remains untested.

Do not overstate results. If the SML only helps on synthetic recall and not
on my model's real objective, say exactly that.
