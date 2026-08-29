# PHL-DAM — what it is and how it works

A description of the architecture **as actually implemented** in this repository,
with the real constants from the code. Where a mechanism has been tested, the
verdict is stated; where it was refuted, that is stated too.

---

## 1. The one-sentence version

PHL-DAM is a **recurrent** sequence model that carries two kinds of state
side by side: a small smoothed "context" state spread across four timescales
(the **PHL** part), and a bank of eight explicit key→value memory slots that are
written and read by content matching (the **DAM** part).

It is not a Transformer and not a state-space model. The nearest relatives in
the literature are the **Neural Turing Machine / Differentiable Neural Computer**
family — externally addressable memory driven by a learned controller.

---

## 2. Shape and size

| Quantity | Value |
|---|---:|
| Model width `d_model` | 64 |
| PHL horizons × width | 4 × 16 |
| Memory slots | 8 |
| Key dimension | 24 |
| Value dimension | 24 |
| Parameters | 33,034 |
| Recurrent state at any length | **456 floats** |

The 456 floats are the whole inference state: `4×16 = 64` for PHL, plus
`8 × (24 key + 24 value + 1 occupancy) = 392` for the memory. It does **not**
grow with sequence length — that is the property it is built around, and it is
what distinguishes it from attention, whose KV cache at length 176 is 16,896
floats.

---

## 3. Data flow for one token

```
 tokens ──► embedding ──► 3-token window ──► context encoder ──┐
                          (t-2, t-1, t)      Linear+Tanh       │
                                                               │
        ┌──────────────────────────────────────────────────────┤
        │                                                      │
        ▼                                                      ▼
  ┌───────────┐                                        ┌──────────────┐
  │ PHL state │  transport across 4 horizons           │  DAM memory  │
  │  4 × 16   │  then RMSNorm, then readout            │   8 slots    │
  └─────┬─────┘                                        └──────┬───────┘
        │                                                     │
        │  phl_contribution                 memory_contribution
        │                                                     │
        └──────────────► + context + ◄────────────────────────┘
                              │
                          RMSNorm
                              │
                          output ──► next-token logits
```

Everything is causal: step *t* sees only tokens ≤ *t*. There are no positional
encodings at all — position is implicit in the recurrence.

---

## 4. The PHL part — multi-timescale context

The PHL state is a 4 × 16 matrix: four **horizons**, each a 16-dimensional
vector. Each step it is mixed by a fixed transport matrix, then new input is
added and the result normalised:

```python
transported = transport @ phl_state          # 4×4 fixed matrix
injected    = phl_input(context).view(4, 16) # new information
phl_state   = RMSNorm(transported + injected)
contribution = phl_readout(phl_state.flatten())
```

The transport matrix is fixed, not learned:

```
horizon 0 ──0.65──► horizon 0        (fast, forgets quickly)
           ──0.35──► horizon 1
horizon 1 ──0.65──► horizon 1
           ──0.35──► horizon 2
horizon 2 ──0.65──► horizon 2
           ──0.35──► horizon 3
horizon 3 ──0.98──► horizon 3        (slow, nearly persistent)
```

So information injected now stays mostly in horizon 0, and a fraction leaks
"upward" into slower horizons each step. Horizon 3 retains 98% per step, giving
it an effective memory of tens of steps. The intended effect is a cheap
multi-timescale summary: fast horizons track local structure, slow ones carry
long-range context.

---

## 5. The DAM part — eight addressable memory slots

This is where the model's actual recall ability lives. Each slot holds a
**key** (24 floats), a **value** (24 floats), and an **occupancy** scalar.

### Writing

At every timestep the model computes a candidate binding from the local token
window — the previous token is treated as a candidate key, the current token as
a candidate value — and a gate deciding whether to store it:

```python
candidate_key   = normalise(key_projection(previous_token))
candidate_value = value_projection(current_token)
write_strength  = sigmoid(write_gate(context))       # 0 … 1
```

Nothing tells the model where the real bindings are. It must learn from the
prediction loss alone that "after a WRITE marker, the next two tokens are a
key and a value worth storing". The write gate starts nearly closed
(bias −3.0, so ≈0.047) and has to open selectively.

**Where** the write goes is decided by an allocation distribution over slots:

```python
allocation_score = 5·(1 − occupancy) + slot_bias        # prefer empty slots
allocation       = softmax(allocation_score / 0.10)     # sharp, near one-hot
write_amount     = write_strength · allocation
```

The `5·(1 − occupancy)` term dominates while any slot is free, so the memory
fills up before it overwrites anything. Once every slot is occupied that term
goes flat and something else has to choose the victim — which is the whole
subject of the eviction experiments below.

There is also a **merge** path: if the candidate key closely matches a stored
key (cosine similarity above ≈0.72), the write is routed to that slot instead of
a new one, so repeating a key updates it rather than duplicating it.

The slots are then updated as a convex blend, and keys are re-normalised:

```python
keys      = normalise((1 − write_amount)·keys   + write_amount·candidate_key)
values    =           (1 − write_amount)·values + write_amount·candidate_value
occupancy = occupancy + write_amount·(1 − occupancy)     # only ever rises
```

### Reading

Reading is content-addressed — there are no slot indices anywhere:

```python
query         = normalise(query_projection(current_token))
content_score = query · keys                       # similarity to every slot
read_score    = content_score / 0.10 + 0.25·log(occupancy)
attention     = softmax(read_score)                # over 8 slots
retrieved     = attention · values
```

A confidence signal is derived from the entropy of that 8-way attention (peaked
= confident), and a read gate decides how much of the retrieved value to inject:

```python
confidence    = 1 − entropy/log(8)
read_strength = sigmoid(read_gate([context, retrieved, confidence]))
memory_contribution = memory_projection(read_strength · retrieved)
```

### Output

```python
hidden = RMSNorm(context + phl_contribution + memory_contribution)
logits = output(hidden)
```

---

## 6. Training objective

```
loss = all-token next-token cross-entropy  +  recall-token cross-entropy
```

The second term weights the specific positions where a stored value must be
produced. An optional write-budget penalty exists but is set to **0** — it was
tested and shown unnecessary.

A characteristic quirk: recall sits at chance (CE ≈ ln 10 = 2.303) for
**hundreds of updates**, then drops sharply — 2.36 → 0.94 → 0.41 → 0.05 within
about 150 steps. The long plateau is normal for this architecture, not a
failure, and mistaking it for one has been a repeated trap.

---

## 7. The eviction problem (extension studied in 004B–004D)

With 8 slots and more bindings than slots, something must decide what to
overwrite. The extension adds a per-slot ledger (owner, insertion time, last
access, access count) and an eviction score that enters the allocation:

```python
allocation_score = 5·(1 − occupancy) + slot_bias − eviction_score
```

Eight policies were compared under identical trained weights:

| Policy | Rule |
|---|---|
| `random` | uniform |
| `fifo` | oldest insertion |
| `lru` | least recently accessed |
| `oracle` | true next-use distance (evaluation only) |
| `content_only` | learned score from occupancy, key/value norms, age, staleness |
| `static_priority` | learned relevance, frozen at write |
| `phl_lease` | learned relevance, **transported through PHL horizons** |
| `learned_utility` | learned score from age, staleness, access count |

### The temporal lease — the distinctive idea, and it was rejected

The `phl_lease` arm was the architecture's most original claim: attach to each
slot a distribution over "when will this matter" horizons
(due / near / short / medium / far / never), and let it **evolve** as time
passes, so a memory becomes more protected as its moment approaches.

It failed its preregistered adoption gate. With the timing predictor working at
the Bayes ceiling (AUROC 0.85 vs 0.849 optimal) and the transport operator
correctly calibrated, transporting relevance was worth **+1.19 pp** over simply
holding it static on the one clean seed, and on another seed coincided with a
training collapse (−36.8 pp). The mechanism does not earn its complexity.

What did survive is not PHL-specific: `content_only`, a **57-parameter** score
over purely local features, beat LRU on 3/3 seeds at the highest pressure.

---

## 8. What is actually established

**Verified**, at 176 tokens with 3 bindings, 6 seeds, matched parameters,
identical streams and budget:

| Model | Params | State | Recall |
|---|---:|---:|---:|
| **PHL-DAM** | 33,034 | 456 | **99.55%** |
| Causal Transformer | 33,074 | 16,896 | 47.08% |
| Selective SSM (S6-style) | 33,025 | 96 | 27.77% |
| Diagonal SSM (S4D-style) | 33,139 | 96 | 25.94% |

Disabling memory retrieval drops PHL-DAM to 9.99% — chance — so the memory
path, not a shortcut, carries the result.

**How to read that table honestly.** The benchmark is synthetic associative
recall: store arbitrary key→value pairs, retrieve by exact key after a delay.
It was built around explicit key-value storage, which is exactly what PHL-DAM
has and the baselines do not. Associative recall is the canonical known weakness
of SSMs. PHL-DAM also carries **4.75× more state** than either SSM. This is a
real win on a task the architecture is designed for, against baselines
structurally disadvantaged on it — not evidence of a better general sequence
model.

## 9. What is not established

- **It does not train reliably beyond this toy scale.** At 456 tokens with
  16–32 writes, 7 of 25 runs diverged and none learned.
- **Never tested on anything real** — no language, no code, no natural data.
- **The novel component was rejected**; the component that works is the
  conventional NTM/DNC-style one, and no NTM/DNC baseline exists yet.
- **Nothing is known about scaling.** 33K parameters is a research toy.

## 10. Current state of work

The instability was traced by bisection to a z-score standardisation of the
eviction score: its Jacobian grows as `1/spread`, reaching ~9.8e5 when all
eight slots score alike, which happens on roughly 1 timestep in 450 — matching
the intermittent, data-dependent spikes exactly. Flooring the spread at 0.1
bounds that Jacobian at 8.75 and leaves eviction ordering unchanged wherever the
spread is informative. Full 700-step validation is in progress; early runs show
gradients ~1 where the baseline had already reached 1e11.
