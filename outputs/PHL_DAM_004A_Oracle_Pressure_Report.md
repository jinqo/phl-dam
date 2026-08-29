# PHL-DAM-004A — Oracle future-relevance upper bound under memory pressure

## Verdict

**ORACLE GAP PRESENT.** Under genuine bounded-memory pressure with eight slots, a policy that knows which stored binding will actually be needed again recalls 98.55%, 82.40% and 67.25% at 16, 24 and 32 writes per episode, against 64.45%, 41.91% and 28.55% for LRU. The paired oracle−LRU margin is +34.0, +40.5 and +38.7 percentage points, positive on 6/6 seeds at every pressure level, against a preregistered threshold of +5 pp. The benchmark therefore contains a large amount of exploitable future-relevance structure and is a fair setting in which to test the PHL temporal-lease hypothesis. This says nothing whatever about whether a learned lease can capture any of that headroom; it establishes only that the headroom exists and that conventional recency policies do not reach it.

## Research question

Under bounded memory, does knowing *when* a stored item will next be needed provide a materially better retention policy than ordinary eviction heuristics? If not, the benchmark cannot discriminate a PHL temporal lease from a static priority, and the learned stage must not be run.

## Frozen protocol

| Item | Setting |
|---|---:|
| Memory capacity | 8 slots |
| Writes per episode | 16 / 24 / 32 |
| Sequence length | 456 tokens |
| Write-to-query delay | 32–256 tokens (realised min 32, max 256) |
| Latent use classes | near 32–56, short 57–96, medium 97–160, far 161–256, persistent (2–3 uses), never |
| Query consumes binding | **No** — an item stays resident after retrieval |
| Content store | Exact: a query is a hit if and only if the binding is resident |
| Episodes | 2,000 per seed per condition |
| Seeds | 0–5 |
| Learned components | None |
| Oracle future information | Yes, and only for the oracle policies |

Every policy sees identical episodes, identical write and query positions, identical delays and identical capacity. Episodes are generated from a pure `random.Random` stream keyed on (seed, episode index), so they are byte-identical regardless of torch RNG state.

**Access, defined precisely.** A slot is accessed at the timestep it is written, and again at any query timestep whose queried key is the key held by that slot. LRU evicts the slot with the smallest last-access time; FIFO the smallest insertion time. Ties break on the lowest slot index for every policy.

### Documented deviation from the PHL-DAM-004 brief

The brief specifies 4–8 queries per episode. With eight slots that caps the number of distinct live items at eight, so a policy that knows only *whether* an item is ever queried can hold every live item at once — which would make the later PHL-vs-static-priority attribution structurally untestable. The canonical condition therefore scales the query budget with pressure (16→7–11, 24→13–19, 32→18–26). The brief's literal 4–8 is retained and reported as the `spec` secondary condition. This was fixed before any policy was run.

### Does the canonical condition actually create capacity pressure?

Query count is not evidence of pressure, so pressure is characterised directly by the peak number of concurrently live (written, still-needed) bindings.

| Condition | Peak live: mean | median | p10 | p90 | max | **>8** | **>10** | **>12** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical W=16 | 5.64 | 6 | 3 | 8 | 11 | 5.5% | 0.2% | 0.0% |
| canonical W=24 | 9.95 | 10 | 7 | 13 | 18 | **74.5%** | 40.6% | 13.5% |
| canonical W=32 | 14.14 | 14 | 11 | 18 | 23 | **97.9%** | 90.4% | 71.9% |
| spec W=16 | 3.92 | 4 | 2 | 6 | 8 | 0.0% | 0.0% | 0.0% |
| spec W=24 | 4.05 | 4 | 2 | 6 | 8 | 0.0% | 0.0% | 0.0% |
| spec W=32 | 4.17 | 4 | 2 | 6 | 8 | 0.0% | 0.0% | 0.0% |

Read this honestly:

- **canonical W=24 and W=32 are the capacity-pressure conditions.** Live items exceed the eight slots in 74.5% and 97.9% of episodes, so a policy must choose *among items it knows are needed*, which is where next-use timing is load-bearing.
- **canonical W=16 is not.** It over-subscribes in only 5.5% of episodes. It is the low-pressure control, and it is where the brief's gate 6 ("benefit specifically under memory pressure, not only on easy conditions") will be adjudicated.
- **The `spec` condition never over-subscribes at any pressure level** — its peak never exceeds 8. Its oracle recall is consequently 99.9% everywhere, and its pressure comes only from dead writes displacing live ones. It cannot separate "will this ever be needed" from "when", and should not be used for the PHL-vs-static attribution.

### Signal available at write time

Each write emits a `tag` token drawn from an 8-symbol alphabet with a fixed many-to-many map onto the latent use class (primary-tag probability 0.65). Under the realised class prior a Bayes-optimal decoder reaches **69.7–71.6% class accuracy** on the canonical conditions and **AUROC 0.849** for future-needed versus never-needed. The tag marginal is independent of write order (largest early/late half gap below 2% on randomised episodes), so recency policies cannot exploit it. This is deliberately far from the 99.95% giveaway cue of the superseded Lease-001 pilot: the signal is informative and clearly imperfect.

## Policies

| Policy | Rule | Future info |
|---|---|---:|
| `random` | uniform over occupied slots | no |
| `fifo` | smallest insertion time | no |
| `lru` | smallest last-access time | no |
| `oracle_future_relevance` | evict any item with no remaining query; otherwise the farthest next use | yes |
| `belady` | farthest next use, never-needed treated as infinity | yes |

**Belady and oracle future relevance are the same policy under this generator**, because a never-queried item has an infinite next use. They agree on every episode of every seed (`belady_matches_oracle_future_relevance: true` throughout), and a test asserts it. They are reported as one upper bound, not as two independent controls.

## Main results — canonical condition

Mean recall over seeds 0–5, sample SD in brackets.

| Policy | W=16 | W=24 | W=32 |
|---|---:|---:|---:|
| random | 59.38% (0.43) | 41.26% (0.17) | 30.26% (0.09) |
| fifo | 63.72% (0.54) | 42.56% (0.18) | 28.73% (0.21) |
| lru | 64.45% (0.51) | 41.91% (0.16) | 28.55% (0.22) |
| **oracle** | **98.55%** (0.14) | **82.40%** (0.16) | **67.25%** (0.20) |
| **oracle − LRU** | **+34.11 pp** | **+40.49 pp** | **+38.70 pp** |

Per-seed oracle−LRU margins:

| W | seed 0 | 1 | 2 | 3 | 4 | 5 | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | +33.46 | +34.16 | +34.02 | +34.82 | +34.21 | +33.97 | 6/6 |
| 24 | +40.31 | +40.47 | +40.56 | +40.66 | +40.36 | +40.58 | 6/6 |
| 32 | +38.28 | +38.79 | +38.73 | +38.96 | +38.53 | +38.87 | 6/6 |

Spread across seeds is under 1 pp everywhere, so these margins are practically as well as statistically decisive — the effect is roughly 7–8× the preregistered 5 pp threshold.

### Spec (secondary) condition

| Policy | W=16 | W=24 | W=32 |
|---|---:|---:|---:|
| random | 59.47% | 40.94% | 30.50% |
| fifo | 63.35% | 41.57% | 28.28% |
| lru | 65.49% | 42.32% | 28.41% |
| oracle | 99.95% | 99.95% | 99.93% |
| oracle − LRU | +34.46 pp | +57.63 pp | +71.52 pp |

The gap is larger here, but it is the *less* informative condition: with at most eight live items the oracle simply retains all of them, so the numbers measure "can you tell live from dead", not "can you tell when".

## Recall by delay (W=32 canonical, seed 0)

| Policy | 32–79 | 80–159 | 160–256 |
|---|---:|---:|---:|
| random | 46.6% | 27.6% | 25.8% |
| fifo | 45.8% | 25.1% | 25.5% |
| lru | 44.8% | 25.2% | 25.4% |
| oracle | 96.4% | 77.6% | 42.3% |

Even the oracle degrades with delay — at 160–256 tokens under 32 writes the item must survive roughly 13 evictions — but it degrades from a far higher base.

## Recall by intervening writes (W=32 canonical, seed 0)

| Policy | 0–4 | 5–8 | 9–16 | 17+ |
|---|---:|---:|---:|---:|
| random | 77.9% | 45.2% | 22.0% | 7.0% |
| fifo | **100.0%** | 73.4% | **0.0%** | **0.0%** |
| lru | **100.0%** | 71.3% | **0.6%** | **0.0%** |
| oracle | 69.4% | 64.4% | 66.7% | 68.0% |

This is the clearest diagnostic in the experiment. FIFO and LRU are step functions: they hold exactly the eight most recent bindings, so an item survives if and only if fewer than about eight writes followed it, and is lost with near-certainty beyond that. The oracle is nearly **flat** across all four bins — its retention tracks need, not recency. Random sits in between because it commits to no ordering at all.

## Memory survival and eviction diagnostics (W=32 canonical, seed 0)

| Policy | evictions/ep | fraction of evictions that were future-needed | dead items still resident at end | wrong-protection rate | old-needed vs recent-dead: correct choice |
|---|---:|---:|---:|---:|---:|
| random | 24.0 | 50.1% | 3.57 | 49.9% | 87.5% |
| fifo | 24.0 | 50.6% | 3.52 | 50.3% | 46.1% |
| lru | 24.0 | 50.8% | 3.49 | 50.5% | 48.4% |
| oracle | 24.0 | 27.4% | 0.54 | **0.0%** | 100.0% |

"Wrong protection" counts eviction decisions where at least one resident item was already dead, and the policy evicted a still-needed one instead. LRU and FIFO do this on roughly **half** of all such decisions; the oracle never does, by construction. At the end of an episode LRU still holds 3.49 never-queried items out of eight slots.

### The controlled old-but-future-relevant contrast

The generator marks ~14.6% of episodes as contrast episodes, in which an item written among the first writes is guaranteed to be queried later and is followed by a forced run of dead writes. Recall on that anchor item:

| Policy | W=16 | W=24 | W=32 |
|---|---:|---:|---:|
| random | 53.7% | 34.0% | 21.3% |
| fifo | 21.8% | 8.1% | **2.8%** |
| lru | 38.4% | 8.6% | **2.8%** |
| oracle | 100.0% | 100.0% | 97.9% |

This is precisely the failure mode a temporal lease is supposed to remove: at 32 writes, LRU loses the old-but-needed item **97.2%** of the time, while an oracle keeps it almost always.

## Unexpected policy ordering, reported rather than suppressed

At W=32 canonical, **random (30.26%) beats both FIFO (28.73%) and LRU (28.55%)**, and at W=24 FIFO edges out LRU. No ordering among the three non-oracle heuristics was required, and this one is workload-driven rather than an implementation fault:

- Most queries in this workload are first-use with long delays, so recency carries little information about future need. Repeat-use queries are only about 14% of the total.
- Worse, recency here is *anti*-correlated with need: the items that have waited longest are exactly the far-horizon items whose query is still coming. FIFO and LRU evict them systematically; random does not commit to that mistake, which is why random wins the old-needed-versus-recent-dead contrast 87.5% of the time against LRU's 48.4%.

The consequence for the rest of this research line is that **LRU is a weak baseline on this workload**, and beating LRU alone will not be evidence for the lease. The learned controls — static priority and learned utility — are the baselines that matter.

## Oracle gate

| Requirement | Observed | Result |
|---|---|---:|
| Oracle − LRU ≥ +5 pp at a meaningful nontrivial pressure condition | +40.49 pp at canonical W=24 | **Pass** |
| Consistent advantage at W=24 and/or W=32 | +40.49 pp and +38.70 pp, 6/6 seeds each | **Pass** |
| Practically and not merely statistically meaningful | Margins 7–8× threshold; per-seed spread < 1 pp | **Pass** |

## What this establishes

Under eight-slot capacity with 16–32 writes and 32–256 token delays, exact knowledge of future use is worth 34–41 percentage points of recall over LRU, and conventional recency heuristics capture essentially none of that headroom.

## What it does not establish

It does not show that any learnable signal recovers the gap: the oracle sees generator truth, while a learned policy sees only a tag whose Bayes-optimal class accuracy is about 70%. It does not compare PHL against anything — no model was trained here. It does not show that LRU is a strong baseline; on this workload it is not, and it is sometimes worse than random. It says nothing about content addressing, which is exact by construction in this stage. And the `spec` condition's very large gaps are an artefact of live items fitting in memory, not evidence of a harder problem.

## Addendum — the same gate at the compact scale

PHL-DAM-004B could not be run at the full scale: the content controller does not
reach its recall breakthrough within 600 updates at 456 tokens, so every arm sits
at chance and no eviction policy is measurable (see
`PHL_DAM_004B_Lease_Report.md` and `phl_dam_004b_scale_ladder.json`). A
`compact` scale profile was therefore added — 176 tokens, delays 29–104, writes
8/12/16 — keeping every structurally load-bearing feature: eight slots,
over-subscription, the noisy tag cue, never-queried distractors, repeat queries
and the contrast subset. Because 004B runs there, the oracle gate has to hold
there too. It does.

| Policy | W=8 | W=12 | W=16 |
|---|---:|---:|---:|
| random | 100.00% | 67.82% | 50.95% |
| fifo | 100.00% | 67.08% | 50.83% |
| lru | 100.00% | 66.70% | 50.83% |
| **oracle** | **100.00%** | **90.75%** | **73.06%** |
| **oracle − LRU** | **+0.00 pp** (0/6) | **+24.05 pp** (6/6) | **+22.23 pp** (6/6) |

W=8 is exactly 100% for every policy because eight writes fit in eight slots and
nothing is ever evicted. That is the intended zero-pressure control, and the
0.00 pp gap there is the correct reading, not a failure. Peak concurrent live
items exceed the eight slots in 39.2% of episodes at W=12 and 81.5% at W=16;
delays span 29–104; the Bayes-optimal tag decoder sits at 68.5–68.8% class
accuracy with AUROC 0.849, matching the full scale.

At W=16 canonical the mechanistic contrast is starker than at full scale: on the
controlled old-but-future-relevant anchor, FIFO and LRU recall it **0.0%** of the
time and the oracle **100%**. Recall by intervening writes inverts between them —
LRU falls 100% → 75.3% → 0.0% across the 0–4 / 5–8 / 9–16 bins while the oracle
*rises* 59.3% → 63.8% → 89.0%, because the items that survive many intervening
writes are precisely the long-horizon ones it is protecting.

**Compact-scale gate: pass** at W=12 and W=16, 6/6 seeds each.

## Decision

**CONTINUE — proceed to PHL-DAM-004B.** The learned stage should treat canonical
W=24 and W=32 (full) or W=12 and W=16 (compact) as the pressure conditions that
matter, the smallest write count as the low-pressure control, and the
static-learned-priority and learned-utility arms rather than LRU as the baselines
the lease must beat.

## Reproduction

```
cd C:\Users\liamp\Documents\Codex\2026-08-28\tes\work
py -3.14 -m unittest test_phl_dam_pressure_task test_phl_dam_eviction_policies -v
py -3.14 phl_dam_004a_oracle.py --writes 16 --seeds 0 1 2 3 4 5 --episodes 2000 --output ..\outputs\phl_dam_004a_w16.json
py -3.14 phl_dam_004a_oracle.py --writes 24 --seeds 0 1 2 3 4 5 --episodes 2000 --output ..\outputs\phl_dam_004a_w24.json
py -3.14 phl_dam_004a_oracle.py --writes 32 --seeds 0 1 2 3 4 5 --episodes 2000 --output ..\outputs\phl_dam_004a_w32.json
```

Environment: Python 3.14.0, torch 2.13.0+cpu, CPU only. Runtime 65 s / 105 s / 145 s per pressure level for both conditions and all six seeds.

Sources: `phl_dam_pressure_task.py`, `phl_dam_eviction_policies.py`, `phl_dam_004a_oracle.py`.
Tests: `test_phl_dam_pressure_task.py` (16), `test_phl_dam_eviction_policies.py` (14).
Raw results: `phl_dam_004a_w16.json`, `phl_dam_004a_w24.json`, `phl_dam_004a_w32.json`.
