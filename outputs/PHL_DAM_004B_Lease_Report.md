# PHL-DAM-004B — Learned temporal lease under memory pressure

## Verdict

The PHL-transported temporal lease fails its preregistered gate on every criterion. At the full scale specified in the brief the experiment is uninformative: no arm reached its recall breakthrough within 600 updates, so all eight policies sit at chance and eviction quality is unmeasurable. At a reduced compact scale where the same model does learn, the transported lease beats LRU by −1.1 pp at the highest pressure (2/3 seeds), is beaten by the simplest non-PHL learned control by 24.9 pp (0/3 seeds), broke through on only 1 of 3 seeds against that control's 3 of 3, and its eviction score separates future-needed from never-needed slots at AUROC 0.28–0.50 — chance or worse. The lease head never learned to decode the write-time cue from the predictive objective alone. A learned eviction score built from purely local recency and content features does beat LRU, consistently and on every seed; it simply contains no lease and no PHL transport. On this evidence the temporal-lease mechanism has not earned its complexity, and the PHL-specific claim is unsupported.

## Research question

Under genuine bounded-memory pressure, can PHL-transported temporal relevance preserve future-needed memories better than ordinary eviction policies and better than non-PHL learned memory management?

## Frozen protocol

| Item | Full scale | Compact scale |
|---|---:|---:|
| Memory capacity | 8 slots | 8 slots |
| Sequence length | 456 tokens | 176 tokens |
| Writes per episode | 16 / 24 / 32 | 8 / 12 / 16 |
| Write-to-query delay | 32–256 | 29–104 |
| Training steps | 600 | 800 |
| Batch size | 16 | 16 |
| Learning rate / optimiser | 2e-3 / AdamW(wd 1e-4) | same |
| Write-budget weight | 0 | 0 |
| Promotion, read-to-write feedback | off | off |
| Seeds | 0–1 (partial) | 0–2 |
| Eval episodes per setting | 512 | 512 |
| Future-use labels to learned arms | none | none |

Training mixes pressure levels uniformly per batch, so one model per (arm, seed) is valid at every level and is evaluated at each separately. Every arm shares the generator, frozen seeds, streams, capacity, delay distribution, evaluation episodes and training budget. No arm receives next-use labels at training or evaluation; the oracle is evaluation-only.

## Models and policies

| Arm | Eviction score | PHL transport |
|---|---|---:|
| `random` | uniform | — |
| `fifo` | insertion time | — |
| `lru` | last access time | — |
| `oracle` | true next-use distance (eval only) | — |
| `content_only` | MLP over occupancy, key norm, value norm, age, staleness | no |
| `static_priority` | lease head, frozen at write, no transport | no |
| `phl_lease` | lease head, transported each token through the horizon lattice | **yes** |
| `learned_utility` | MLP over age, staleness, access count, insert write strength | no |

`static_priority` and `phl_lease` are parameter-identical by construction (400 eviction-policy parameters each; the transport operator is a fixed buffer with none). They differ structurally only in whether relevance is transported. They are trained independently, so this isolates the architectural contribution of transport, not a weight-level ablation.

Lease state is six horizons — due / near / short / medium / far / never — initialised at write from the local controller representation and evolved by one-token interval-overlap transport. K and V never move between slots; only the lease evolves. A test asserts that permuting the lease leaves keys and values bit-identical, and a second test asserts the lease nonetheless changes the eviction score, so the separation is real and not vacuous.

## Parameter and state accounting

| Arm | Total parameters | Eviction-policy parameters | Content parameters | Recurrent state floats |
|---|---:|---:|---:|---:|
| `content_only` | 38,251 | 57 | 38,194 | 496 |
| `learned_utility` | 38,243 | 49 | 38,194 | 496 |
| `static_priority` | 38,594 | 400 | 38,194 | 544 |
| `phl_lease` | 38,594 | 400 | 38,194 | 544 |

## Main result 1 — the full scale is uninformative

Six runs (four arms, seeds 0–1), 600 updates each. None reached the recall breakthrough.

| Arm | Recall | Residency | Recall-token CE |
|---|---:|---:|---:|
| `content_only` | 9.14% | 0.000 | 2.3124 |
| `static_priority` | 9.23% | 0.000 | 2.3120 |
| `phl_lease` | 9.27% | 0.000 | 2.3113 |
| `learned_utility` | 9.51% | 0.000 | 2.3138 |

ln 10 = 2.3026, so recall-token CE is at chance for ten possible values. The decisive internal signature: **within every run, `random`, `fifo`, `lru`, `oracle` and the learned arm return identical recall to three decimals**, because the write gate never crossed its commitment threshold, residency is 0.000 and there were **zero eviction decisions** in any run. An oracle indistinguishable from LRU is the tell that this is a content-path failure, not an eviction result.

### Why — the scale ladder

The same model, same `phl_lease` arm, same seed, on progressively more Stage-B-like versions of the task (`phl_dam_004b_scale_ladder.json`):

| Rung | Breakthrough step | Recall | Residency | Recall given resident | Write gate at binding / elsewhere |
|---|---:|---:|---:|---:|---:|
| A — 176 tokens, 3 writes | 150 | **99.18%** | 98.52% | 99.5% | 0.996 / 0.040 |
| B — 176 tokens, 8 writes | 150 | **89.40%** | 91.11% | 95.5% | 0.652 / 0.018 |
| Full — 456 tokens, 16–32 writes | never (600) | 9.3% | 0.000 | — | 0.024 / 0.042 |

Rung A reaching 99.18% recall with a 98.5% residency ledger rules out an
implementation defect: identical code and identical arm, only the task is
smaller. **Amended by PHL-DAM-004C:** the null is not about 456 tokens and not
about the budget. Rerunning the full scale at 1400 updates - 2.3x this budget -
still yields no breakthrough on either seed, and a 451-token task with 8/12/16
writes learns by step 550. The null reflects the controller's reach at **2-4x
slot over-subscription** (writes 16/24/32), which at these delays forces a
~456-token sequence. See `PHL_DAM_004C_Scale_Collapse_Report.md`. Stage B's own recall CE also sits at chance until step ~250–300 before dropping sharply, so a long plateau is this architecture's normal dynamic — the full-scale runs simply never leave it.

## Main result 2 — the compact scale, where the lease can actually be tested

Twelve runs, four arms × seeds 0–2, 800 updates. Recall, residency, and the within-run contrast against LRU (identical content weights, only the eviction rule differs).

### Highest pressure, canonical W=16 (81.5% of episodes over-subscribe the 8 slots)

| Arm | Recall | Residency | Recall given resident | vs LRU within-run | vs oracle |
|---|---:|---:|---:|---:|---:|
| **`content_only`** | **59.27%** | **53.19%** | 95.51% | **+4.53 pp (3/3)** | −6.10 pp |
| `static_priority` | 36.79% | 30.35% | 88.03% | +5.30 pp (2/3) | −3.12 pp |
| `learned_utility` | 35.41% | 30.47% | 79.50% | −3.45 pp (0/3) | −9.85 pp |
| **`phl_lease`** | **34.38%** | **26.98%** | 83.82% | **−1.08 pp (2/3)** | −14.69 pp |

### Moderate pressure, canonical W=12

| Arm | Recall | Residency | vs LRU within-run | vs oracle |
|---|---:|---:|---:|---:|
| **`content_only`** | **74.15%** | 70.86% | **+7.86 pp (3/3)** | +2.18 pp |
| `phl_lease` | 46.25% | 42.98% | +3.20 pp (3/3) | −8.00 pp |
| `learned_utility` | 42.95% | 37.24% | −2.29 pp (0/3) | −9.10 pp |
| `static_priority` | 42.53% | 36.01% | +5.53 pp (3/3) | −0.07 pp |

### Zero-pressure control, canonical W=8

Eight writes fit in eight slots. `content_only` still leads at 87.92%; `phl_lease` 58.86%. The lease provides no advantage where there is nothing to evict, as expected.

## Paired seed effects

| Contrast (canonical W=16) | Mean | Per seed | Wins |
|---|---:|---|---:|
| `phl_lease` − LRU (within-run) | −1.08 pp | −7.7 / +1.9 / +2.5 | 2/3 |
| `phl_lease` − oracle (within-run) | −14.69 pp | −17.3 / −11.7 / −15.1 | 0/3 |
| `phl_lease` − `content_only` (across-run) | **−24.89 pp** | −8.2 / −43.0 / −23.4 | **0/3** |
| `phl_lease` − `static_priority` (across-run) | −2.42 pp | +8.9 / +12.8 / −28.9 | 2/3 |
| `content_only` − LRU (within-run) | **+4.53 pp** | +4.0 / +1.7 / +7.9 | **3/3** |

The `phl_lease` − `static_priority` sign flips across seeds (+8.9, +12.8, −28.9), so the transport-versus-static contrast — the one comparison that could isolate PHL — is dominated by seed noise at n=3.

## Breakthrough count is itself a result

| Arm | Runs that learned (final recall CE < 2.0) | Final CE by seed |
|---|---:|---|
| `content_only` | **3/3** | 1.75 / 0.30 / 1.61 |
| `static_priority` | 2/3 | 1.97 / — / 1.55 |
| `learned_utility` | 2/3 | 1.76 / — / 1.51 |
| **`phl_lease`** | **1/3** | 1.72 / — / — |

The transported lease is the only arm that failed to learn on a majority of seeds, while the simplest control learned on all three. At n=3 this could be noise, but it points away from the hypothesis, and it is a cost of the mechanism rather than a nuisance to average away: an eviction rule that destabilises content learning is worse, not neutral.

## Lease diagnostics — the mechanistic core of the negative result

Mean lease distribution over the six horizons at eviction time, split by whether the slot's occupant was actually needed again (canonical W=16):

| Run | future-needed | never-needed |
|---|---|---|
| `phl_lease` seed1 | [.039 .057 .194 .015 .073 **.622**] | [.047 .070 .243 .014 .059 **.567**] |
| `phl_lease` seed2 | [.071 .005 .006 .014 .005 **.899**] | [.076 .005 .005 .015 .003 **.896**] |
| `static_priority` seed0 | [.495 .038 .047 .011 .040 .370] | [.402 .042 .040 .011 .039 .466] |

The distributions for items that *will* be queried again and items that never will are nearly identical. The lease is not encoding future relevance.

| Diagnostic (canonical W=16) | `phl_lease` s0 | s1 | s2 |
|---|---:|---:|---:|
| Eviction decisions observed | 15 | 9,567 | 7,898 |
| AUROC, future-needed vs never-needed | 0.282 | 0.433 | 0.496 |
| Pearson(lease protection, next-use distance) | −0.456 | +0.037 | −0.139 |
| Fraction of evictions that took a needed item | 0.600 | 0.345 | 0.416 |
| Wrong-protection rate | 0.600 | 0.337 | 0.402 |
| Old-needed vs recent-dead: correct choice | 0.400 | 0.538 | 0.557 |
| Lease entropy | 1.482 | 0.911 | 0.317 |

AUROC at or below 0.5 means the eviction score ranks future-needed slots no better than chance. The correlation between predicted imminence and true next-use distance is near zero and flips sign across seeds. On the controlled old-but-future-relevant contrast the lease chooses correctly 40–56% of the time — coin-flip territory, against the oracle's 100% in 004A. Seed 0's figures rest on only 15 eviction decisions and should not be weighted.

## Oracle gap

The oracle must be read per run, not pooled: it is evaluated under each run's own
trained content weights, and those weights differ between arms. Pooled across all
twelve runs it averages 49.90% recall / 55.65% residency at canonical W=16, but
that number mixes incomparable content paths and should not be used as a single
ceiling.

| Arm | Oracle under that arm's own weights | Arm | Gap to oracle |
|---|---:|---:|---:|
| `content_only` | 65.36% | 59.27% | **+6.10 pp** |
| `phl_lease` | 49.06% | 34.38% | **+14.69 pp** |

`content_only` closes to within 6.10 pp of the ceiling available to it;
`phl_lease` remains 14.69 pp short of its own. The transported lease is therefore
*further* from the oracle than the non-PHL control, the opposite of the pattern
that would support the hypothesis. Note also that the oracle ceiling itself is
lower under `phl_lease`'s weights (49.06% vs 65.36%) — its content path is weaker,
which is the co-adaptation confound the residency split exists to expose.

## Gate audit

| # | Preregistered requirement | Observed | Result |
|---|---|---|---:|
| 1 | Beat LRU by ≥5 pp mean recall | −1.08 pp at W=16; +3.20 pp at W=12 | **Fail** |
| 2 | Beat strongest non-PHL learned control by ≥5 pp | −24.89 pp vs `content_only` | **Fail** |
| 3 | Win ≥2/3 paired seeds against both important controls | 2/3 vs LRU, 0/3 vs `content_only` | **Fail** |
| 4 | No material all-token CE regression | 1.5790 vs 1.5315 for `content_only` | Marginal fail |
| 5 | Remain finite in all runs | finite in 18/18 runs | Pass |
| 6 | Benefit specifically under pressure | advantage shrinks as pressure rises (+3.2 pp at W=12 → −1.1 pp at W=16) | **Fail** |

**Overall: fail.** Five of six criteria fail, including all three that carry the attribution.

## Attribution analysis

**Did temporal PHL transport help?** No. The transported lease does not beat LRU under pressure, and against its parameter-identical static twin the difference flips sign across seeds (+8.9, +12.8, −28.9 pp).

**Did simple learned utility explain the result?** Not the `learned_utility` arm — it is the weakest against LRU (−3.45 pp, 0/3). But `content_only` — occupancy, key norm, value norm, age, staleness, 57 parameters, no lease, no transport — beats LRU by +4.53 pp on 3/3 seeds at W=16 and +7.86 pp at W=12. Learned eviction from purely local features works. That is a real positive finding, and it belongs to the simplest control, not to PHL.

**Did static priority explain the result?** `static_priority` also beats LRU (+5.30 pp at W=16, +5.53 pp at W=12), and beats `phl_lease` at W=16. Whatever the lease head contributes, transporting its output does not add to it.

**How close was the lease to the oracle?** 14.69 pp short at W=16, versus 6.10 pp for `content_only` — further from the ceiling than the control it was meant to beat.

## Known defect in the compact `phl_lease` runs — disclosed after publication

`LEASE_BIN_EDGES` was hardcoded to the full-scale delay ranges
(0–31 / 32–56 / 57–96 / 97–160 / 161–256) and was **not** rebuilt when the
compact profile was selected, whose delays are 29–104. The transport operator's
per-token rates are `1 / bin width`, so under the compact runs mass walked the
lattice roughly 2.5× too slowly: starting from full "far" mass, only 8.8% has
reached the "due" horizon after 100 tokens, while a compact far item is actually
queried at 83–104 tokens. The transported lease was therefore signalling
imminence long after the query had already passed.

This confound acts **specifically against `phl_lease`**. It does not touch
`static_priority` (which ignores transport by construction), `content_only` or
`learned_utility`, and it does not affect the full-scale null (nothing learned
there). The compact `phl_lease` numbers in this report should be read as a
lower bound on that arm, not as a clean measurement.

It does **not** rescue the central mechanistic finding. `static_priority`, which
uses the same head with no transport at all, also shows near-identical lease
distributions for future-needed and never-needed items ([.495 …] vs [.402 …]),
so the write-time head failed to encode future relevance independently of how
fast that estimate was later transported. The gate failures on criteria 2 and 3 —
being beaten 24.9 pp by `content_only`, 0/3 seeds — are likewise unaffected by
transport speed.

The fix (scale-derived lease bins) and a rerun under equal timing supervision are
the subject of PHL-DAM-004B-S.

## Failure analysis

The mechanism failed at the first link in its own chain: the lease head never learned to decode the write-time tag. AUROC 0.28–0.50 and near-identical lease distributions for needed and never-needed items mean the end-to-end recall gradient — routed through a straight-through allocation, through an eviction that only matters when memory is full, and through a query that may arrive a hundred tokens later — did not reach the lease head with enough signal to teach it anything about future use. Transport cannot help a relevance estimate that does not exist. This is a failure of *learnability under the predictive objective*, not proof that transported relevance would be useless given a working predictor.

Implementation pathologies found and fixed before any result was collected, each documented in code:

1. An infinite eviction score for free slots produced NaN through the straight-through softmax from the first update. Replaced with a finite additive penalty.
2. `F.normalize` on an empty slot's zero key back-propagates a Jacobian of order 1e12, which overflows float32 through the slot recurrence. Empty slots now take an un-normalised branch — forward-identical to Stage B, Jacobian bounded at one.
3. `sqrt'(0)` is infinite and was multiplied by a clamp's zero gradient, giving NaN. The epsilon moved inside the square root.
4. A hard one-hot allocation with a straight-through estimator, as originally planned, is incompatible with Stage B's soft-write bootstrap: occupancy needs roughly fifteen soft writes to cross the free threshold, so each slot accumulated a blend of consecutive tokens instead of one clean binding, and recall stayed at chance. Replaced with Stage B's own allocation softmax plus one added term — the standardised eviction score. While slots are free the behaviour is Stage B's exactly; once all eight are occupied the occupancy term is flat and the eviction score alone selects the victim.

Two deviations from the approved plan, both fixed before any arm comparison existed and applied identically to every arm: the training budget rose from 500 to 600 updates (full) and 800 (compact), set from Stage B's measured plateau length in existing artifacts; and evaluation used 384 episodes per setting at full scale to pay for it.

The preregistered 004B-S secondary — an identical supervised delay-class auxiliary loss given equally to every learned arm — was **not** run, because its trigger condition was not met. It was to fire only if *all* learned arms failed to beat LRU; `content_only` beat LRU on 3/3 seeds at both pressure levels. Running it anyway would have been moving a threshold after seeing the result.

## Scope

**What this establishes.** Under end-to-end predictive training with no future-use labels, a PHL-transported per-slot temporal lease does not improve retention over LRU, over a parameter-identical static priority, or over a 57-parameter learned score built from local recency and content features, at 8 slots and 12–16 writes on this benchmark. Its predicted relevance is uncorrelated with true next-use distance. Learned eviction itself does help, and the credit belongs to the simplest non-PHL control.

**What it does not establish.** It does not refute the temporal-lease idea in general: the lease head never learned the cue, so transport was never given a working input to transport. It does not test the brief's full-scale protocol at all — that run is a null caused by content-path learnability at 456 tokens. It does not establish anything about PHL-DAM versus Transformers, about attention, about scaling, about reasoning or code, or about asymptotic efficiency. Three seeds cannot settle a 5 pp effect, and the compact scale is a weaker test than the brief specified (delays 29–104 rather than 32–256, writes 8–16 rather than 16–32).

## Decision

**KILL LEASE HYPOTHESIS — in its current end-to-end form.** The mechanism fails its preregistered gate on five of six criteria and is beaten by the simplest available control. It should not be carried forward, scaled, or added to the PHL-DAM backbone on this evidence.

The one experiment that would still be informative, and that this result specifically motivates, is the supervised-predictor variant: give every learned arm an identical delay-class auxiliary loss, verify the lease head actually decodes the cue (AUROC well above 0.5), and only then ask whether transporting a *working* relevance estimate beats holding it static. If transport fails there too, the hypothesis is dead outright rather than merely unlearnable.

## Recommended next experiments

1. ~~Establish a training budget at which the model reaches breakthrough at 456
   tokens~~ — **done, negative**: 1400 updates is not enough, and the blocker is
   write count rather than length (PHL-DAM-004C).
2. ~~004B-S: equal supervised delay-class loss for all learned arms, with lease
   AUROC as a gating precondition~~ — **done**: the timing gate passed at the
   Bayes ceiling and transport still lost to static priority
   (`PHL_DAM_004BS_Supervised_Lease_Report.md`). Lease hypothesis killed.
3. Independently, pursue `content_only` — learned local eviction is the finding
   that actually replicated 3/3 and deserves its own attribution study against
   LRU and Belady.

## Reproduction

```
cd C:\Users\liamp\Documents\Codex\2026-08-28\tes\work
py -3.14 -m unittest -v

# Full scale (the null)
py -3.14 phl_dam_004b_lease.py --arm ARM --seed N --steps 600 --eval-episodes 384 \
    --output ..\outputs\phl_dam_004b_ARM_seedN.json
py -3.14 phl_dam_004b_aggregate.py --seeds 0 1 --output ..\outputs\phl_dam_004b_fullscale_aggregate.json

# Scale ladder (why the null)
py -3.14 phl_dam_004b_scale_ladder.py A 500
py -3.14 phl_dam_004b_scale_ladder.py B 500

# Compact scale (the lease test)
py -3.14 phl_dam_004b_lease.py --arm ARM --scale compact --seed N --steps 800 \
    --eval-episodes 512 --output ..\outputs\phl_dam_004b_compact_ARM_seedN.json
py -3.14 phl_dam_004b_aggregate.py --prefix phl_dam_004b_compact_ --seeds 0 1 2 \
    --output ..\outputs\phl_dam_004b_compact_aggregate.json
```

ARM ∈ {content_only, static_priority, phl_lease, learned_utility}; N ∈ {0,1,2}.
Environment: Python 3.14.0, torch 2.13.0+cpu, CPU only, one thread per process.
