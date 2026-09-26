# PHL-DAM v4 — round 4 confirmation (fixed before any confirmation run)

Round 3 left one opponent unbeaten: the factorized DNC with a copy readout.
v3 beat it on recall at every pressure level and on every 176-token speed
measure, but under write pressure it broke through as early as v3, and v3's
176-token training-loss lead (0.017) was under the 0.05 threshold.

Dev work since (pressure seeds 100–101 only; 176-token seeds 100–102 only):

* At 5-step logging resolution the copy DNC genuinely breaks through earlier
  than v3 under pressure (steps 25–45 vs 35–50) — the round-3 tie was not a
  logging artefact.
* A screen of write-gate init, learned read temperature, tied query/key,
  copy-scale init and key bias found copy_scale 2.0 + tie_query_key moves
  v3's breakthrough to steps 20–30 at W=24/32 (copy DNC 35–45), and at 176
  tokens to 10 steps (v3: 20) with training recall-loss area 0.002–0.013.

## Candidate

`v4` = v3 + `copy_scale=2.0` + `tie_query_key=True`: fast, normalized_values,
use_phl=False, copy_readout, key_norm_epsilon=0.05, dnc_write_addressing,
d_model=76, 8 slots × (24+24), 392 state floats. One architecture, both tasks.

## Part A — 176 tokens, fresh seeds 18–23

Opponents: phl_dam, ntm_dnc_factorized, ntm_dnc_factorized_copy, ntm_dnc,
transformer, ssm_selective, ssm_diagonal. Protocol and criteria **identical to
round 3** (including the 0.05 decision threshold on training recall loss).
Memory criterion: state ≤ 392.

## Part B — pressure ladder, seeds 0–4, W ∈ {8, 16, 20, 24, 32}

No v4 run has used seeds 0–4. Criteria **identical to round 3** (learned count
≥; higher mean recall with no robust paired loss; mean breakthrough strictly
earlier), with one measurement change applied to every arm: breakthrough is
the first training step with recall CE < 2.0 **logged every 5 steps** instead
of every 25.

* v4: full 700-step runs with `--log-every 5`.
* Factorized DNC and copy DNC: final recall from their existing 700-step runs
  (004I / 004K, unchanged); breakthrough from 150-step reruns with
  `--log-every 5` on the same seeds (training is deterministic, so the first
  150 steps are the same trajectory — checked before use by matching the
  step-25..150 losses to the stored runs).
* Original PHL-DAM: existing 004G runs; its breakthroughs are ≥ step 200 or
  never, so 25-step resolution cannot change any comparison with v4.

The ladder is won against an opponent only if all three criteria hold at
every level. Every result is reported, pass or fail.
