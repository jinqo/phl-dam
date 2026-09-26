# PHL-DAM v3 — round 3 confirmation (fixed before any confirmation run)

Round 2 confirmed v2_TC on the 176-token benchmark, but two things were left
standing: a factorized DNC given the same copy readout was faster, and under
write pressure the factorized DNC won on final recall. Dev work after round 2
(seeds 100–105 at 176 tokens, 100–101 on the pressure task only):

* traced every gradient spike under pressure to `F.normalize` on zero slot
  keys (amplification exactly 1e12) → `key_norm_epsilon`;
* measured that trained v2 never writes 21% of the bindings it later needs,
  because Stage B's allocation smears overflow writes across every slot;
* replaced Stage B's merge-or-allocate rule with the DNC's write addressing
  (`dnc_write_addressing`). Dev: beats the factorized DNC at W=20 (83.1 /
  85.6% vs 77.2 / 79.8%) and W=32 (73.2 vs 69.6%), and at 176 tokens reaches
  90% in 20 steps vs the copy-DNC's 27.

The write addressing is the DNC's, and is credited as such.

## Candidate — one architecture for both tasks

`v3`: fast, normalized_values, use_phl=False, copy_readout,
key_norm_epsilon=0.05, dnc_write_addressing, d_model=76, 8 slots × (24+24).
32,301 parameters / 392 state floats at 176 tokens; 38,421 / 392 on the
pressure vocabulary.

Secondary, 176 tokens only: `v3_TC` (same switches, d_model 84, 4 slots ×
(8+8)): 32,317 parameters, 68 state floats.

## Part A — 176-token benchmark, fresh seeds 12–17

Protocol identical to round 2 (500 steps, probe every 10, 2,000-episode
final eval, round-robin timing). Opponents: `phl_dam`, `ntm_dnc_factorized`,
`ntm_dnc`, `transformer`, `ssm_selective`, `ssm_diagonal`, **and
`ntm_dnc_factorized_copy`**, which is now a full opponent rather than a
reported control.

Criteria per candidate against every opponent — identical to round 2
(accuracy; steps-to-90% mean fewer and ≤ on ≥ 5/6 seeds; seconds-to-90% with
never = ∞; training recall-loss area lower and robust; memory: `v3` ≤ 392,
`v3_TC` < 96).

## Part B — write-pressure ladder, seeds 0–4, W ∈ {8, 16, 20, 24, 32}

`phl_dam_ntm_pressure.py`, 700 steps, batch 16, 192 evaluation episodes,
learned = final training recall CE < 2.0. Opponents: PHL-DAM (004G, frozen),
factorized DNC (004I, frozen), factorized DNC + copy readout (run now, same
seeds). No v3 variant was run on seeds 0–4 before this file was written.

Criteria for `v3` against each opponent, at each level:

1. learned count ≥ the opponent's;
2. mean recall higher, with the paired contrast (`paired_verdict`, threshold
   5 pp) reported; "beats" at a level requires a higher mean *and* no robust
   paired loss;
3. mean breakthrough step (first logged recall CE < 2.0) earlier.

The ladder is won against an opponent only if criteria 1–3 hold at every
level. Every level and every criterion is reported, pass or fail.
