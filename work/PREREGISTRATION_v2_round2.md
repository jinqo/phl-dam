# PHL-DAM v2 — round 2 confirmation (fixed before any seed-6–11 run)

Round 1 (seeds 0–5, `PREREGISTRATION_v2_confirmation.md`) found v2_S tied with
the factorized DNC. A component diagnostic then showed the breakout waits on
the output layer learning to decode retrieved values; the copy readout
(`copy_readout`) removes that wait. Candidates were chosen on dev seeds
100–105 only. Seeds 0–5 are spent; this round uses fresh seeds 6–11.

## Candidates

| Name | Switches | Params | State |
|---|---|---:|---:|
| `v2_TC` | fast, normalized_values, use_phl=False, d_model=84, num_slots=4, d_key=8, d_value=8, copy_readout | 32,147 | 68 |
| `v2_SC` | fast, normalized_values, use_phl=False, d_model=76, copy_readout | 32,147 | 392 |

Both are reported whatever they score.

## Baselines (pass/fail is judged against these)

`phl_dam` (published Stage B), `ntm_dnc_factorized`, `ntm_dnc`,
`transformer`, `ssm_selective`, `ssm_diagonal` — unchanged.

## Reported control (not a pass/fail baseline, reported with equal prominence)

`ntm_dnc_factorized_copy`: the factorized DNC given the same copy readout. It
tests whether the gain is PHL-DAM's or the readout's. Dev seeds say the
readout's: this control reached 90% in a mean 27 steps against v2's 47–62.

## Protocol

Unchanged from round 1: `phl_dam_learning_curve.py`, seeds 6–11, 500 steps,
batch 16, AdamW lr 2e-3 wd 1e-4, clip 1.0, stream `seed + 10_000`, probe every
10 steps on 256 episodes (`seed + 30_000`), final recall on 2,000 episodes
(`seed + 20_000`). Timing from the round-robin benchmark.

## Criteria (per candidate, against every baseline)

1. **Accuracy** — as round 1: mean final recall ≥ baseline and robust paired
   win below 99%; at a ceiling baseline (≥ 99.5%), paired mean difference
   > −0.5 pp.
2. **Steps to 90%** — mean fewer, and fewer-or-equal on ≥ 5 of 6 seeds.
   A run that never reaches 90% within 500 steps counts as 510.
3. **Seconds to 90%** — FIXED from round 1: an arm with any seed that never
   reaches 90% has infinite time-to-90%, so it can never beat an arm that
   always does. Otherwise mean steps-to-90% × median seconds per step.
4. **Training recall loss** — mean training recall CE over probe points lower,
   robust paired contrast.
5. **Memory** — `v2_TC` state below every baseline (< 96); `v2_SC` state ≤ 392.
