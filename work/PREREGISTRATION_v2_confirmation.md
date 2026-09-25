# PHL-DAM v2 — confirmation protocol (fixed before any seed-0–5 run)

Written after dev screening on seeds 100–102 and before running anything on the
reporting seeds. Nothing below may be changed after the confirmation runs start.

## Candidates (chosen on dev seeds 100–102 only)

| Name | Switches | Params | State floats |
|---|---|---:|---:|
| `v2_S` | fast, normalized_values, use_phl=False, d_model=76 | 32,146 | 392 |
| `v2_T` | fast, normalized_values, use_phl=False, d_model=84, num_slots=4, d_key=8, d_value=8 | 32,146 | 68 |

Both are reported whatever they score. No other variant is run on seeds 0–5.

## Baselines

`phl_dam` (Stage B, the published model), `ntm_dnc_factorized`, `ntm_dnc`,
`transformer`, `ssm_selective`, `ssm_diagonal` — unchanged code, parameters
33,025–33,139.

## Protocol

`phl_dam_learning_curve.py`: seeds 0–5, 500 steps, batch 16, AdamW lr 2e-3
wd 1e-4, clip 1.0, the unchanged `seed + 10_000` training stream; probe every
10 steps on 256 held-out episodes (`seed + 30_000`); final recall on the
standard 2,000-episode stream (`seed + 20_000`).

## Criteria — each judged per candidate against every baseline

1. **Accuracy.** Final recall: mean ≥ every baseline's mean. Against a
   baseline below 99%, the paired contrast must be `robust` under
   `paired_verdict`. Against a baseline at ceiling (≥ 99.5% mean), "beats" is
   not measurable; the claim is only "not worse", i.e. the paired mean
   difference is > −0.5 pp.
2. **Trains faster (steps).** Steps to 90% probe recall: mean fewer than every
   baseline, and fewer or equal on at least 5 of 6 paired seeds. A run that
   never reaches 90% counts as 510 steps.
3. **Trains faster (time).** Seconds to 90% from a separate interleaved
   timing benchmark (arms measured round-robin in one process so load drift
   cancels), times each arm's mean steps-to-90%.
4. **Less recall loss during training.** Mean training recall CE over all
   probe points (area under the recall-loss curve): lower than every baseline,
   robust paired contrast.
5. **Memory.** Inference state floats ≤ the DNC's 392 (`v2_S`), or below every
   baseline including the SSMs' 96 (`v2_T`). Peak training RSS is reported.

A criterion that fails is reported as failed.
