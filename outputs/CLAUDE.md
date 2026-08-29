# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The PHL-DAM research line: a staged, preregistered empirical investigation of whether
a **PHL backbone plus a Dynamic Associative Memory (DAM)** learns content-addressed
recall better than matched controls. It is not an application — every artifact here is
evidence for or against a hypothesis, and the reports are written to be falsifiable.

Two sibling directories:

- `..\work\` — the experiment scripts, their tests, and every raw run.
- `.\` (`outputs\`) — the published artifacts: one `*_Report.md` and one
  `*_Reproduction.zip` per experiment, plus the raw per-seed and aggregate JSON.

Nothing here is a git repository and there is no package metadata; the scripts are
flat top-level modules that import each other by name, so **always run them with
`..\work` as the working directory**.

## Interpreter

```
py -3.14      # C:\Users\liamp\AppData\Local\Programs\Python\Python314\python.exe
```

Python 3.14.0 with torch 2.13.0+cpu — the exact environment recorded in every report's
test-boundary section. The system default `python` is 3.11 and has no torch. All runs
are CPU-only and `phl_dam_stage_b.py` calls `torch.set_num_threads(1)` at import,
deliberately, for speed and reproducibility on this machine; do not remove it without
re-baselining.

## Commands

Run from `..\work`:

```
py -3.14 -m unittest -v                                       # all test_phl_dam_*.py
py -3.14 -m unittest test_phl_dam_stage_b -v                  # one module
py -3.14 -m unittest test_phl_dam_stage_b.StageBTests.test_no_temporal_lease_parameters_or_state

py -3.14 phl_dam_stage_a.py --seeds 0 1 2 --output stage_a.json
py -3.14 phl_dam_stage_b.py --seed 0 --output phl_dam_stage_b_results.json
py -3.14 phl_dam_stage_b.py --seed 1 --output phl_dam_stage_c_seed1.json          # Stage C = Stage B, new seeds
py -3.14 phl_dam_stage_b.py --seed 3 --write-budget-weight 0 --output no_write_budget_seed3.json
py -3.14 phl_dam_backbone_comparison.py --model dam_only|dam_only_matched|fast_weight --seed N --output OUT.json
py -3.14 phl_dam_transformer_rematch.py --seed N --output OUT.json
py -3.14 phl_dam_lease_pressure.py --seed 0 --output phl_dam_lease_pressure_smoke.json
py -3.14 phl_dam_004a_oracle.py --writes 24 --seeds 0 1 2 3 4 5 --episodes 2000 --output OUT.json
py -3.14 phl_dam_004a_oracle.py --scale compact --writes 16 --seeds 0 1 2 --output OUT.json
py -3.14 phl_dam_004b_lease.py --arm phl_lease --scale compact --seed 0 --steps 800 --output OUT.json
py -3.14 phl_dam_004b_aggregate.py --prefix phl_dam_004b_compact_ --seeds 0 1 2 --output OUT.json
```

Shared defaults across the learned-controller experiments: 500 steps, batch 16,
lr 2e-3, 2,000 held-out episodes (6,000 recall queries). Changing any of them
invalidates comparability with every existing result file — pass explicit flags for a
side experiment instead of editing the defaults.

## Experiment lineage

| Stage | Script | Question |
|---|---|---|
| A | `phl_dam_stage_a.py` | Does the memory primitive work at all, with an **oracle** supplying WRITE/QUERY/slot? |
| B | `phl_dam_stage_b.py` | Can a **learned** controller find writes and reads from tokens alone, one seed? |
| C | `phl_dam_stage_b.py`, seeds 1–2 | Does B repeat unchanged across seeds? (gate: mean ≥30%, min >15%) |
| Backbone | `phl_dam_backbone_comparison.py` | PHL+DAM vs DAM-only vs fast-weight/delta, 3 seeds |
| Attribution-002/003 | same, `--model dam_only_matched` | Is the gap PHL, or just PHL's extra parameters? 6 seeds |
| Rematch | `phl_dam_transformer_rematch.py` | Parameter-matched causal Transformer on the same streams |
| No-write-budget | Stage B with `--write-budget-weight 0` | Is the advantage an artifact of the write-budget penalty? |
| Lease-001 | `phl_dam_lease_pressure.py` | Learned temporal retention under slot pressure. **Superseded by 004A/004B**: no LRU control, `query_consumes_binding` made LRU meaningless, its cue decoded at 99.95%, and its "transport" was a hand-written countdown never attached to `PHLDAM` |
| 004A | `phl_dam_004a_oracle.py` | Does an oracle future-relevance policy beat LRU under real 8-slot capacity pressure? Gate: oracle − LRU ≥ 5 pp. **Verdict: ORACLE GAP PRESENT** (+34 to +41 pp, 6/6 seeds) |
| 004B | `phl_dam_004b_lease.py` | Can a PHL-transported per-slot temporal lease beat LRU *and* non-PHL learned controls inside the real finite-capacity PHL-DAM? **Verdict: KILL LEASE HYPOTHESIS** in its end-to-end form — fails 5 of 6 gate criteria; beaten 24.9 pp by a 57-parameter `content_only` control; lease AUROC 0.28–0.50 for future-needed vs never-needed |
| 004B ladder | `phl_dam_004b_scale_ladder.py` | Why the 456-token 004B run is a null: the same model reaches 99.18% recall at 176 tokens and never leaves chance at 456 within 600 updates |
| 004B-S | `phl_dam_004b_lease.py --timing-weight 1` | Given a timing predictor that provably works (AUROC 0.85 vs Bayes 0.849), does PHL transport beat a parameter-identical static priority? **Verdict: KILL LEASE HYPOTHESIS** — transport is -11.9 pp at W=16, +1.2 pp on the one clean seed, -36.8 pp on a seed where it collapsed into memory thrashing |
| 004D | `phl_dam_004d_write_pressure.py` | Write count varied at genuinely fixed sequence length (456), delays and query budget pinned. **Verdict: hypothesis NOT established** - 7/25 runs diverged including 2 at the W=8 baseline. Failure signature is loss and inversion of write selectivity (failure mode A); gradient explosion precedes gate failure 11:2. Fix the instability, then re-run |
| 004C | `phl_dam_004b_lease.py --scale dilate*` | What blocks learning at full scale? **Write count, not length or budget**: 451 tokens learns at 8/12/16 writes; 456 tokens at 16/24/32 fails at 1400 updates. Length and write count are confounded by construction |

The 004 line has two **scale profiles**, selected with `--scale` and set by
`phl_dam_pressure_task.set_scale`: `full` is the PHL-DAM-004 brief as written
(456 tokens, 32-256 delays, 16/24/32 writes) and `compact` is the same task at
176 tokens. The compact profile exists because of a measured fact: at 456
tokens the content controller never reaches its recall breakthrough within 600
updates, so every arm sits at chance and no eviction policy is measurable.
Any 004 result must state which scale produced it.

The 004 line does **not** import `phl_dam_stage_b.py`: it needs a different
vocabulary (a write-time `tag` token), a 456-token stream and finite capacity,
so it has its own generator in `phl_dam_pressure_task.py`. Stage B is therefore
untouched and every pre-004 result stands unchanged. 004A and 004B share that
generator and `phl_dam_eviction_policies.py`, so editing either changes both
stages at once.

Later stages import earlier ones rather than reimplementing: `phl_dam_stage_b.py`
owns `make_batch`, `common_objective`, `_gather_positions`, `seed_everything`,
`VOCAB_SIZE`, `SEQUENCE_LENGTH`, and `PHLDAM`; the backbone comparison and the
Transformer rematch both import from it. **Editing `phl_dam_stage_b.py` therefore
changes the protocol for every downstream experiment at once.** The existing reports
depend on its stream generator being byte-identical; if you must change it, re-run
everything downstream rather than mixing old and new numbers.

## Invariants the tests enforce

`test_phl_dam_*.py` are not smoke tests — they encode the preregistered claims, and
several would need to be deliberately deleted for a result to become invalid:

- **No temporal leases.** Every model through Attribution-003 asserts no parameter
  name contains `lease` and the state object has no `leases` attribute. The reports
  repeat this in prose. Lease work belongs in `phl_dam_lease_pressure.py` only.
- **Content-only.** `forward` receives tokens and nothing else — no WRITE/QUERY
  positions, slot labels, or controller targets. `test_forward_receives_tokens_only_and_is_finite`
  guards this.
- **Delays stay in the preregistered bins** 29–63 / 64–95 / 96–169, so distance
  tables remain comparable across experiments.
- **The retrieval-disabled control must be near chance** (10%, ten possible values).
  Every report states it; it is what separates real recall from a key-to-value shortcut.
- **Ablation switches must preserve the default path exactly.** When `use_phl=False`
  was added, the default PHL path kept deterministic step-1 equivalence, and the
  backbone report cites that check. Hold new switches to the same bar.
- **Parameter and recurrent-state counts are asserted numerically**
  (`recurrent_state_floats`, `active_parameter_count`) because the whole attribution
  argument turns on matching them: PHL+DAM 33,034 params / 456 floats vs
  dam_only_matched 33,098 / 392 vs Transformer 33,074 / 16,896.

## Result artifacts

Every run writes one JSON with the same top-level shape: `experiment`, `model`,
`configuration` (seed and full hyperparameters, including `phl_enabled` /
`lease_state_present`), `metrics` (recall, retrieval-disabled recall, address top-1,
CEs, per-distance-bin recall, finiteness), and the complete `training_history` per
step. Aggregates (`*_aggregate.json`) are hand-assembled roll-ups holding a `protocol`
block and per-model means/SDs — there is no aggregation script, so keep an aggregate's
`protocol` consistent with the per-seed `configuration` blocks it summarizes.

A `*_Reproduction.zip` bundles exactly the sources, tests, and result JSON needed to
re-derive one report; the unpacked `*_repro_check` / `*_repro_verified` /
`*_reproduction` directories in `..\work` are verification copies of those bundles,
not separate lines of work. When you produce a new result, the finished unit is
report + raw JSON + reproduction bundle, all three.

## Writing reports

Match the established voice: a bolded **Verdict** first, then a "Test boundary"
section listing the exact scope, then results tables, then explicit
**"What this establishes"** / **"What it does not establish"** sections. Existing
reports actively argue against their own headline — the backbone report says the
result "is not yet clean proof of a uniquely PHL-specific mechanism benefit" and names
the next control needed. Preserve that: state the confound, name the follow-up
experiment, and never round a screen up into a claim.
