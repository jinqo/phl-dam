# PHL-DAM

A staged, preregistered empirical investigation of **PHL-DAM** — a recurrent
sequence model that carries a small multi-timescale context state alongside a
bank of eight explicit key→value memory slots addressed by content.

Every artifact here is evidence for or against a hypothesis. The reports are
written to be falsifiable, several of them argue against their own headline, and
the mechanisms that failed are documented as carefully as the ones that worked.

**Author:** Liam Pattiata — research programme, hypotheses, experimental design,
preregistered gates, and all decisions about what to test and what counts as
evidence.

Experiment code, test suites and run execution were carried out with AI
assistance under that direction. See `NOTICE`.

---

## Headline result

On a synthetic associative-recall benchmark at 176 tokens — store three
arbitrary key→value bindings, retrieve by exact key after a 29–169 token delay —
at matched parameter count, identical streams, objective, optimiser and budget,
six paired seeds:

| Model | Params | Inference state @176 | Recall | Recall CE |
|---|---:|---:|---:|---:|
| **PHL-DAM** | 33,034 | 456 | **99.55% ± 1.07** | **0.016** |
| Causal Transformer | 33,074 | 16,896 | 47.08% ± 4.47 | 1.226 |
| Selective SSM (S6-style) | 33,025 | 96 | 27.77% ± 0.56 | 1.892 |
| Diagonal SSM (S4D-style) | 33,139 | 96 | 25.94% ± 0.88 | 1.981 |

PHL-DAM wins 6/6 paired seeds against every baseline. Disabling memory
retrieval drops it to 9.99% — chance — so the memory path carries the result
rather than a shortcut.

## How to read that table honestly

This is the part most write-ups leave out, and leaving it out would be the
fastest way to have the work dismissed.

1. **The benchmark rewards exactly this architecture.** It asks a model to store
   arbitrary bindings and retrieve one by content. PHL-DAM has literal key-value
   slots; the baselines must reconstruct that from a compressed state.
   Associative recall is the textbook known weakness of state-space models — the
   selective variant exists precisely because fixed-decay SSMs fail it, and it
   recovered only +1.8 pp here. A model with addressable memory beating models
   without it on an addressing task is close to the expected outcome.
2. **PHL-DAM spends 4.75× more state than the SSMs** (456 floats vs 96). On the
   constant-small-state axis, the SSMs win comfortably.
3. **It does not train at the next scale up.** At 456 tokens with 16–32 writes
   per episode it sits at chance. See the write-pressure ladder below.
4. **It has never seen natural data** — no language, no code. At 33K parameters
   this is a research probe, not a model.

The comparison that would genuinely discriminate is against **NTM/DNC**, which
also has addressable memory and which PHL-DAM's working core closely resembles.
That baseline is implemented in `work/phl_dam_ntm_baseline.py`; results pending.

## What was tested and rejected

| Mechanism | Verdict |
|---|---|
| Content-addressed DAM slots | **Kept** — ablate retrieval and recall falls to chance |
| Write-budget regulariser | **Removed** — coefficient 0 performs as well or better |
| PHL temporal lease transport | **Rejected** — failed its preregistered adoption gate |
| PHL horizon backbone | **Under test** — its measured contribution is optimisation reliability (6/6 vs 3/6 seeds), not accuracy |

The temporal lease was the architecture's most distinctive idea: attach to each
slot a distribution over "when will this matter" horizons and evolve it as time
passes. It was given every chance — a timing predictor trained to the Bayes
ceiling (AUROC 0.85 vs 0.849 optimal) and a transport operator calibrated to the
task's own delays — and still lost to simply holding the prediction static. It
is not carried forward. See `outputs/PHL_DAM_004BS_Supervised_Lease_Report_v2.md`.

## Where the architecture stops working

A write-pressure ladder at **fixed** sequence length (456 tokens), fixed delays
and fixed query budget, varying only the number of write events competing for
eight slots:

| Writes per episode | Runs that learned |
|---:|---:|
| 8 | 4/5 |
| 16 | 3/5 |
| 20 | 0/5 |
| 24 | 0/5 |

A cliff between 16 and 20 writes — roughly 2× slot capacity — not a gradual
decline. Failing runs show *negative* write selectivity: the write gate ends up
firing more often away from bindings than at them.

## Diagnosed and fixed: an intermittent gradient explosion

Gradients reached 1e19 and 28% of runs diverged. Located by bisection to a
z-score standardisation of the eviction score, whose Jacobian grows as
`1/spread` and reached ~9.8e5 whenever all eight slots scored alike — about one
timestep in 450, which matched the intermittent signature exactly. Flooring the
spread bounds the Jacobian at 8.75 and leaves the eviction ordering unchanged
wherever the spread is informative.

Result: **0/10 divergences** where the baseline had 3/10, and learning roughly
doubled at 8 and 16 writes. Three earlier hypotheses were tested and refuted
first; all are documented.

## Layout

```
work/      experiment scripts, their tests, and every raw run
outputs/   published reports, result JSON, reproduction bundles
```

Scripts are flat modules that import each other by name — **run them with
`work/` as the working directory**. Interpreter: `py -3.14` (Python 3.14,
torch 2.13.0+cpu, CPU only).

```
cd work
py -3.14 -m unittest -v          # ~180 tests
```

Each experiment ships a `*_Report.md`, its raw per-seed JSON, and a
`*_Reproduction.zip` bundling the exact sources, tests and results needed to
re-derive it. Superseded reports keep a banner pointing at their replacement
rather than being deleted.

## Licence

- **Code** — GPL-3.0-or-later (`LICENSE`). Full canonical licence text
  included. Anyone who distributes this or a modified version must release
  their corresponding source under the same terms.
- **Reports and result data** — CC BY-NC-SA 4.0 (`LICENSE-DOCS`), licensed by
  reference to the canonical URL as Creative Commons recommends. Attribution
  required, non-commercial, share-alike.

If you would rather close the network-service gap — GPL obliges source release
on *distribution*, whereas AGPL also obliges it when a modified version is run
as a hosted service — swap `LICENSE` for the canonical text at
https://www.gnu.org/licenses/agpl-3.0.txt and change the SPDX identifier to
`AGPL-3.0-or-later`. GPL-3.0 was used here because a byte-accurate copy was
available to verify; a licence with subtly wrong wording is weaker than none.

Note what a licence can and cannot do: it governs *reuse* of this code and these
documents. It cannot stop anyone reading the method or independently
reimplementing the ideas — ideas and experimental findings are not copyrightable
anywhere. Anything published publicly is readable by definition. The protection
a public repository does give you is a timestamped, attributable public record
of authorship, which is what establishes priority.
