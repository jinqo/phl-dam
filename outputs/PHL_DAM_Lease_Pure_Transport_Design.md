# Design — pure transport attribution test (not yet run)

## Why this design exists

Every lease result so far compares *independently trained* models. `phl_lease`
and `static_priority` share an architecture and a parameter count but not their
weights, so any difference confounds three things:

1. the transport mechanism itself,
2. how the content path co-adapted to that mechanism during training,
3. training-path luck — in PHL-DAM-004B-S the mean was dominated by one seed
   where `phl_lease` collapsed.

The negative results so far are therefore sufficient to decline adoption, but
not to establish that transport itself is useless. This design removes all three
confounds and asks only the causal question:

> Given the exact same representations and the exact same timing predictions,
> does temporal transport itself improve eviction decisions?

## Protocol

1. **Train one content backbone.** Any arm whose eviction score does not consume
   the lease — `content_only` is the natural choice — trained to convergence at
   a scale where it reliably learns.
2. **Train one timing predictor.** The shared `lease_head`, supervised on the
   true next-use horizon, until its live-vs-never AUROC is within a preregistered
   tolerance of the Bayes ceiling.
3. **Freeze both.** No further gradient reaches either.
4. **Duplicate the checkpoint** so both evaluation arms start bit-identical.
   A test should assert byte equality of both state dicts before divergence.
5. **Evaluate two eviction rules on that single frozen model:**
   * static timing priority — the predictor's output held constant from write,
   * transported timing priority — the same output evolved by the horizon
     lattice.
6. **Allow only the transport operator to differ.** Same weights, same episodes,
   same seeds, same everything else.

## Why this is stronger than 004B-S

The contrast becomes a within-model A/B on identical weights and identical
predictions, so it is paired at the episode level rather than the seed level.
That removes training-path variance entirely, which is what made the 004B-S
bootstrap intervals span zero, and it allows many more paired samples (one per
eviction decision) than three seeds can provide.

## Preregistered decision rule

Do not resume development of lease transport unless this test produces a
meaningful positive result: transported priority must beat static priority by a
margin fixed *before* running, on a majority of episodes as well as in the mean,
with a bootstrap interval excluding zero.

If it does not, the lease/transport line is dead outright rather than merely
unlearnable, and the surviving research direction is learned local memory
management (see PHL-DAM-004D).

## Cost

One backbone training run plus one predictor training run, then two evaluation
passes. Substantially cheaper than any of 004B, 004B-S or 004D, because nothing
is trained per arm.

## Status

**Not run.** Deliberately excluded from PHL-DAM-004D, whose subject is the write
controller rather than the lease.
