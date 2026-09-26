# v5 = v4 + address-based erase — confirmation (fixed before any seed-0–4 run)

Dev (pressure seeds 100–101, 176-token seeds 100–102): a v4 diagnostic at
W=32 showed v4 loses its most recent bindings (45–48% recall with no
intervening write vs 78–83% for the oldest) while the DNC shows the opposite
profile; an address-based erase gate initialised mostly closed (bias -2) gave
+0.5 to +3.2 pp at W=32 with unchanged breakthrough and unchanged 176-token
behaviour. Two dev seeds cannot distinguish that from noise.

Test: v5 (`erase_gate=True, erase_bias=-2.0`, otherwise v4) on seeds 0–4 at
W=24 and W=32, 700 steps, `--log-every 5`, paired against v4's round-4 runs
on the same seeds.

v5 is adopted only if, at both levels: mean recall is higher than v4's, the
paired contrast is not a robust loss, and mean breakthrough is not later than
v4's. Otherwise v4 stays the recommended architecture and v5 is reported as
not adopted.
