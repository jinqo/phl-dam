#!/bin/sh
# PHL-DAM backbone attribution, post-stability-fix.
#
# Preregistered question: now that the gradient instability is fixed, does the
# PHL horizon lattice still earn its place? The existing attribution runs showed
# its contribution was optimisation reliability (6/6 vs 3/6 seeds), not peak
# accuracy - when PHL-off succeeded it matched PHL exactly. If the spread floor
# supplied that reliability, PHL may now be redundant.
#
# Preregistered bar, fixed before running:
#   * "earns its place" = beats `none` by >= 1 learned run out of 5 at BOTH
#     W=8 and W=16, or by >= 5 pp mean recall at either.
#   * `ssm` replaces PHL if it matches or beats PHL on both counts.
#   * ties go to the SIMPLER model: none > ssm > phl.
for bb in phl ssm none; do
  for w in 8 16; do
    for s in 0 1 2 3 4; do
      echo "$bb $w $s"
    done
  done
done | xargs -P 6 -n 3 sh -c '
  py -3.14 phl_dam_004d_write_pressure.py --backbone $0 --writes $1 --seed $2 \
    --steps 700 --eval-episodes 192 \
    --output ../outputs/phl_dam_004h_backbone_$0_w$1_seed$2.json \
    > 004h_$0_w$1_s$2.log 2>&1'
echo "BACKBONE COMPARISON COMPLETE"
