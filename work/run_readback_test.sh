#!/bin/sh
# Does read-back consistency break the W>=20 wall?
#
# Preregistered bar, fixed before any run:
#   The wall is "broken" only if readback lifts learned-runs at W=20 from the
#   observed 0/5 to >= 2/5, AND does not reduce learned-runs at W=8 or W=16.
#   A single learned run at W=20 is noise, not a result.
#   If W=20 stays at 0/5 or 1/5, the invention failed and gets reported as such.
for w in 8 16 20 24; do
  for s in 0 1 2 3 4; do echo "$w $s"; done
done | xargs -P 6 -n 2 sh -c '
  py -3.14 phl_dam_004d_write_pressure.py --writes $0 --seed $1 --readback-weight 1.0 \
    --steps 700 --eval-episodes 192 \
    --output ../outputs/phl_dam_004i_readback_w$0_seed$1.json > 004i_w$0_s$1.log 2>&1'
echo "READBACK LADDER COMPLETE"
