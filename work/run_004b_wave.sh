#!/bin/sh
# One PHL-DAM-004B run per line of stdin: "<arm> <seed>". Six at a time.
run_one() {
  arm=$1; seed=$2
  py -3.14 phl_dam_004b_lease.py --arm "$arm" --seed "$seed" \
      --steps 600 --eval-episodes 384 \
      --output "../outputs/phl_dam_004b_${arm}_seed${seed}.json" \
      > "004b_${arm}_seed${seed}.log" 2>&1
}
