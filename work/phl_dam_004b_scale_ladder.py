"""Diagnostic ladder: which factor blocks the PHL-DAM-004B breakthrough?

Runs the 004B model on progressively Stage-B-like versions of the 004 task.
Stage B (3 bindings, 176 tokens, 8 slots, never full) reaches its recall
breakthrough around step 300. If a rung here also fails, the blocker is not
capacity and not the lease - it is the content controller's reach on this
task shape.

Usage: ladder.py RUNG STEPS
  RUNG=A  176 tokens, 3 writes   (closest to Stage B)
  RUNG=B  176 tokens, 8 writes   (memory exactly full, no over-subscription)
  RUNG=C  456 tokens, 8 writes   (full length, no over-subscription)
"""

import sys
import torch

import phl_dam_pressure_task as task
import phl_dam_004b_lease as m

RUNG = sys.argv[1]
STEPS = int(sys.argv[2])

if RUNG in ("A", "B"):
    task.SEQUENCE_LENGTH = 176
    task.WRITE_REGION_END = 40
    task.MIN_DELAY, task.MAX_DELAY = 29, 110
    task.FIRST_USE_DELAY = {
        task.NEAR: (29, 45),
        task.SHORT: (46, 70),
        task.MEDIUM: (71, 90),
        task.FAR: (91, 110),
        task.PERSISTENT: (35, 60),
    }
    task.PERSISTENT_GAP = (30, 50)
    task.DELAY_BINS = (("29-55", 29, 55), ("56-85", 56, 85), ("86-110", 86, 110))
    writes = 3 if RUNG == "A" else 8
    task.QUERY_BUDGET = {
        "canonical": {3: (2, 3), 8: (4, 7)},
        "spec": {3: (2, 3), 8: (4, 7)},
    }
else:
    writes = 8
    task.QUERY_BUDGET["canonical"][8] = (4, 7)

task.PRESSURE_LEVELS = (writes,)
m.EVAL_SETTINGS = (("canonical", writes),)
print(
    f"rung={RUNG} T={task.SEQUENCE_LENGTH} writes={writes} steps={STEPS} "
    f"delay={task.MIN_DELAY}-{task.MAX_DELAY}",
    flush=True,
)

model, history = m.train("phl_lease", 0, STEPS, 16, 2e-3, torch.device("cpu"))
row = m.evaluate_arm(
    model, "phl_lease", seed=0, writes=writes, condition="canonical",
    episodes=256, batch_size=16, device=torch.device("cpu"),
)
given = row["recall_given_resident"]
print(
    f"RESULT rung={RUNG} recall={row['recall']:.4f} residency={row['residency']:.4f} "
    f"r|res={given if given is None else round(given, 4)} "
    f"recall_ce={row['recall_token_ce']:.4f} "
    f"wgate_bind={row['controller']['mean_write_gate_at_binding']:.4f} "
    f"wgate_else={row['controller']['mean_write_gate_elsewhere']:.4f}",
    flush=True,
)
