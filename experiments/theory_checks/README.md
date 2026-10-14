# Theory checks

Scripts that reproduce the numbers quoted in `paper/theory_section.tex`. Run from the repo root,
for example `python experiments/theory_checks/check_ring_straight_line.py`.

| Script | What it checks | Time |
|---|---|---|
| `check_ring_straight_line.py` | Straight-line ring exposure: series formula, worst angle at the midpoint, best radii (argmax G = 3.060, argmax M6 = 2.4528) | seconds |
| `free_layout_straight_K6.py` / `_K4.py` | Free-layout search in the straight-line limit from 45 starts: all converge to the regular ring | ~5 min each |
| `dual_measure_test.py` | Shows the uniform-midpoint dual measure cannot certify the ring (bound 2.449 vs 1.304) | ~1 min |
| `lower_bound_floor.py` | Rotational-average floor for random drones (0.410 / 0.268 / 0.199) | ~5 min |
| `check_round2_claims.py` | Local-max perturbation test, ring vs ring-plus-center for K = 3-9, collapse condition | ~5 min |
| `best_radius_vs_D.py` | Best ring radius against D from the PDE (small-D slope, large-D trend) | ~1 min |
