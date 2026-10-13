"""Step 2: check points on the regime map with the Monte Carlo simulator.

For each chosen (Pe, Da), the PDE gives the worst-case entry point on the area edge
and u there. The simulator starts drones at that same point and estimates u
independently. The script reports:

  * PDE vs Monte Carlo (z-score): do the two methods agree?
  * Monte Carlo vs the straight-line value: is the sign of the effect (helps or
    hurts) clear from the simulator alone, i.e. does its 95% interval exclude
    the straight-line value?

Points: two where the map says randomness helps a lot, two where it hurts the most,
and two near the boundary between them. Uses the same layout as regime_map.py.

Run:  PYTHONPATH=src python experiments/regime_spotcheck.py      (about 1–2 minutes)
Writes results/regime_spotcheck.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regime_map import ROOT, R, scenario, worst_straight  # noqa: E402

from cdd import simulate_u, solve  # noqa: E402

POINTS = [  # (Pe, Da, what the map says)
    (0.10, 0.77, "helps (strong)"),
    (0.19, 1.29, "helps (strong)"),
    (2.31, 0.17, "hurts (strongest)"),
    (1.23, 0.28, "hurts"),
    (0.66, 0.28, "near the boundary"),
    (4.33, 0.77, "near the boundary"),
]
N_PATHS = 40_000


def main():
    rows = []
    print(f"{'Pe':>6} {'Da':>6}  {'map says':18} {'PDE':>7} {'Monte Carlo':>17} {'z':>6}"
          f" {'straight':>8}  sign clear?")
    for k, (pe, da, label) in enumerate(POINTS):
        t = time.time()
        scn = scenario(pe, da)
        u_pde, angle = solve(scn, nr=400, ntheta=256).worst_entry()
        x0 = R * np.array([np.cos(angle), np.sin(angle)])
        # keep each random step small next to a sensor footprint: sigma*sqrt(dt) <= 0.14
        dt = min(1e-2, 0.02 / scn.sigma**2)
        res = simulate_u(x0, scn, dt=dt, n_paths=N_PATHS, seed=300 + k)
        straight = worst_straight(scn)
        lo, hi = res.interval
        clear = "yes" if (lo > straight or hi < straight) else "no (too close to call)"
        rows.append({"Pe": pe, "Da": da, "map_says": label, "u_pde": u_pde,
                     "u_monte_carlo": res.estimate, "half_width_95": res.half_width,
                     "z": res.z_score(u_pde), "u_straight": straight,
                     "sign_clear": clear == "yes"})
        print(f"{pe:6.2f} {da:6.2f}  {label:18} {u_pde:7.4f} {res.estimate:9.4f} +/- "
              f"{res.half_width:.4f} {res.z_score(u_pde):+6.2f} {straight:8.4f}  {clear}"
              f"   ({time.time() - t:.0f} s)")
    print("\n|z| < 3.5 everywhere means the simulator confirms the PDE map.")
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "regime_spotcheck.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results/regime_spotcheck.csv")


if __name__ == "__main__":
    main()
