"""Question 3, part 2b: what if the sensors' range or detection rate is misjudged by 25%?

A planner picks the best ring radius using nominal values (range rho = 1, detection
number Da = 1). The real sensors then differ by +/-25% in range or in rate. For each
case the table reports:

    u_planned  the drone's best chance against the ring that was planned with the
               nominal values, when the true values apply
    u_best     the drone's best chance against the ring that would have been planned
               had the true values been known (radius re-optimized)
    regret     u_planned - u_best: protection lost by planning with the wrong numbers
    shift      how far the best radius moves (true best minus planned)

Rings of K = 4, 6 and 8 sensors, response distance d = 2, three drone types. The
planned radii come from results/q3_best_ring.csv.

Run:  PYTHONPATH=src python experiments/q3_sensitivity.py      (about 10 minutes)
Writes results/q3_sensitivity.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q3_best_ring import DA, DEPTH, PES, ROOT  # noqa: E402

from cdd import Scenario, ring_sensors, solve  # noqa: E402

D_RESP = 2.0
KS = (4, 6, 8)
CASES = {"range -25%": (0.75, 1.0), "range +25%": (1.25, 1.0),
         "rate -25%": (1.0, 0.75), "rate +25%": (1.0, 1.25)}   # (rho factor, rate factor)


def worst_u(k, a, pe, rho, rate):
    scn = Scenario(r0=D_RESP, R=D_RESP + DEPTH, v=1.0, sigma=np.sqrt(2.0 / pe),
                   lam0=DA * rate, rho=rho, sensors=ring_sensors(k, a),
                   kernel="gaussian", outer="reflect")
    return solve(scn, nr=400, ntheta=256).worst_entry()[0]


def main():
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        planned = {(int(r["K"]), r["drone"]): float(r["best_radius"])
                   for r in csv.DictReader(f) if float(r["response_distance"]) == D_RESP}

    rows = []
    t0 = time.time()
    print(f"{'K':>2} {'drone':22s} {'case':11s} {'u_planned':>9} {'u_best':>7} "
          f"{'regret':>7} {'shift':>6}")
    for k in KS:
        for label, pe in PES.items():
            a_plan = planned[(k, label)]
            for case, (rho, rate) in CASES.items():
                f = lambda a: worst_u(k, a, pe, rho, rate)
                u_plan = f(a_plan)
                lo, hi = max(0.3, a_plan - 1.2), min(D_RESP + DEPTH - 0.5, a_plan + 1.2)
                res = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                                      options={"xatol": 0.02})
                a_best, u_best = (res.x, res.fun) if res.fun < u_plan else (a_plan, u_plan)
                rows.append({"K": k, "drone": label, "Pe": pe, "case": case,
                             "planned_radius": a_plan, "true_best_radius": a_best,
                             "u_planned": u_plan, "u_best": u_best,
                             "regret": u_plan - u_best, "shift": a_best - a_plan})
                print(f"{k:2d} {label:22s} {case:11s} {u_plan:9.4f} {u_best:7.4f} "
                      f"{u_plan - u_best:7.4f} {a_best - a_plan:+6.2f}")
        print(f"   ({time.time() - t0:.0f} s)")

    with open(ROOT / "results" / "q3_sensitivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = max(rows, key=lambda r: r["regret"])
    print(f"\nLargest regret: {worst['regret']:.4f} (K = {worst['K']}, {worst['drone']}, "
          f"{worst['case']})")
    print("Saved results/q3_sensitivity.csv")


if __name__ == "__main__":
    main()
