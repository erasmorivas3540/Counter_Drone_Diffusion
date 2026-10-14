"""How much do the real-world conclusions depend on the assumed inputs?

real_world_mapping.py had to assume two inputs for which no published values exist: the time
to an actionable detection (sets Da) and the drone's maneuver statistics (set D, hence Pe).
This script halves and doubles each, independently (3 x 3 = 9 cases per scenario), and
reports for each scenario the range of:

    Pe and Da; the regime (wandering Pe < 1, mixed 1-20, directed > 20); the best ring radius;
    the drone's chance against it; and the cost of planning with a straight-line ring.

Run:  PYTHONPATH=src python experiments/real_world_sensitivity.py      (about 30 min, 1 core)
Resumable: each case is saved to results/checkpoints/ as it finishes.
Writes results/real_world_sensitivity.csv and results/real_world_sensitivity.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint, make_key  # noqa: E402
from q3_best_ring import ROOT  # noqa: E402
from real_world_mapping import A_STRAIGHT, SCENARIOS, analyse  # noqa: E402

FACTORS = (0.5, 1.0, 2.0)


def regime(pe):
    return "wandering" if pe < 1 else ("mixed" if pe <= 20 else "directed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    ck = Checkpoint("real_world_sensitivity", fresh=args.fresh)
    t0 = time.time()
    rows = []
    for sid, drone, sensor, v, rho, sv, tau, t_det in SCENARIOS:
        for f_det in FACTORS:
            for f_D in FACTORS:
                D = f_D * sv**2 * tau
                lam0 = 1.0 / (f_det * t_det)
                pe, da = v * rho / D, lam0 * rho / v
                key = make_key(sid, f_det, f_D)
                if key not in ck:
                    a, u, u_s, method = analyse(pe, da)
                    ck.add(key, {"a": a, "u": u, "u_straight": u_s, "method": method})
                r = ck[key]
                rows.append({"id": sid, "drone": drone, "sensor": sensor,
                             "detection_time_factor": f_det, "maneuver_D_factor": f_D,
                             "Pe": pe, "Da": da, "regime": regime(pe),
                             "best_ring_radius_m": r["a"] * rho,
                             "straight_line_ring_radius_m": A_STRAIGHT * rho,
                             "drone_chance": r["u"], "drone_chance_straight_ring": r["u_straight"],
                             "cost_of_straight_planning": r["u_straight"] - r["u"],
                             "method": r["method"]})
        print(f"{sid} done ({time.time() - t0:.0f} s)", flush=True)

    with open(ROOT / "results" / "real_world_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Sensitivity of the real-world scenarios to the assumed inputs", "",
             "Time to an actionable detection and maneuver diffusion each halved and doubled "
             "(9 cases per scenario). Ranges are min-max over the 9 cases.", "",
             "| ID | Scenario | Pe | Da | Regimes seen | Best ring (m) | Drone's chance | "
             "Cost of straight-line ring |",
             "|---|---|---|---|---|---|---|---|"]
    for sid, drone, sensor, *_ in SCENARIOS:
        sub = [r for r in rows if r["id"] == sid]

        def rng(col, fmt):
            vals = [r[col] for r in sub]
            return f"{format(min(vals), fmt)}–{format(max(vals), fmt)}"
        regimes = sorted({r["regime"] for r in sub}, key=["wandering", "mixed", "directed"].index)
        lines.append(f"| {sid} | {drone} vs {sensor} | {rng('Pe', '.2g')} | {rng('Da', '.2g')} | "
                     f"{', '.join(regimes)} | {rng('best_ring_radius_m', '.0f')} | "
                     f"{rng('drone_chance', '.3f')} | {rng('cost_of_straight_planning', '.3f')} |")
        print(lines[-1])
    (ROOT / "results" / "real_world_sensitivity.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved results/real_world_sensitivity.csv and results/real_world_sensitivity.md "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
