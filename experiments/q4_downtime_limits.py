"""Question 4, part 1: sensor downtime in its two exact limits, and the best ring under each.

Sensors are ON a fraction p of the time (availability). Two exact limits (cdd/downtime.py):

  fast      outages are brief compared with a flight: each sensor acts as ON at rate p*lam0
  frozen    outages outlast a flight: each sensor is ON or OFF for the whole approach.
            unaware drone: does not know which sensors are down, picks the entry point that
                           is best on average over outage patterns
            aware drone:   sees the outage (or probes for it) and then picks its entry

For K = 6 sensors on a ring, response distance d = 2, Da = 1 and the three drone types,
this script finds, for p = 0.9 and 0.7:
  - the drone's best chance at the always-on best ring radius under each scenario, and
  - the best ring radius for each scenario (scan, then bounded Brent), with its value.

Grid: coarse (200 x 132) for the search, fine (400 x 264) for reported values; the angle
counts are multiples of K so the 64 outage patterns reduce to 14 by rotation symmetry.

Run:  PYTHONPATH=src python experiments/q4_downtime_limits.py         (about 25 min, 1 core)
Resumable: every evaluated (drone, p, radius, scenario) is saved to results/checkpoints/.
Writes results/q4_downtime_limits.csv and figures/q4_downtime_limits.png.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint, make_key  # noqa: E402
from q3_best_ring import DA, PES, ROOT  # noqa: E402
from q3_free_layout import D_RESP, K, R_MAX, R_OUT, scenario_xy  # noqa: E402

from cdd import ring_sensors, solve  # noqa: E402
from cdd.downtime import frozen_edge, u_fast, u_frozen_parts_ring  # noqa: E402

AVAILABILITY = (0.9, 0.7)
SCENARIOS = ("always on", "fast", "frozen, unaware drone", "frozen, aware drone")
SCAN = np.linspace(1.0, R_MAX, 15)


def evaluate(a, pe, p, fine=False):
    """Drone's best chance for a ring of radius a under every scenario."""
    nr, nth = (400, 264) if fine else (200, 132)      # angles divisible by K = 6
    scn = scenario_xy(ring_sensors(K, a), pe)
    unaware_edge, aware = frozen_edge(u_frozen_parts_ring(scn, p, nr=nr, ntheta=nth))
    return {"always on": solve(scn, nr=nr, ntheta=nth).worst_entry()[0],
            "fast": u_fast(scn, p, nr=nr, ntheta=nth).worst_entry()[0],
            "frozen, unaware drone": float(unaware_edge.max()),
            "frozen, aware drone": aware}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    ck = Checkpoint("q4_downtime_limits", fresh=args.fresh)
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        ring = {r["drone"]: float(r["best_radius"]) for r in csv.DictReader(f)
                if float(r["response_distance"]) == D_RESP and int(r["K"]) == K}
    t0 = time.time()

    def cached(a, pe, p, fine=False):
        key = make_key(pe, p, round(float(a), 6), "fine" if fine else "coarse")
        if key not in ck:
            ck.add(key, evaluate(a, pe, p, fine))
        return ck[key]

    rows, curves = [], {}
    for label, pe in PES.items():
        for p in AVAILABILITY:
            base = cached(ring[label], pe, p, fine=True)
            scan_vals = {s: [cached(a, pe, p)[s] for a in SCAN] for s in SCENARIOS}
            curves[(label, p)] = scan_vals
            for s in SCENARIOS:
                i = int(np.argmin(scan_vals[s]))
                lo, hi = SCAN[max(i - 1, 0)], SCAN[min(i + 1, SCAN.size - 1)]
                res = minimize_scalar(lambda a: cached(a, pe, p)[s], bounds=(lo, hi),
                                      method="bounded", options={"xatol": 0.01})
                best_fine = cached(res.x, pe, p, fine=True)[s]
                rows.append({"drone": label, "Pe": pe, "availability": p, "scenario": s,
                             "u_at_always_on_best_radius": base[s],
                             "always_on_best_radius": ring[label],
                             "best_radius": res.x, "u_at_best_radius": best_fine,
                             "gain_from_reoptimizing": base[s] - best_fine,
                             "radius_shift": res.x - ring[label]})
                print(f"{label:22s} p={p:.1f} {s:22s}: u at always-on radius {base[s]:.4f}; "
                      f"best radius {res.x:.2f} ({res.x - ring[label]:+.2f}) -> u {best_fine:.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)

    with open(ROOT / "results" / "q4_downtime_limits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    colors = dict(zip(SCENARIOS, ("0.3", "C0", "C3", "C1")))
    fig, axes = plt.subplots(len(AVAILABILITY), len(PES), figsize=(4.4 * len(PES), 3.8 * len(AVAILABILITY)),
                             sharex=True)
    for i, p in enumerate(AVAILABILITY):
        for j, (label, pe) in enumerate(PES.items()):
            ax = axes[i, j]
            for s in SCENARIOS:
                ax.plot(SCAN, curves[(label, p)][s], "-", color=colors[s], label=s)
                best = next(r for r in rows if r["drone"] == label and r["availability"] == p
                            and r["scenario"] == s)
                ax.plot(best["best_radius"], best["u_at_best_radius"], "o", color=colors[s], ms=5)
            ax.set_title(f"{label}, availability {p:.0%}", fontsize=9)
            ax.set_ylabel("drone's best chance u")
            if i == len(AVAILABILITY) - 1:
                ax.set_xlabel("ring radius")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Sensor downtime: drone's chance against ring radius (K = {K}, response "
                 f"distance {D_RESP:g}, Da = {DA:g}); dots = best radius", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "q4_downtime_limits.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved results/q4_downtime_limits.csv and figures/q4_downtime_limits.png "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
