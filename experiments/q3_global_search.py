"""Question 3, part 2c: a global search over free sensor positions.

q3_free_layout.py used a local optimizer, which only explores near its starting layout.
This script uses differential evolution, a global method that starts from a population
of random layouts spread over the whole area, then refines the best one with Powell.
It does NOT start from the ring, so finding a layout at least as good as the best ring
is independent evidence about whether the ring is optimal.

Variables: radius and angle of each of K = 6 sensors (sensor 0 fixed at angle 0, since
rotating a whole layout changes nothing), 11 numbers. Objective: the drone's best chance,
the worst-case u over all entry points, on the coarse grid (200 x 128); final layouts are
re-scored on the fine grid (400 x 256). Response distance d = 2, Da = 1, three drone types.

Run:  PYTHONPATH=src python experiments/q3_global_search.py      (about 30-40 minutes)
Needs results/q3_best_ring.csv. Writes results/q3_global_search.csv and
figures/q3_global_search.png.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q3_free_layout import D_RESP, K, R_MAX, R_MIN, R_OUT, worst_u  # noqa: E402
from q3_best_ring import DA, PES, ROOT  # noqa: E402

from cdd import ring_sensors  # noqa: E402

SEED = 2026
MAXITER = 40          # generations; population = 10 x 11 = 110 layouts per generation


def to_xy(p):
    """p = [r_0..r_{K-1}, angle_1..angle_{K-1}]; sensor 0 sits at angle 0."""
    radii, angles = p[:K], np.r_[0.0, p[K:]]
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def objective(p, pe):
    return worst_u(to_xy(p), pe)


def main():
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        best_ring = {r["drone"]: float(r["best_radius"]) for r in csv.DictReader(f)
                     if float(r["response_distance"]) == D_RESP and int(r["K"]) == K}

    bounds = [(R_MIN, R_MAX)] * K + [(0.0, 2 * np.pi)] * (K - 1)
    rows, found = [], {}
    t0 = time.time()
    for label, pe in PES.items():
        ring_u = worst_u(ring_sensors(K, best_ring[label]), pe, fine=True)
        de = differential_evolution(objective, bounds, args=(pe,), seed=SEED, maxiter=MAXITER,
                                    popsize=10, tol=1e-4, init="latinhypercube",
                                    polish=False, updating="deferred", workers=2)
        local = minimize(objective, de.x, args=(pe,), method="Powell", bounds=bounds,
                         options={"maxfev": 800, "xtol": 1e-3, "ftol": 1e-5})
        x = local.x if local.fun < de.fun else de.x
        sensors = to_xy(x)
        u = worst_u(sensors, pe, fine=True)
        r_sorted = np.sort(x[:K])
        found[label] = (sensors, u, best_ring[label], ring_u)
        rows.append({"drone": label, "Pe": pe, "u_best_ring": ring_u,
                     "best_ring_radius": best_ring[label], "u_global": u,
                     "global_minus_ring": u - ring_u,
                     "radii_sorted": " ".join(f"{r:.2f}" for r in r_sorted),
                     "radius_spread": r_sorted[-1] - r_sorted[0],
                     "sensors_xy": " ".join(f"{a:.3f},{b:.3f}" for a, b in sensors),
                     "evaluations": de.nfev + local.nfev})
        print(f"{label:22s} best ring u = {ring_u:.4f}; global search u = {u:.4f} "
              f"({u - ring_u:+.4f}); radii {rows[-1]['radii_sorted']}  "
              f"({de.nfev + local.nfev} evaluations, {time.time() - t0:.0f} s)")

    with open(ROOT / "results" / "q3_global_search.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(PES), figsize=(4.3 * len(PES), 5.2))
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, (label, (sensors, u, a, ring_u)) in zip(axes, found.items()):
        ax.fill(R_OUT * np.cos(t), R_OUT * np.sin(t), color="0.95")
        ax.plot(D_RESP * np.cos(t), D_RESP * np.sin(t), color="0.5", lw=1, ls="--")
        ax.plot(*ring_sensors(K, a).T, "o", color="C0", ms=9, mfc="none", mew=1.8,
                label=f"best ring: u = {ring_u:.3f}")
        ax.plot(*sensors.T, "^", color="C3", ms=8, mfc="none", mew=1.8,
                label=f"global search: u = {u:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-R_OUT, R_OUT)
        ax.set_ylim(-R_OUT, R_OUT)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(f"Global search from random layouts vs the best ring (K = {K}, response "
                 f"distance {D_RESP:g}, Da = {DA:g})", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    fig.savefig(ROOT / "figures" / "q3_global_search.png", dpi=200)
    plt.close(fig)
    print("\nSaved results/q3_global_search.csv and figures/q3_global_search.png")


if __name__ == "__main__":
    main()
