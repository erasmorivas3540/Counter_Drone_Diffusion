"""Step 4: does the regime map depend on the sensor layout?

Repeats the question-1 comparison (worst-case u with random maneuvering minus the
straight-line value) for five ring layouts on a coarser 8 x 6 (Pe, Da) grid:

    K = 4, 6, 8 sensors at ring radius 2.5     (how many sensors)
    K = 6 sensors at ring radius 1.5 and 3.5   (how far out the ring sits)

Site radius 0.5, area radius 5, reflecting edge, Gaussian footprints, as in
regime_map.py. The figure overlays each layout's zero line (where randomness switches
from helping to hurting the drone); if the lines have similar shapes, the finding is
not an accident of one layout.

Run:  PYTHONPATH=src python experiments/regime_layouts.py      (about 5 minutes)
Writes results/regime_layouts.csv and figures/regime_layouts.png.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from cdd import Scenario, ring_sensors, solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R0, R = 0.5, 5.0
LAYOUTS = [(4, 2.5), (6, 2.5), (8, 2.5), (6, 1.5), (6, 3.5)]   # (K sensors, ring radius)
PES = np.logspace(-1, 2, 8)
DAS = np.logspace(-1, 1, 6)


def scenario(k, ring, pe, da):
    return Scenario(r0=R0, R=R, v=1.0, sigma=np.sqrt(2.0 / pe), lam0=da, rho=1.0,
                    sensors=ring_sensors(k, ring), kernel="gaussian", outer="reflect")


def worst_straight(scn, k, n_angles=121):
    """Best straight-line u over entry angles; by symmetry angles 0..pi/K suffice."""
    best = 0.0
    for a in np.linspace(0.0, np.pi / k, n_angles):
        e = np.array([np.cos(a), np.sin(a)])
        rate = lambda s: float(scn.detection_rate(s * e)[0])
        integral = quad(rate, scn.r0, scn.R, epsabs=1e-12, epsrel=1e-10, limit=200)[0]
        best = max(best, np.exp(-integral / scn.v))
    return best


def main():
    rows, deltas = [], {}
    t0 = time.time()
    for k, ring in LAYOUTS:
        delta = np.zeros((PES.size, DAS.size))
        for j, da in enumerate(DAS):
            straight = worst_straight(scenario(k, ring, 1.0, da), k)   # sigma not used
            for i, pe in enumerate(PES):
                rand = solve(scenario(k, ring, pe, da), nr=400, ntheta=256).worst_entry()[0]
                delta[i, j] = rand - straight
                rows.append({"K": k, "ring_radius": ring, "Pe": pe, "Da": da,
                             "u_worst_random": rand, "u_worst_straight": straight,
                             "delta": rand - straight})
        deltas[(k, ring)] = delta
        print(f"K = {k}, ring {ring}: largest help {delta.max():+.3f}, largest harm "
              f"{delta.min():+.3f}, helps in {100 * (delta > 0).mean():.0f}% of the grid  "
              f"({time.time() - t0:.0f} s)")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "regime_layouts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    styles = {(4, 2.5): ("C0", "-"), (6, 2.5): ("k", "-"), (8, 2.5): ("C3", "-"),
              (6, 1.5): ("C2", "--"), (6, 3.5): ("C4", "--")}
    for (k, ring), delta in deltas.items():
        color, ls = styles[(k, ring)]
        ax.contour(PES, DAS, delta.T, levels=[0.0], colors=color, linestyles=ls, linewidths=1.8)
        ax.plot([], [], color=color, ls=ls, label=f"K = {k}, ring radius {ring}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peclet number Pe (directed  →)")
    ax.set_ylabel("detection number Da")
    ax.set_title("Where randomness switches from helping to hurting\n"
                 "(helps above and to the left of each line)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "regime_layouts.png", dpi=200)
    plt.close(fig)
    print("\nSaved results/regime_layouts.csv and figures/regime_layouts.png")


if __name__ == "__main__":
    main()
