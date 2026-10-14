"""Question 3, part 2a: does freeing the sensors beat the best ring?

Starts from the best ring found by q3_best_ring.py and lets every sensor move: each
sensor's radius and angle become free variables (2K numbers), and Powell's method
minimizes the drone's best chance, the worst-case u over all entry points on the area
edge. Two starting layouts per case:

    ring       the best ring from results/q3_best_ring.csv
    staggered  the same ring with sensors alternately 0.6 inside and 0.6 outside it

The search uses a coarse grid (200 x 128, within about 5e-4 of the fine grid and five
times faster); every final layout is re-scored on the fine grid (400 x 256).

Setup: K = 6 sensors, response distance d = 2, Da = 1, and the three drone types of
q3_best_ring.py.

Run:  PYTHONPATH=src python experiments/q3_free_layout.py      (about 10-15 minutes)
Needs results/q3_best_ring.csv. Writes results/q3_free_layout.csv and
figures/q3_free_layout.png.
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
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q3_best_ring import DA, DEPTH, PES, ROOT  # noqa: E402

from cdd import Scenario, ring_sensors, solve  # noqa: E402

K, D_RESP = 6, 2.0
R_OUT = D_RESP + DEPTH
R_MIN, R_MAX = 0.3, R_OUT - 0.5
MAX_EVALS = 600


def scenario_xy(sensors, pe):
    return Scenario(r0=D_RESP, R=R_OUT, v=1.0, sigma=np.sqrt(2.0 / pe), lam0=DA, rho=1.0,
                    sensors=sensors, kernel="gaussian", outer="reflect")


def to_xy(params):
    radii, angles = params[:K], params[K:]
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def worst_u(sensors, pe, fine=False):
    nr, nth = (400, 256) if fine else (200, 128)
    return solve(scenario_xy(sensors, pe), nr=nr, ntheta=nth).worst_entry()[0]


def optimize(start, pe):
    """Powell search over radii and angles; radii kept in [R_MIN, R_MAX]."""
    bounds = [(R_MIN, R_MAX)] * K + [(None, None)] * K
    res = minimize(lambda p: worst_u(to_xy(p), pe), start, method="Powell", bounds=bounds,
                   options={"maxfev": MAX_EVALS, "xtol": 1e-3, "ftol": 1e-5})
    return res.x, res.nfev


def main():
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        best = {r["drone"]: float(r["best_radius"]) for r in csv.DictReader(f)
                if float(r["response_distance"]) == D_RESP and int(r["K"]) == K}

    rows, layouts = [], {}
    t0 = time.time()
    angles0 = 2 * np.pi * np.arange(K) / K
    for label, pe in PES.items():
        a = best[label]
        ring_u = worst_u(ring_sensors(K, a), pe, fine=True)
        layouts[(label, "best ring")] = (ring_sensors(K, a), ring_u)
        starts = {"ring": np.r_[np.full(K, a), angles0],
                  "staggered": np.r_[a + 0.6 * (-1.0) ** np.arange(K), angles0]}
        for name, x0 in starts.items():
            x, nfev = optimize(x0, pe)
            sensors = to_xy(x)
            u = worst_u(sensors, pe, fine=True)
            layouts[(label, f"free from {name}")] = (sensors, u)
            radii = np.sort(x[:K])
            rows.append({"drone": label, "Pe": pe, "start": name, "ring_radius": a,
                         "u_best_ring": ring_u, "u_free": u, "improvement": ring_u - u,
                         "relative_improvement": (ring_u - u) / ring_u,
                         "radii_sorted": " ".join(f"{r:.2f}" for r in radii),
                         "sensors_xy": " ".join(f"{x:.3f},{y:.3f}" for x, y in sensors),
                         "evaluations": nfev})
            print(f"{label:22s} start {name:9s}: best ring u = {ring_u:.4f} -> free layout "
                  f"u = {u:.4f} ({100 * (ring_u - u) / ring_u:+.1f}%), radii "
                  f"{rows[-1]['radii_sorted']}  ({time.time() - t0:.0f} s)")

    with open(ROOT / "results" / "q3_free_layout.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(PES), figsize=(4.3 * len(PES), 5.4))
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, label in zip(axes, PES):
        ax.fill(R_OUT * np.cos(t), R_OUT * np.sin(t), color="0.95")
        ax.plot(D_RESP * np.cos(t), D_RESP * np.sin(t), color="0.5", lw=1, ls="--")
        ax.text(0, -D_RESP + 0.2, "response line", ha="center", fontsize=7, color="0.4")
        for (lab, kind), (sensors, u) in layouts.items():
            if lab != label:
                continue
            style = {"best ring": ("o", "C0"), "free from ring": ("^", "C3"),
                     "free from staggered": ("s", "C2")}[kind]
            ax.plot(*sensors.T, style[0], color=style[1], ms=8, mfc="none", mew=1.8,
                    label=f"{kind}: u = {u:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-R_OUT, R_OUT)
        ax.set_ylim(-R_OUT, R_OUT)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(f"Free sensor positions vs the best ring (K = {K}, response distance "
                 f"{D_RESP:g}, Da = {DA:g})", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(ROOT / "figures" / "q3_free_layout.png", dpi=200)
    plt.close(fig)
    print("\nSaved results/q3_free_layout.csv and figures/q3_free_layout.png")


if __name__ == "__main__":
    main()
