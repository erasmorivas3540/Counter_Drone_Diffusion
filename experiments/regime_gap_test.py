"""Is the gap between neighbouring sensors what sets the regime-map switching line?

regime_layouts.py found that 4 sensors at ring radius 2.5 and 6 sensors at radius 3.5
give almost the same switching line (where random maneuvering goes from helping the
drone to hurting it), and both have nearly the same gap between neighbours. This
script tests that idea directly: it builds three groups of layouts, each group sharing
one gap but using different sensor counts K and ring radii,

    gap g = 2 a sin(pi / K)       (straight-line distance between neighbouring sensors)
    ring radius a = g / (2 sin(pi / K))

and overlays the switching lines. If the gap is what matters, lines in the same group
(same color) lie on top of each other while the groups separate.

Same model as regime_map.py: site radius 0.5, area radius 5, reflecting edge, Gaussian
footprints, rho = 1, v = 1, sigma = sqrt(2/Pe), lam0 = Da.

Run:  PYTHONPATH=src python experiments/regime_gap_test.py          (about 10 minutes)
      PYTHONPATH=src python experiments/regime_gap_test.py --quick  (about 3 minutes)
      add --plot-only to redraw the figure from the saved CSV in seconds
Writes results/regime_gap_test.csv and figures/regime_gap_test.png.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regime_layouts import ROOT, scenario, worst_straight  # noqa: E402

from cdd import solve  # noqa: E402

GROUPS = {1.8: (4, 6, 8), 2.6: (4, 6, 8), 3.8: (4, 5, 6)}   # gap -> sensor counts
COLORS = {1.8: "C2", 2.6: "k", 3.8: "C0"}
STYLES = {4: "-", 5: "-.", 6: "--", 8: ":"}


def ring_radius(gap: float, k: int) -> float:
    return gap / (2.0 * np.sin(np.pi / k))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="coarser 6 x 5 grid")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from results/regime_gap_test.csv")
    args = parser.parse_args()
    if args.plot_only:
        replot()
        return
    pes = np.logspace(-1, 2, 6 if args.quick else 8)
    das = np.logspace(np.log10(0.05), 1, 5 if args.quick else 8)

    rows, results = [], {}
    t0 = time.time()
    for gap, ks in GROUPS.items():
        for k in ks:
            a = ring_radius(gap, k)
            delta = np.zeros((pes.size, das.size))
            for j, da in enumerate(das):
                straight = worst_straight(scenario(k, a, 1.0, da), k)
                for i, pe in enumerate(pes):
                    rand = solve(scenario(k, a, pe, da), nr=400, ntheta=256).worst_entry()[0]
                    delta[i, j] = rand - straight
                    rows.append({"gap": gap, "K": k, "ring_radius": a, "Pe": pe, "Da": da,
                                 "u_worst_random": rand, "u_worst_straight": straight,
                                 "delta": rand - straight})
            results[(gap, k)] = (a, delta)
            print(f"gap {gap}: K = {k}, ring radius {a:.2f}, arc between sensors "
                  f"{2 * np.pi * a / k:.2f}  ->  helps in {100 * (delta > 0).mean():3.0f}% "
                  f"of the grid  ({time.time() - t0:.0f} s)")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "regime_gap_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # How tightly does each group agree? Compare the sign of delta cell by cell.
    print("\nShare of grid cells where layouts in the same group agree on helps/hurts:")
    for gap, ks in GROUPS.items():
        signs = np.array([results[(gap, k)][1] > 0 for k in ks])
        agree = (signs.all(axis=0) | (~signs).all(axis=0)).mean()
        print(f"  gap {gap}: {100 * agree:.0f}%")

    plot(pes, das, results)


def plot(pes, das, results):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (gap, k), (a, delta) in results.items():
        if delta.min() < 0 < delta.max():
            ax.contour(pes, das, delta.T, levels=[0.0], colors=COLORS[gap],
                       linestyles=STYLES[k], linewidths=1.8)
        ax.plot([], [], color=COLORS[gap], ls=STYLES[k],
                label=f"gap {gap}: K = {k}, radius {a:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peclet number Pe (directed  →)")
    ax.set_ylabel("detection number Da")
    ax.set_title("Switching lines grouped by the gap between sensors\n"
                 "(same color = same gap; randomness helps above/left of a line)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "regime_gap_test.png", dpi=200)
    plt.close(fig)
    print("\nSaved results/regime_gap_test.csv and figures/regime_gap_test.png")


def replot():
    with open(ROOT / "results" / "regime_gap_test.csv") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    pes = np.array(sorted({r["Pe"] for r in rows}))
    das = np.array(sorted({r["Da"] for r in rows}))
    results = {}
    for gap, ks in GROUPS.items():
        for k in ks:
            sel = [r for r in rows if r["gap"] == gap and r["K"] == k]
            delta = np.zeros((pes.size, das.size))
            for r in sel:
                delta[np.searchsorted(pes, r["Pe"]), np.searchsorted(das, r["Da"])] = r["delta"]
            results[(gap, k)] = (sel[0]["ring_radius"], delta)
    plot(pes, das, results)


if __name__ == "__main__":
    main()
