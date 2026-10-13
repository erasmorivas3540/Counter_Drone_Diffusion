"""Question 1 and 2: where does random maneuvering help the drone, and how wrong is
the straight-line shortcut?

Layout: K sensors evenly on a ring of radius A around a site of radius R0, inside an
area of radius R with a reflecting edge, Gaussian footprints. Lengths are in sensor
ranges (rho = 1) and the drift speed is v = 1, so the two dimensionless numbers set
the rest:

    Pe = v rho / D   ->  D = 1/Pe,  sigma = sqrt(2/Pe)
    Da = lam0 rho / v ->  lam0 = Da

For each (Pe, Da) the script solves the PDE and records the WORST-CASE u: the drone's
best chance of arriving undetected over all entry points on the area edge. It compares
that with the straight-line value (sigma = 0), computed exactly by integrating the
detection rate along each ray:

    delta  = u_worst(random) - u_worst(straight)   > 0: randomness helps the drone
    ratio  = u_worst(random) / u_worst(straight)        (question 2: shortcut error)

Grid: nr = 400 radial points keeps the cell Peclet number v h / (2D) below 1 for
Pe <= 177, so the solver stays second order; the script refines the grid at the
extreme corners and reports the change, so you can see the numbers are converged.

Run:  PYTHONPATH=src python experiments/regime_map.py          (about 4 minutes)
      PYTHONPATH=src python experiments/regime_map.py --quick  (6 x 5 grid, ~1 min)
Writes results/regime_map.csv and figures/regime_map.png.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from cdd import Scenario, ring_sensors, solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K, A, R0, R = 6, 2.5, 0.5, 5.0          # sensors, ring radius, site radius, area radius
NR, NTHETA = 400, 256


def scenario(pe: float, da: float) -> Scenario:
    return Scenario(r0=R0, R=R, v=1.0, sigma=np.sqrt(2.0 / pe), lam0=da, rho=1.0,
                    sensors=ring_sensors(K, A), kernel="gaussian", outer="reflect")


def worst_random(scn: Scenario, nr=NR, ntheta=NTHETA) -> float:
    """Largest u on the area edge from the PDE (the drone's best entry point)."""
    return solve(scn, nr=nr, ntheta=ntheta).worst_entry()[0]


def worst_straight(scn: Scenario, n_angles: int = 121) -> float:
    """Largest straight-line u over entry angles: exp(-(1/v) * integral of kappa).

    By symmetry the best ray lies between two neighbouring sensors, so angles from
    0 to pi/K cover every case.
    """
    best = 0.0
    for a in np.linspace(0.0, np.pi / K, n_angles):
        e = np.array([np.cos(a), np.sin(a)])
        rate = lambda s: float(scn.detection_rate(s * e)[0])
        integral = quad(rate, scn.r0, scn.R, epsabs=1e-12, epsrel=1e-10, limit=200)[0]
        best = max(best, np.exp(-integral / scn.v))
    return best


def convergence_check(pes, das):
    """Refine the grid (2x in r and theta) at the four corners of the map."""
    print("\nGrid check at the corners (change in worst-case u when the grid doubles):")
    worst = 0.0
    for pe in (pes[0], pes[-1]):
        for da in (das[0], das[-1]):
            scn = scenario(pe, da)
            coarse = worst_random(scn)
            fine = worst_random(scn, nr=2 * NR, ntheta=2 * NTHETA)
            worst = max(worst, abs(fine - coarse))
            print(f"  Pe {pe:8.2f}  Da {da:6.2f}:  {coarse:.5f} -> {fine:.5f}  "
                  f"(change {fine - coarse:+.1e})")
    print(f"  largest change {worst:.1e}")


def figure(pes, das, delta, ratio, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) separate color ranges for help (red) and harm (blue): the harm is smaller,
    # and a symmetric scale would wash it out
    m1 = ax1.pcolormesh(pes, das, delta.T, shading="nearest", cmap="RdBu_r",
                        norm=TwoSlopeNorm(0.0, min(delta.min(), -1e-3), max(delta.max(), 1e-3)))
    ax1.contour(pes, das, delta.T, levels=[0.0], colors="k", linewidths=1.5)
    cb1 = fig.colorbar(m1, ax=ax1, label="u(random) - u(straight line)")
    lo, hi = m1.norm.vmin, m1.norm.vmax
    cb1.set_ticks(np.concatenate([np.linspace(lo, 0, 3)[:-1], np.linspace(0, hi, 5)]))
    cb1.ax.set_yticklabels([f"{t:+.2f}" for t in cb1.get_ticks()])
    ax1.set_title("(a) Q1: red = randomness helps the drone")
    # (b) log10 of the ratio, so a factor of 10 too low or too high looks the same size;
    # where the straight-line u is tiny the ratio is huge, so the scale stops at 100x
    log_ratio = np.clip(np.log10(ratio), -2.0, 2.0)
    m2 = ax2.pcolormesh(pes, das, log_ratio.T, shading="nearest", cmap="PuOr_r",
                        norm=TwoSlopeNorm(0.0, -2.0, 2.0))
    ax2.contour(pes, das, log_ratio.T, levels=np.log10([0.9, 1.1]), colors="k",
                linewidths=1, linestyles="--")
    cb = fig.colorbar(m2, ax=ax2, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["1/100", "1/10", "1", "10", "\u2265100"])
    cb.set_label("u(random) / u(straight line)")
    ax2.set_title("(b) Q2: shortcut error (dashed: within 10%)")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Peclet number Pe (directed  →)")
        ax.set_ylabel("detection number Da")
    fig.suptitle(f"Worst-case undetected arrival, {K} sensors on a ring of radius {A}",
                 fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="6 x 5 grid instead of 12 x 10")
    args = parser.parse_args()
    pes = np.logspace(-1, 2, 6 if args.quick else 12)
    das = np.logspace(-1, 1, 5 if args.quick else 10)

    delta = np.zeros((pes.size, das.size))
    ratio = np.zeros_like(delta)
    rows = []
    t0 = time.time()
    for j, da in enumerate(das):
        straight = worst_straight(scenario(1.0, da))     # sigma does not enter here
        for i, pe in enumerate(pes):
            rand = worst_random(scenario(pe, da))
            delta[i, j], ratio[i, j] = rand - straight, rand / straight
            rows.append({"Pe": pe, "Da": da, "sigma": np.sqrt(2 / pe),
                         "u_worst_random": rand, "u_worst_straight": straight,
                         "delta": rand - straight, "ratio": rand / straight})
        print(f"Da {da:6.2f} done  ({time.time() - t0:5.0f} s)")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "regime_map.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure(pes, das, delta, ratio, ROOT / "figures" / "regime_map.png")

    print("\nu(random) - u(straight), rows = Da, columns = Pe")
    print("   Da \\ Pe " + " ".join(f"{pe:7.2f}" for pe in pes))
    for j, da in enumerate(das):
        print(f"  {da:8.2f} " + " ".join(f"{delta[i, j]:+7.3f}" for i in range(pes.size)))
    helps = (delta > 0).mean()
    print(f"\nRandomness helps the drone in {100 * helps:.0f}% of the (Pe, Da) grid.")
    convergence_check(pes, das)
    print(f"\nSaved results/regime_map.csv and figures/regime_map.png")


if __name__ == "__main__":
    main()
