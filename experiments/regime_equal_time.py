"""Step 3: is it randomness, or just speed?

In regime_map.py, lowering Pe raises sigma at a fixed drift speed, so a random drone
can also reach the site faster than the straight-line drone it is compared with. This
script removes that difference: it compares the random drone with a straight-line drone
that takes the SAME AVERAGE TIME to reach the site.

1. Mean arrival time from the area edge, T(R), with no sensors. Because nothing then
   depends on angle, T solves a radial ODE (the generator applied to T equals -1):

       D (T'' + T'/r) - v T' = -1,   T(r0) = 0,   T'(R) = 0 (reflecting edge)

2. A straight-line drone covering R - r0 in time T(R) flies at v_eq = (R - r0)/T(R).
   Its chance of arriving undetected along the best ray is

       u_straight_eq = exp(-I / v_eq),   I = integral of kappa along that ray,

   and I comes from the straight-line column already in results/regime_map.csv
   (u_straight = exp(-I) there, since v = 1).

3. delta_eq = u_random - u_straight_eq. Where it stays positive, randomness itself
   helps the drone, not just speed.

Needs results/regime_map.csv from regime_map.py; no new PDE solves (a few seconds).
Run:  PYTHONPATH=src python experiments/regime_equal_time.py
Writes results/regime_equal_time.csv and figures/regime_equal_time.png.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy.integrate import solve_bvp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regime_map import R, R0, ROOT  # noqa: E402


def mean_arrival_time(pe: float, v: float = 1.0) -> float:
    """T(R): mean time to reach the site from the area edge, no sensors, reflecting edge."""
    D = v / pe                                   # Pe = v rho / D with rho = 1

    def rhs(s, y):
        T, dT = y
        return np.vstack([dT, (v * dT - 1.0) / D - dT / s])

    def bc(ya, yb):
        return np.array([ya[0], yb[1]])

    mesh = np.linspace(R0, R, 400)
    guess = np.vstack([(mesh - R0) / v, np.full_like(mesh, 1.0 / v)])
    sol = solve_bvp(rhs, bc, mesh, guess, tol=1e-8, max_nodes=200_000)
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed at Pe={pe}: {sol.message}")
    return float(sol.sol(R)[0])


def main():
    with open(ROOT / "results" / "regime_map.csv") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    pes = np.array(sorted({r["Pe"] for r in rows}))
    das = np.array(sorted({r["Da"] for r in rows}))

    times = {pe: mean_arrival_time(pe) for pe in pes}
    straight_time = R - R0                        # straight flight at v = 1
    print("Mean arrival time from the edge (no sensors); straight flight takes "
          f"{straight_time:.2f}")
    for pe in pes:
        faster = "faster" if times[pe] < straight_time else "slower"
        print(f"  Pe {pe:7.2f}: T = {times[pe]:6.2f}  ({faster} than straight flight)")

    delta_eq = np.zeros((pes.size, das.size))
    out = []
    for r in rows:
        i, j = np.searchsorted(pes, r["Pe"]), np.searchsorted(das, r["Da"])
        integral = -np.log(max(r["u_worst_straight"], 1e-300))   # I, with v = 1
        v_eq = straight_time / times[r["Pe"]]
        u_eq = np.exp(-integral / v_eq)
        delta_eq[i, j] = r["u_worst_random"] - u_eq
        out.append({**r, "mean_time": times[r["Pe"]], "v_equal_time": v_eq,
                    "u_straight_equal_time": u_eq, "delta_equal_time": delta_eq[i, j]})

    with open(ROOT / "results" / "regime_equal_time.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    print("\nu(random) - u(straight, same mean time), rows = Da, columns = Pe")
    print("   Da \\ Pe " + " ".join(f"{pe:7.2f}" for pe in pes))
    for j, da in enumerate(das):
        print(f"  {da:8.2f} " + " ".join(f"{delta_eq[i, j]:+7.3f}" for i in range(pes.size)))
    print(f"\nWith equal mean times, randomness helps in {100 * (delta_eq > 0).mean():.0f}% "
          "of the grid.")

    fig, ax = plt.subplots(figsize=(6, 4.3))
    m = ax.pcolormesh(pes, das, delta_eq.T, shading="nearest", cmap="RdBu_r",
                      norm=TwoSlopeNorm(0.0, min(delta_eq.min(), -1e-3), max(delta_eq.max(), 1e-3)))
    ax.contour(pes, das, delta_eq.T, levels=[0.0], colors="k", linewidths=1.5)
    cb = fig.colorbar(m, ax=ax, label="u(random) - u(straight, same mean time)")
    lo, hi = m.norm.vmin, m.norm.vmax
    cb.set_ticks(np.concatenate([np.linspace(lo, 0, 3)[:-1], np.linspace(0, hi, 5)]))
    cb.ax.set_yticklabels([f"{t:+.2f}" for t in cb.get_ticks()])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peclet number Pe (directed  →)")
    ax.set_ylabel("detection number Da")
    ax.set_title("Randomness vs a straight drone with the same mean time")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "regime_equal_time.png", dpi=200)
    plt.close(fig)
    print("Saved results/regime_equal_time.csv and figures/regime_equal_time.png")


if __name__ == "__main__":
    main()
