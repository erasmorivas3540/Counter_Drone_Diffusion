"""Question 4, part 2: how fast sensors switch on and off matters, not just how often.

For the always-on best ring (K = 6, response distance d = 2, Da = 1) and each drone type,
Monte Carlo estimates of the drone's chance from the midpoint entry (the worst entry for a
symmetric ring) as the switching rate gamma = alpha + beta varies over four decades, at
availabilities p = 0.9 and 0.7. The two exact limits are drawn as reference lines:
fast switching (PDE with rate p*lam0) and frozen states (64-pattern mixture, unaware drone).

The x-axis is gamma * T, where T is the drone's mean flight time from the edge: below about 1
a sensor rarely changes state during one approach (frozen); above about 10 it flickers many
times (fast).

Run:  PYTHONPATH=src python experiments/q4_downtime_switching.py      (about 30 min, 1 core)
Resumable: each Monte Carlo run is saved to results/checkpoints/ as it finishes.
Writes results/q4_downtime_switching.csv and figures/q4_downtime_switching.png.
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
from scipy.sparse import diags  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint, make_key  # noqa: E402
from q3_best_ring import DA, PES, ROOT  # noqa: E402
from q3_free_layout import D_RESP, K, R_OUT, scenario_xy  # noqa: E402

from cdd import ring_sensors  # noqa: E402
from cdd.downtime import frozen_edge, simulate_u_switching, u_fast, u_frozen_parts_ring  # noqa: E402

AVAILABILITY = (0.9, 0.7)
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
N_PATHS = 40_000
NTH = 264                                    # divisible by 12, so pi/6 is a grid angle


def mean_flight_time(D, n=4000):
    """T(R) from D(T'' + T'/r) - T' = -1, T(d) = 0, T'(R) = 0 (finite differences)."""
    r = np.linspace(D_RESP, R_OUT, n + 1)
    h = r[1] - r[0]
    ri = r[1:]
    dp = D * (ri + h / 2) / (ri * h * h)
    dm = D * (ri - h / 2) / (ri * h * h)
    P, M = dp - 1 / (2 * h), dm + 1 / (2 * h)
    lower, upper = M[1:].copy(), P[:-1].copy()
    lower[-1] += P[-1]
    A = diags([lower, -(dp + dm), upper], [-1, 0, 1], format="csc")
    return float(spsolve(A, -np.ones(n))[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    ck = Checkpoint("q4_downtime_switching", fresh=args.fresh)
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        ring = {r["drone"]: float(r["best_radius"]) for r in csv.DictReader(f)
                if float(r["response_distance"]) == D_RESP and int(r["K"]) == K}
    t0 = time.time()
    rows, refs = [], {}
    entry_angle = np.pi / K
    for label, pe in PES.items():
        scn = scenario_xy(ring_sensors(K, ring[label]), pe)
        T = mean_flight_time(scn.D)
        x0 = R_OUT * np.array([np.cos(entry_angle), np.sin(entry_angle)])
        j = int(round(entry_angle / (2 * np.pi / NTH)))
        dt = min(1e-2, 0.02 / scn.sigma**2)
        for p in AVAILABILITY:
            key = make_key("limits", pe, p)
            if key not in ck:
                fast = float(u_fast(scn, p, nr=400, ntheta=NTH).u[-1, j])
                unaware, _ = frozen_edge(u_frozen_parts_ring(scn, p, nr=400, ntheta=NTH))
                ck.add(key, {"fast": fast, "frozen": float(unaware[j])})
            refs[(label, p)] = ck[key]
            for g in GAMMAS:
                key = make_key("mc", pe, p, g)
                if key not in ck:
                    r = simulate_u_switching(x0, scn, p, g, dt=dt, n_paths=N_PATHS,
                                             seed=int(1000 * pe + 100 * p + 10 * np.log10(g) + 500))
                    ck.add(key, {"u": r.estimate, "half_width": r.half_width, "timed_out": r.timed_out})
                res = ck[key]
                rows.append({"drone": label, "Pe": pe, "availability": p, "gamma": g,
                             "mean_flight_time": T, "gamma_T": g * T, "u": res["u"],
                             "ci_half_width": res["half_width"],
                             "u_fast_limit": refs[(label, p)]["fast"],
                             "u_frozen_limit": refs[(label, p)]["frozen"]})
                print(f"{label:22s} p={p:.1f} gamma={g:<6g} (gamma T = {g * T:7.2f}): "
                      f"u = {res['u']:.4f} ± {res['half_width']:.4f}   [fast {refs[(label, p)]['fast']:.4f}, "
                      f"frozen {refs[(label, p)]['frozen']:.4f}]  ({time.time() - t0:.0f} s)", flush=True)

    with open(ROOT / "results" / "q4_downtime_switching.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, len(PES), figsize=(4.5 * len(PES), 4.0), sharey=False)
    for ax, label in zip(axes, PES):
        for p, color in zip(AVAILABILITY, ("C0", "C3")):
            sub = [r for r in rows if r["drone"] == label and r["availability"] == p]
            x = np.array([r["gamma_T"] for r in sub])
            y = np.array([r["u"] for r in sub])
            e = np.array([r["ci_half_width"] for r in sub])
            ax.errorbar(x, y, yerr=e, fmt="o-", color=color, ms=4, capsize=2,
                        label=f"availability {p:.0%}")
            ax.axhline(sub[0]["u_fast_limit"], color=color, ls=":", lw=1)
            ax.axhline(sub[0]["u_frozen_limit"], color=color, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("switching rate × mean flight time (γT)")
        ax.set_ylabel("drone's chance from the midpoint entry")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, title="dashed: frozen limit\ndotted: fast limit", title_fontsize=7)
    fig.suptitle(f"Slow outages help the drone more than brief ones (K = {K} ring, response "
                 f"distance {D_RESP:g}, Da = {DA:g})", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "q4_downtime_switching.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved results/q4_downtime_switching.csv and figures/q4_downtime_switching.png "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
