"""Question 4, part 3: how many sensors make up for downtime?

Target: the drone's best chance against six always-on sensors on their best ring (K = 6,
response distance d = 2, Da = 1). For availability p = 0.9 and 0.7 and K = 6..12 sensors on
a ring, this script finds the ring radius that is best against each threat and reports:

  aware drone    knows which sensors are down (long outages): the hardest case; the ring
                 radius is re-optimized for it
  unaware drone  long outages it cannot see; evaluated at the always-on best radius for K
                 (re-optimizing gained < 0.001 in q4_downtime_limits.py)
  fast           brief outages; evaluated at the same radius

and the smallest K that brings each threat back to the always-on six-sensor level.

Outage patterns: one per rotation class is solved, and patterns are kept in order of how
many sensors are down until the leftover probability is below 1e-3 (p = 0.9) or 1e-2
(p = 0.7). Dropped patterns count as the drone getting through, so reported chances are
upper bounds, high by at most the dropped probability (column `dropped`).

Grid: 200 radial x about 132 angles (a multiple of K), accurate to about 5e-4.

Run:  PYTHONPATH=src python experiments/q4_redundancy.py          (about 1.5 h on 2 cores)
Resumable: every evaluation is saved to results/checkpoints/ as it finishes.
Writes results/q4_redundancy.csv, results/q4_redundancy_rule.csv and
figures/q4_redundancy.png.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint, make_key  # noqa: E402
from q3_best_ring import DA, PES, ROOT  # noqa: E402
from q3_free_layout import D_RESP, R_MAX, scenario_xy  # noqa: E402

from cdd import ring_sensors, solve  # noqa: E402
from cdd.downtime import frozen_bounds, u_fast, u_frozen_parts_ring_truncated  # noqa: E402

WORKERS = int(os.environ.get("CDD_WORKERS", "2"))
KS = range(6, 13)
TAIL = {0.9: 1e-3, 0.7: 1e-2}
CK_NAME = "q4_redundancy"
SCAN = np.linspace(1.0, 5.0, 9)


def ntheta(K):
    return K * round(132 / K)


def on_value(K, a, pe):
    return solve(scenario_xy(ring_sensors(K, a), pe), nr=200, ntheta=ntheta(K)).worst_entry()[0]


def downtime_values(K, a, pe, p):
    scn = scenario_xy(ring_sensors(K, a), pe)
    parts, dropped = u_frozen_parts_ring_truncated(scn, p, nr=200, ntheta=ntheta(K),
                                                   tail_tol=TAIL[p])
    unaware, aware = frozen_bounds(parts, dropped)
    fast = u_fast(scn, p, nr=200, ntheta=ntheta(K)).worst_entry()[0]
    return {"aware": aware, "unaware": unaware, "fast": fast, "dropped": dropped}


def task(args):
    K, label, pe, p = args
    ck = Checkpoint(CK_NAME)

    def on(a):
        key = make_key("on", K, pe, round(float(a), 6))
        if key not in ck:
            ck.add(key, on_value(K, a, pe))
        return ck[key]

    def dn(a):
        key = make_key("down", K, pe, p, round(float(a), 6))
        if key not in ck:
            ck.add(key, downtime_values(K, a, pe, p))
        return ck[key]

    def best(f, scan):
        vals = [f(a) for a in scan]
        i = int(np.argmin(vals))
        lo, hi = scan[max(i - 1, 0)], scan[min(i + 1, len(scan) - 1)]
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 0.02})
        return float(res.x), float(res.fun)

    a_on, u_on = best(on, np.linspace(1.0, R_MAX, 12))
    at_on = dn(a_on)
    a_aw, u_aw = best(lambda a: dn(a)["aware"], SCAN)
    return {"drone": label, "Pe": pe, "availability": p, "K": K,
            "always_on_radius": a_on, "u_always_on": u_on,
            "u_unaware": at_on["unaware"], "u_fast": at_on["fast"],
            "aware_radius": a_aw, "u_aware": u_aw, "u_aware_at_always_on_radius": at_on["aware"],
            "dropped": dn(a_aw)["dropped"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    Checkpoint(CK_NAME, fresh=args.fresh)
    jobs = [(K, label, pe, p) for K in KS for label, pe in PES.items() for p in TAIL]
    jobs.sort(key=lambda j: (-j[0], j[3]))              # biggest first
    t0 = time.time()
    rows = []
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(task, jobs):
            rows.append(r)
            print(f"K={r['K']:2d} {r['drone']:22s} p={r['availability']:.1f}: always on "
                  f"{r['u_always_on']:.4f} (a={r['always_on_radius']:.2f}) | fast {r['u_fast']:.4f} | "
                  f"unaware {r['u_unaware']:.4f} | aware {r['u_aware']:.4f} (a={r['aware_radius']:.2f})"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda r: (r["Pe"], -r["availability"], r["K"]))
    with open(ROOT / "results" / "q4_redundancy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Design rule: smallest K whose chance is back at the always-on six-sensor level.
    rule = []
    for label, pe in PES.items():
        target = next(r["u_always_on"] for r in rows if r["drone"] == label and r["K"] == 6)
        for p in TAIL:
            sub = [r for r in rows if r["drone"] == label and r["availability"] == p]
            entry = {"drone": label, "Pe": pe, "availability": p, "target_six_always_on": target}
            for threat in ("fast", "unaware", "aware"):
                ok = [r["K"] for r in sub if r[f"u_{threat}"] <= target]
                entry[f"sensors_needed_{threat}"] = min(ok) if ok else f">{max(KS)}"
            rule.append(entry)
            print(f"{label:22s} p={p:.1f}: sensors needed to match 6 always-on ({target:.3f}): "
                  f"fast {entry['sensors_needed_fast']}, unaware {entry['sensors_needed_unaware']}, "
                  f"aware {entry['sensors_needed_aware']}", flush=True)
    with open(ROOT / "results" / "q4_redundancy_rule.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rule[0]))
        w.writeheader()
        w.writerows(rule)

    styles = {"u_always_on": ("always on", "0.3", "-"), "u_fast": ("brief outages", "C0", "-"),
              "u_unaware": ("long outages, unaware drone", "C3", "-"),
              "u_aware": ("long outages, aware drone", "C1", "-")}
    fig, axes = plt.subplots(len(TAIL), len(PES), figsize=(4.4 * len(PES), 3.7 * len(TAIL)),
                             sharex=True)
    for i, p in enumerate(TAIL):
        for j, (label, pe) in enumerate(PES.items()):
            ax = axes[i, j]
            sub = [r for r in rows if r["drone"] == label and r["availability"] == p]
            Ks = [r["K"] for r in sub]
            for col, (name, color, ls) in styles.items():
                ax.plot(Ks, [r[col] for r in sub], "o" + ls, color=color, ms=4, label=name)
            target = next(r["u_always_on"] for r in sub if r["K"] == 6)
            ax.axhline(target, color="0.3", ls="--", lw=1, label="six always-on sensors")
            ax.set_title(f"{label}, availability {p:.0%}", fontsize=9)
            ax.set_ylabel("drone's best chance u")
            if i == len(TAIL) - 1:
                ax.set_xlabel("number of sensors on the ring")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"How many sensors make up for downtime? (response distance {D_RESP:g}, "
                 f"Da = {DA:g}; ring radius optimized for each case)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "q4_redundancy.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved results/q4_redundancy.csv, results/q4_redundancy_rule.csv and "
          f"figures/q4_redundancy.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
