"""Question 3, part 1: the best ring radius for K = 3 to 8 sensors.

For each number of sensors K and each drone type (Pe), find the ring radius a that
makes the drone's best chance of getting through, the worst-case u over all entry
points on the area edge, as small as possible. The search is a coarse scan of radii
followed by a bounded Brent refinement around the best scanned value.

For comparison, the same search is run with the straight-line shortcut (sigma = 0,
u computed exactly by integrating along rays). That is the radius a planner would pick
if they assumed drones fly straight. The script reports how much protection that
choice gives up against random drones:

    cost = u_random(a_straight) - u_random(a_best)   (extra chance the drone gets through)

Response distance: a detection only helps if it comes before the drone is within a
distance d of the site, so "arrival" means reaching the response line r = d (the model's
r0 = d). Without it, the best ring presses against the site, where detection is too late
to act on. The script runs d = 1, 2 and 3 sensor ranges; the area edge sits a fixed
approach depth of 4.5 beyond the response line, so every case has the same distance to
cross. Sensors may sit anywhere from radius 0.3 to just inside the area edge, including
inside the response line. A result flagged [at search edge] means the best radius is at
the end of that range.

Model otherwise as in regime_map.py: reflecting edge, Gaussian footprints, rho = 1,
v = 1, sigma = sqrt(2/Pe), lam0 = Da, Da = 1 throughout.

Run:  PYTHONPATH=src python experiments/q3_best_ring.py          (about 25 minutes)
      PYTHONPATH=src python experiments/q3_best_ring.py --quick  (K = 4, 6, 8; ~12 min)
Writes results/q3_best_ring.csv and figures/q3_best_ring.png.
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
from regime_layouts import ROOT, worst_straight  # noqa: E402

from cdd import Scenario, ring_sensors, solve  # noqa: E402

DA = 1.0
PES = {"wandering (Pe = 0.5)": 0.5, "mixed (Pe = 2)": 2.0, "directed (Pe = 20)": 20.0}
RESPONSE = (1.0, 2.0, 3.0)      # response distance d: a detection only counts before r = d
DEPTH = 4.5                     # approach depth R - d, the same for every d


def scenario(k, a, pe, d):
    """Arrival = reaching the response line r = d; the area edge sits DEPTH beyond it."""
    return Scenario(r0=d, R=d + DEPTH, v=1.0, sigma=np.sqrt(2.0 / pe), lam0=DA, rho=1.0,
                    sensors=ring_sensors(k, a), kernel="gaussian", outer="reflect")


def scan_radii(d):
    """Sensors may sit anywhere from near the site to just inside the area edge."""
    return np.linspace(0.3, d + DEPTH - 0.5, 20)


def u_random(k, a, pe, d):
    return solve(scenario(k, a, pe, d), nr=400, ntheta=256).worst_entry()[0]


def u_straight(k, a, d):
    return worst_straight(scenario(k, a, 1.0, d), k, n_angles=61)   # sigma unused


def best_radius(f, scan):
    """Minimize f(a): scan, then refine with bounded Brent around the best point.

    Returns the radius, the value, and whether the best point sits on the scan's edge
    (then the true optimum may lie outside the range searched).
    """
    values = np.array([f(a) for a in scan])
    i = int(np.argmin(values))
    step = scan[1] - scan[0]
    lo, hi = max(scan[0], scan[i] - step), min(scan[-1], scan[i] + step)
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 0.01})
    at_edge = i in (0, len(scan) - 1)
    if res.fun < values[i]:
        return float(res.x), float(res.fun), at_edge
    return float(scan[i]), float(values[i]), at_edge


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="K = 4, 6, 8 only")
    args = parser.parse_args()
    ks = [4, 6, 8] if args.quick else list(range(3, 9))

    rows = []
    t0 = time.time()
    for d in RESPONSE:
        scan = scan_radii(d)
        print(f"\nResponse distance d = {d:g} (area edge at {d + DEPTH:g})")
        for k in ks:
            a_s, u_s, edge_s = best_radius(lambda a: u_straight(k, a, d), scan)
            for label, pe in PES.items():
                a_b, u_b, edge_b = best_radius(lambda a: u_random(k, a, pe, d), scan)
                u_at_s = u_random(k, a_s, pe, d)
                rows.append({"response_distance": d, "K": k, "Pe": pe, "drone": label,
                             "best_radius": a_b, "best_at_scan_edge": edge_b,
                             "u_worst_at_best": u_b, "straight_line_radius": a_s,
                             "straight_at_scan_edge": edge_s,
                             "u_straight_line_at_its_best": u_s,
                             "u_random_at_straight_choice": u_at_s,
                             "cost_of_straight_choice": u_at_s - u_b,
                             "gap_at_best": 2 * a_b * np.sin(np.pi / k)})
                flag = "  [at search edge]" if edge_b else ""
                print(f"  K = {k}, {label:22s} best radius {a_b:4.2f}{flag}, worst-case u "
                      f"{u_b:.3f}; straight-line pick {a_s:4.2f} gives {u_at_s:.3f} "
                      f"(+{u_at_s - u_b:.3f})  ({time.time() - t0:.0f} s)")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "q3_best_ring.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, len(RESPONSE), figsize=(4.2 * len(RESPONSE), 7.5),
                             sharex=True, squeeze=False)
    for c, d in enumerate(RESPONSE):
        ax1, ax2 = axes[0, c], axes[1, c]
        for n, label in enumerate(PES):
            sel = [r for r in rows if r["drone"] == label and r["response_distance"] == d]
            ks_ = [r["K"] for r in sel]
            ax1.plot(ks_, [r["best_radius"] for r in sel], "o-", color=f"C{n}", label=label)
            ax2.plot(ks_, [r["u_worst_at_best"] for r in sel], "o-", color=f"C{n}",
                     label=f"{label}: best ring")
            ax2.plot(ks_, [r["u_random_at_straight_choice"] for r in sel], "x--",
                     color=f"C{n}", label=f"{label}: straight-line pick")
        first = [r for r in rows if r["drone"] == list(PES)[0] and r["response_distance"] == d]
        ax1.plot([r["K"] for r in first], [r["straight_line_radius"] for r in first], "s:",
                 color="0.4", label="straight-line shortcut")
        ax1.axhline(d, color="0.7", lw=0.8)
        ax1.text(ks[0], d, " response line", va="bottom", fontsize=7, color="0.4")
        ax1.set_title(f"response distance d = {d:g}")
        ax1.set_ylabel("best ring radius")
        ax2.set_ylabel("drone's best chance (worst-case u)")
        ax2.set_yscale("log")
        ax2.set_xlabel("number of sensors K")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=6)
    fig.suptitle(f"Best ring radius by response distance (Da = {DA:g}, approach depth {DEPTH:g})",
                 fontsize=11)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "q3_best_ring.png", dpi=200)
    plt.close(fig)
    print("\nSaved results/q3_best_ring.csv and figures/q3_best_ring.png")


if __name__ == "__main__":
    main()
