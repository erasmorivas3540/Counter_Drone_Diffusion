"""From physical units to the model's numbers, and back.

The model has four dimensionless inputs (K = 6 sensors on a ring throughout):

    Pe = v rho / D         how directed the approach is
    Da = lam0 rho / v      expected detections per sensor footprint crossed at full speed
    d / rho, R / rho       response distance and area edge in sensor footprints

with
    v     net closing speed toward the site (m/s), NOT airspeed: a drone that zigzags at
          20 m/s airspeed may close at only 10 m/s
    rho   Gaussian footprint scale of one sensor (m). Detection rate falls to e^{-2} = 14%
          of its peak at 2 rho, so rho is taken as half the sensor's quoted tracking range
    D     diffusion coefficient of the drone's position (m^2/s). For an Ornstein-Uhlenbeck
          maneuver model with lateral velocity spread sigma_v (m/s) and correlation time
          tau (s), D = sigma_v^2 tau for each horizontal component
    lam0  peak detection hazard (1/s) = 1 / (mean time to a detection good enough to act on
          while the drone sits at a sensor's center)

Sourced inputs: drone speeds (DJI Mavic 3 Pro, 21 m/s max; Switchblade 300/600 cruise
101/113 km/h; Shahed-136 about 185 km/h) and sensor ranges (small-drone radars 1-3 km,
DSIAC; Kongsberg DR100: 2 km tracking for a 0.5 m^2 target; acoustic arrays about 60 m,
Ghouli 2026). Assumed inputs, marked in the table: lam0 (time to an actionable detection)
and the maneuver statistics (sigma_v, tau), for which no published values exist.

For each scenario the script reports Pe and Da; the best ring radius in meters; the drone's
best chance against it; the chance a straight-line planner's ring would allow; the drone's
mean flight time; and the outage durations that count as "brief" or "long" in Question 4.

Directed cases (Pe > 50) use the straight-line closed form, accurate to about D|H| ~ 0.1%
by the small-D expansion of the theory section; the rest use the PDE (400 x 264 grid).

Run:  PYTHONPATH=src python experiments/real_world_mapping.py          (about 2 minutes)
Writes results/real_world_mapping.csv and results/real_world_mapping.md.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q3_best_ring import ROOT  # noqa: E402

from cdd import Scenario, ring_sensors, solve  # noqa: E402

K, D_RHO, R_RHO = 6, 2.0, 6.5           # response line at 2 rho, area edge at 6.5 rho

# name, drone, sensor, v (m/s), rho (m), sigma_v (m/s), tau (s), time to actionable detection (s)
SCENARIOS = [
    ("A1", "quadcopter, direct approach", "short-range radar, open terrain", 15, 1000, 4, 4, 5),
    ("A2", "quadcopter, direct approach", "short-range radar, heavy clutter", 15, 1000, 4, 4, 60),
    ("A3", "quadcopter, evasive weaving", "short-range radar, heavy clutter", 10, 1000, 10, 15, 60),
    ("B1", "loitering munition (Switchblade-class)", "short-range radar, RF-cued classification", 28, 1000, 3, 3, 20),
    ("C1", "one-way attack drone (Shahed-class)", "long-range radar", 50, 3000, 3, 3, 20),
    ("D1", "quadcopter, slow evasive approach", "EO/IR camera network", 8, 300, 8, 10, 10),
    ("D2", "quadcopter, searching/loitering", "EO/IR camera network", 3, 300, 8, 20, 10),
    ("E1", "quadcopter, searching/loitering", "acoustic array", 3, 60, 8, 20, 2),
    ("E2", "quadcopter, slow evasive approach", "acoustic array", 8, 60, 8, 10, 2),
]


def straight_exposure(a, theta):
    """Straight-line exposure (Da = 1, rho = 1) of a K-ring along the ray at angle theta."""
    s = a * np.exp(2j * np.pi * np.arange(K) / K)
    t = theta - np.angle(s)
    c, sn = np.abs(s) * np.cos(t), np.abs(s) * np.sin(t)
    return np.sum(np.sqrt(np.pi / 2) * np.exp(-sn**2 / 2)
                  * (erf((R_RHO - c) / np.sqrt(2)) - erf((D_RHO - c) / np.sqrt(2))))


def M_K(a):
    return straight_exposure(a, np.pi / K)            # worst entry: midpoint (proved)


A_STRAIGHT = minimize_scalar(lambda a: -M_K(a), bounds=(1.5, 3.5), method="bounded",
                             options={"xatol": 1e-6}).x


def pde_chance(a, pe, da):
    scn = Scenario(r0=D_RHO, R=R_RHO, v=1.0, sigma=np.sqrt(2.0 / pe), lam0=da, rho=1.0,
                   sensors=ring_sensors(K, a), kernel="gaussian", outer="reflect")
    return solve(scn, nr=400, ntheta=264).worst_entry()[0]


def mean_flight_time(D, n=4000):
    """Dimensionless T(R): D(T'' + T'/r) - T' = -1, T(d) = 0, T'(R) = 0."""
    if D < 1e-3:
        return R_RHO - D_RHO
    r = np.linspace(D_RHO, R_RHO, n + 1)
    h = r[1] - r[0]
    ri = r[1:]
    dp, dm = D * (ri + h / 2) / (ri * h * h), D * (ri - h / 2) / (ri * h * h)
    P, M = dp - 1 / (2 * h), dm + 1 / (2 * h)
    lower, upper = M[1:].copy(), P[:-1].copy()
    lower[-1] += P[-1]
    return float(spsolve(diags([lower, -(dp + dm), upper], [-1, 0, 1], format="csc"), -np.ones(n))[-1])


def analyse(pe, da):
    if pe > 50:
        a_best = A_STRAIGHT
        u_best = float(np.exp(-da * M_K(a_best)))
        return a_best, u_best, u_best, "straight-line closed form"
    f = lambda a: pde_chance(a, pe, da)  # noqa: E731
    scan = np.linspace(1.0, 6.0, 11)
    vals = [f(a) for a in scan]
    i = int(np.argmin(vals))
    res = minimize_scalar(f, bounds=(scan[max(i - 1, 0)], scan[min(i + 1, 10)]), method="bounded",
                          options={"xatol": 0.01})
    return float(res.x), float(res.fun), float(f(A_STRAIGHT)), "PDE"


def main():
    rows = []
    for sid, drone, sensor, v, rho, sv, tau, t_det in SCENARIOS:
        D = sv**2 * tau
        lam0 = 1.0 / t_det
        pe, da = v * rho / D, lam0 * rho / v
        a, u, u_at_straight, method = analyse(pe, da)
        T_dimless = mean_flight_time(1.0 / pe)
        T = T_dimless * rho / v
        rows.append({"id": sid, "drone": drone, "sensor": sensor, "closing_speed_m_s": v,
                     "footprint_rho_m": rho, "sigma_v_m_s (assumed)": sv, "tau_s (assumed)": tau,
                     "time_to_detection_s (assumed)": t_det, "D_m2_s": D, "Pe": pe, "Da": da,
                     "response_line_m": D_RHO * rho, "area_edge_m": R_RHO * rho,
                     "best_ring_radius_m": a * rho, "straight_line_ring_radius_m": A_STRAIGHT * rho,
                     "drone_chance_best_ring": u, "drone_chance_straight_line_ring": u_at_straight,
                     "cost_of_straight_line_planning": u_at_straight - u,
                     "mean_flight_time_s": T, "outage_counts_as_long_if_over_s": T,
                     "outage_counts_as_brief_if_under_s": T / 30, "method": method})
        r = rows[-1]
        print(f"{sid}: Pe = {pe:7.2f}, Da = {da:5.2f} | ring {r['best_ring_radius_m']:6.0f} m "
              f"(straight-line plan {r['straight_line_ring_radius_m']:5.0f} m) | drone's chance "
              f"{u:.3f} (vs {u_at_straight:.3f}) | flight {T:5.0f} s | {method}", flush=True)

    with open(ROOT / "results" / "real_world_mapping.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Real-world scenarios mapped onto the model", "",
             "Six sensors on a ring; response line at 2 footprints (2ρ), area edge at 6.5ρ. "
             "Assumed inputs (time to an actionable detection, maneuver spread σ_v and "
             "correlation time τ) are illustrative; speeds and ranges are sourced.", "",
             "| ID | Drone | Sensor | v (m/s) | ρ (m) | Pe | Da | Best ring (m) | Straight-line ring (m) | "
             "Drone's chance | With straight-line ring | Flight time (s) | Long outage if > (s) | "
             "Brief if < (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['drone']} | {r['sensor']} | {r['closing_speed_m_s']} | "
                     f"{r['footprint_rho_m']} | {r['Pe']:.2g} | {r['Da']:.2g} | "
                     f"{r['best_ring_radius_m']:.0f} | {r['straight_line_ring_radius_m']:.0f} | "
                     f"{r['drone_chance_best_ring']:.3f} | {r['drone_chance_straight_line_ring']:.3f} | "
                     f"{r['mean_flight_time_s']:.0f} | {r['outage_counts_as_long_if_over_s']:.0f} | "
                     f"{r['outage_counts_as_brief_if_under_s']:.0f} |")
    (ROOT / "results" / "real_world_mapping.md").write_text("\n".join(lines) + "\n")
    print("\nSaved results/real_world_mapping.csv and results/real_world_mapping.md")


if __name__ == "__main__":
    main()
