"""Question 3, part 2d: exhaustive search over structured families of 6-sensor layouts.

Each family has two or three numbers. Each one is scanned on a grid, and the best few
grid points are polished with Powell inside the family:

    staggered   3 sensors on a ring of radius a1 and 3 on a ring of radius a2 (a1 <= a2),
                the outer three turned by phi in [0, pi/3]. phi = pi/3 with a1 = a2 is the
                plain 6-ring, so this family contains the ring.
    zigzag      the staggered family's phi = pi/3 slice: equal angles, alternating radii.
    five_plus   5 sensors on a ring of radius a and 1 at radius b in [0, R_MAX] at angle
                psi in [0, pi/5] from a ring sensor (b = 0 is "5 on a ring + 1 at the center").
    weighted    4 sensors on a ring of radius a1 and 2 opposite each other at radius a2 at
                angle psi in [0, pi/4] from a ring sensor; a2 can be inside or outside a1.

The angle ranges cover every distinct layout once, by the families' symmetries.

Objective, setup and grids as in q3_free_layout.py: K = 6, response distance d = 2,
Da = 1, three drone types. Search on the coarse grid (200 x 128); final layouts are
re-scored on the fine grid (400 x 256) and compared with the best ring from
results/q3_best_ring.csv.

Run:  PYTHONPATH=src python experiments/q3_structured_layouts.py   (about 25 min on 2 cores)
Resumable: progress is saved to results/checkpoints/; rerun the same command to pick up
where it stopped (--fresh starts over).
Writes results/q3_structured_layouts.csv, results/q3_structured_scan.csv and
figures/q3_structured_layouts.png.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q3_best_ring import DA, PES, ROOT  # noqa: E402
from q3_free_layout import D_RESP, K, R_MAX, R_MIN, R_OUT, worst_u  # noqa: E402
from checkpoint import Checkpoint, make_key  # noqa: E402

from cdd import ring_sensors  # noqa: E402

WORKERS = int(os.environ.get("CDD_WORKERS", "2"))
N_POLISH = 3          # grid points per family and drone that get a Powell polish
POLISH_EVALS = 150


def _ring(n, a, phase=0.0):
    return ring_sensors(n, a, phase) if n else np.zeros((0, 2))


def staggered(a1, a2, phi):
    return np.vstack([_ring(3, a1), _ring(3, a2, phi)])


def zigzag(a1, a2):
    return staggered(a1, a2, np.pi / 3)


def five_plus(a, b, psi):
    return np.vstack([_ring(5, a), [[b * np.cos(psi), b * np.sin(psi)]]])


def weighted(a1, a2, psi):
    return np.vstack([_ring(4, a1), _ring(2, a2, psi)])


RADII = np.linspace(R_MIN, R_MAX, 15)
FAMILIES = {
    # name: (layout function, parameter bounds, scan grid)
    "staggered": (staggered, [(R_MIN, R_MAX), (R_MIN, R_MAX), (0.0, np.pi / 3)],
                  [p for p in itertools.product(RADII, RADII, np.linspace(0, np.pi / 3, 5))
                   if p[0] <= p[1]]),
    "zigzag": (zigzag, [(R_MIN, R_MAX), (R_MIN, R_MAX)],
               [p for p in itertools.product(np.linspace(R_MIN, R_MAX, 24), repeat=2)
                if p[0] <= p[1]]),
    "five_plus": (five_plus, [(R_MIN, R_MAX), (0.0, R_MAX), (0.0, np.pi / 5)],
                  list(itertools.product(RADII, np.linspace(0.0, R_MAX, 16),
                                         np.linspace(0, np.pi / 5, 3)))),
    "weighted": (weighted, [(R_MIN, R_MAX), (R_MIN, R_MAX), (0.0, np.pi / 4)],
                 list(itertools.product(RADII, RADII, np.linspace(0, np.pi / 4, 4)))),
}


def _score(job):
    family, params, pe = job
    return job, worst_u(FAMILIES[family][0](*params), pe)


def _polish_job(job):
    family, start, pe = job
    x, fun, nfev = polish(family, start, pe)
    return job, (list(map(float, x)), float(fun), int(nfev))


def polish(family, start, pe):
    fn, bounds, _ = FAMILIES[family]
    res = minimize(lambda p: worst_u(fn(*p), pe), np.asarray(start, float), method="Powell",
                   bounds=bounds, options={"maxfev": POLISH_EVALS, "xtol": 1e-3, "ftol": 1e-5})
    return res.x, res.fun, res.nfev


def best_ring_radii():
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        return {r["drone"]: float(r["best_radius"]) for r in csv.DictReader(f)
                if float(r["response_distance"]) == D_RESP and int(r["K"]) == K}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="ignore saved checkpoints")
    args = parser.parse_args()
    rings = best_ring_radii()
    t0 = time.time()

    # Scan: every layout is saved as soon as it is scored, so a rerun resumes.
    scan_ck = Checkpoint("q3_structured_scan", fresh=args.fresh)
    jobs = [(fam, p, pe) for fam, (_, _, grid) in FAMILIES.items() for p in grid
            for pe in PES.values()]
    todo = [j for j in jobs if make_key(*j) not in scan_ck]
    print(f"Scan: {len(jobs)} layouts, {len(jobs) - len(todo)} already saved, "
          f"{len(todo)} to do on {WORKERS} workers ...", flush=True)
    with Pool(WORKERS) as pool:
        for i, (job, u) in enumerate(pool.imap_unordered(_score, todo, chunksize=4), 1):
            scan_ck.add(make_key(*job), u)
            if i % 250 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    scores = [scan_ck[make_key(*j)] for j in jobs]
    print(f"  scan done ({time.time() - t0:.0f} s)", flush=True)

    with open(ROOT / "results" / "q3_structured_scan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "Pe", "params", "u_coarse"])
        for (fam, p, pe), s in zip(jobs, scores):
            w.writerow([fam, pe, " ".join(f"{x:.4f}" for x in p), s])

    # Top grid points per family and drone, then a Powell polish (each run saved).
    polish_ck = Checkpoint("q3_structured_polish", fresh=args.fresh)
    polish_jobs = []
    for fam in FAMILIES:
        for pe in PES.values():
            cand = sorted((s, p) for (f_, p, pe_), s in zip(jobs, scores) if f_ == fam and pe_ == pe)
            polish_jobs += [(fam, p, pe) for _, p in cand[:N_POLISH]]
    todo = [j for j in polish_jobs if make_key(*j) not in polish_ck]
    print(f"Polish: {len(polish_jobs)} runs, {len(todo)} to do", flush=True)
    with Pool(WORKERS) as pool:
        for job, res in pool.imap_unordered(_polish_job, todo):
            polish_ck.add(make_key(*job), res)
            print(f"  polished {job[0]} Pe={job[2]:g}: u = {res[1]:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    polished = [tuple(polish_ck[make_key(*j)]) for j in polish_jobs]
    print(f"  polish done ({time.time() - t0:.0f} s)", flush=True)

    rows, best_layouts = [], {}
    for label, pe in PES.items():
        ring_u = worst_u(ring_sensors(K, rings[label]), pe, fine=True)
        best_layouts[(label, "best ring")] = (ring_sensors(K, rings[label]), ring_u)
        for fam, (fn, _, grid) in FAMILIES.items():
            res = [(fun, np.array(x), nfev) for (f_, _, pe_), (x, fun, nfev) in zip(polish_jobs, polished)
                   if f_ == fam and pe_ == pe]
            fun, x, nfev = min(res, key=lambda t: t[0])
            sensors = fn(*x)
            u = worst_u(sensors, pe, fine=True)
            best_layouts[(label, fam)] = (sensors, u)
            rows.append({"drone": label, "Pe": pe, "family": fam,
                         "params": " ".join(f"{v:.3f}" for v in x),
                         "u_best_ring": ring_u, "best_ring_radius": rings[label],
                         "u_family_best": u, "family_minus_ring": u - ring_u,
                         "sensors_xy": " ".join(f"{a:.3f},{b:.3f}" for a, b in sensors),
                         "evaluations": len(grid) + N_POLISH * POLISH_EVALS})
            print(f"{label:22s} {fam:10s} params {rows[-1]['params']:22s} u = {u:.4f} "
                  f"(ring {ring_u:.4f}, {u - ring_u:+.4f})", flush=True)

    with open(ROOT / "results" / "q3_structured_layouts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    styles = {"best ring": ("o", "C0"), "staggered": ("^", "C3"), "zigzag": ("v", "C1"),
              "five_plus": ("s", "C2"), "weighted": ("D", "C4")}
    fig, axes = plt.subplots(1, len(PES), figsize=(4.3 * len(PES), 5.8))
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, label in zip(axes, PES):
        ax.fill(R_OUT * np.cos(t), R_OUT * np.sin(t), color="0.95")
        ax.plot(D_RESP * np.cos(t), D_RESP * np.sin(t), color="0.5", lw=1, ls="--")
        for (lab, kind), (sensors, u) in best_layouts.items():
            if lab == label:
                m, c = styles[kind]
                ax.plot(*sensors.T, m, color=c, ms=8, mfc="none", mew=1.6,
                        label=f"{kind}: u = {u:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-R_OUT, R_OUT)
        ax.set_ylim(-R_OUT, R_OUT)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(f"Best layout in each structured family vs the best ring (K = {K}, "
                 f"response distance {D_RESP:g}, Da = {DA:g})", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(ROOT / "figures" / "q3_structured_layouts.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved results/q3_structured_layouts.csv and figures/q3_structured_layouts.png "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
