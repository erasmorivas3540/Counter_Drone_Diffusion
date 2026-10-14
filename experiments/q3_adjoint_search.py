"""Question 3, part 2e: gradient search over free sensor positions (adjoint method).

The objective is the drone's best chance, max over entry angles of u on the area edge.
It is replaced by a smooth soft maximum with sharpness β (see cdd/adjoint.py), whose exact
gradient with respect to all 12 sensor coordinates costs one extra solve. L-BFGS-B then
minimizes it over radii in [R_MIN, R_MAX] and free angles, raising β in stages
(30 -> 100 -> 300 -> 1000 -> 3000) so the early stages see a smooth landscape and the last
ones track the true maximum.

Starts, for each drone type:
    ring        the best ring (from results/q3_best_ring.csv)
    ring+noise  the best ring with each sensor nudged by about 0.2 (in case the ring is a
                saddle point that the gradient cannot leave by symmetry)
    family:*    the best layout of each structured family (results/q3_structured_layouts.csv,
                if it exists)
    random_00.. N_RANDOM random layouts, radii uniform in [1, R_MAX], angles uniform

Every final layout is re-scored with the true maximum on the fine grid (400 x 256).
Setup as in q3_free_layout.py: K = 6, d = 2, Da = 1, three drone types.

Run:  PYTHONPATH=src python experiments/q3_adjoint_search.py     (about 45-60 min on 2 cores)
Resumable: each finished β stage of each start is saved to results/checkpoints/; rerun
the same command to continue (--fresh starts over). --random N changes the random starts.
Writes results/q3_adjoint_search.csv (one row per start), results/q3_adjoint_best.csv
and figures/q3_adjoint_search.png.
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
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint, make_key  # noqa: E402
from q3_best_ring import DA, PES, ROOT  # noqa: E402
from q3_free_layout import D_RESP, K, R_MAX, R_MIN, R_OUT, scenario_xy, worst_u  # noqa: E402

from cdd import ring_sensors  # noqa: E402
from cdd.adjoint import LayoutObjective, grad_xy_to_polar, polar_to_xy  # noqa: E402

WORKERS = int(os.environ.get("CDD_WORKERS", "2"))
BETAS = (30.0, 100.0, 300.0, 1000.0, 3000.0)
ITERS_PER_STAGE = 60
SEED = 2027

_OBJ: dict = {}


def objective_for(pe):
    if pe not in _OBJ:          # one per worker process and drone type
        _OBJ[pe] = LayoutObjective(scenario_xy(np.zeros((K, 2)), pe), nr=200, ntheta=128)
    return _OBJ[pe]


def run_start(job):
    """L-BFGS-B with β continuation from one start; each stage saved to the checkpoint."""
    label, pe, name, p0, ck_name = job
    ck = Checkpoint(ck_name)
    obj = objective_for(pe)
    p = np.asarray(p0, float)
    bounds = [(R_MIN, R_MAX)] * K + [(None, None)] * K
    n_eval = 0
    for beta in BETAS:
        key = make_key(pe, name, beta)
        if key in ck:                                    # stage already done earlier
            p = np.asarray(ck[key]["p"])
            n_eval = ck[key]["n_eval"]
            continue

        def f(q):
            radii, angles = q[:K], q[K:]
            J, g_xy, _ = obj.value_and_grad(polar_to_xy(radii, angles), beta)
            d_r, d_a = grad_xy_to_polar(g_xy, radii, angles)
            return J, np.r_[d_r, d_a]

        res = minimize(f, p, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": ITERS_PER_STAGE, "ftol": 1e-10, "gtol": 1e-7})
        p, n_eval = res.x, n_eval + res.nfev
        ck.add(key, {"p": list(map(float, p)), "n_eval": int(n_eval),
                     "u_max": obj.worst(polar_to_xy(p[:K], p[K:]))})
    return label, pe, name, list(map(float, p)), n_eval


GRID_ACCURACY = 5e-4     # coarse-grid error in worst-case u (q3_free_layout.py)


def align_to(xy, ref):
    """Rotate (and if it helps, mirror) a layout to best match a reference, for plotting only."""
    best, best_err = xy, np.inf
    for flip in (1.0, -1.0):
        z = xy[:, 0] + 1j * flip * xy[:, 1]
        for zr in ref[:, 0] + 1j * ref[:, 1]:
            for zk in z:
                if abs(zk) == 0:
                    continue
                w = z * np.exp(1j * (np.angle(zr) - np.angle(zk)))
                err = sum(np.min(np.abs(w - r)) for r in ref[:, 0] + 1j * ref[:, 1])
                if err < best_err:
                    best, best_err = np.column_stack([w.real, w.imag]), err
    return best


def load_starts(n_random):
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        rings = {r["drone"]: float(r["best_radius"]) for r in csv.DictReader(f)
                 if float(r["response_distance"]) == D_RESP and int(r["K"]) == K}
    families = {}
    path = ROOT / "results" / "q3_structured_layouts.csv"
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                xy = np.array([[float(v) for v in s.split(",")] for s in r["sensors_xy"].split()])
                families.setdefault(r["drone"], {})[r["family"]] = xy
    rng = np.random.default_rng(SEED)
    angles0 = 2 * np.pi * np.arange(K) / K
    random_starts = [np.r_[rng.uniform(1.0, R_MAX, K), rng.uniform(0, 2 * np.pi, K)]
                     for _ in range(n_random)]
    nudge = np.random.default_rng(SEED + 1).normal(0.0, 0.2, 2 * K)
    starts = {}
    for label in PES:
        a = rings[label]
        s = {"ring": np.r_[np.full(K, a), angles0],
             "ring+noise": np.r_[np.full(K, a), angles0] + nudge * np.r_[np.ones(K), np.full(K, 1 / a)]}
        for fam, xy in families.get(label, {}).items():
            # keep radii off zero so the polar angle stays well defined
            s[f"family:{fam}"] = np.r_[np.maximum(np.hypot(*xy.T), R_MIN), np.arctan2(xy[:, 1], xy[:, 0])]
        for i, p in enumerate(random_starts):
            s[f"random_{i:02d}"] = p
        starts[label] = s
    return rings, starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--random", type=int, default=30)
    args = parser.parse_args()
    ck_name = "q3_adjoint_search"
    Checkpoint(ck_name, fresh=args.fresh)          # clears the file when --fresh
    rings, starts = load_starts(args.random)
    jobs = [(label, pe, name, list(map(float, p0)), ck_name)
            for label, pe in PES.items() for name, p0 in starts[label].items()]
    print(f"{len(jobs)} starts x {len(BETAS)} stages on {WORKERS} workers", flush=True)
    t0 = time.time()
    results = []
    with Pool(WORKERS) as pool:
        for i, out in enumerate(pool.imap_unordered(run_start, jobs), 1):
            results.append(out)
            label, pe, name, p, n_eval = out
            u = worst_u(polar_to_xy(np.array(p[:K]), np.array(p[K:])), pe)
            print(f"  [{i}/{len(jobs)}] {label:22s} {name:20s} u = {u:.4f} "
                  f"({n_eval} evals, {time.time() - t0:.0f} s)", flush=True)

    # Fine-grid scores; one row per start.
    fine_ck = Checkpoint("q3_adjoint_fine", fresh=args.fresh)
    rows = []
    for label, pe, name, p, n_eval in sorted(results, key=lambda r: (r[1], r[2])):
        key = make_key(pe, name, tuple(p))
        if key not in fine_ck:
            fine_ck.add(key, worst_u(polar_to_xy(np.array(p[:K]), np.array(p[K:])), pe, fine=True))
        sensors = polar_to_xy(np.array(p[:K]), np.array(p[K:]))
        rows.append({"drone": label, "Pe": pe, "start": name, "u_fine": fine_ck[key],
                     "radii_sorted": " ".join(f"{r:.2f}" for r in np.sort(p[:K])),
                     "sensors_xy": " ".join(f"{a:.3f},{b:.3f}" for a, b in sensors),
                     "evaluations": n_eval})
    with open(ROOT / "results" / "q3_adjoint_search.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_rows, fig_data = [], {}
    for label, pe in PES.items():
        ring_u = worst_u(ring_sensors(K, rings[label]), pe, fine=True)
        mine = [r for r in rows if r["drone"] == label]
        best = min(mine, key=lambda r: r["u_fine"])
        rnd = [r["u_fine"] for r in mine if r["start"].startswith("random")]
        best_rows.append({"drone": label, "Pe": pe, "u_best_ring": ring_u, "u_adjoint_best": best["u_fine"],
                          "adjoint_minus_ring": best["u_fine"] - ring_u, "best_start": best["start"],
                          "random_starts": len(rnd),
                          "random_within_0.002_of_ring": int(sum(u <= ring_u + 0.002 for u in rnd)),
                          "random_median_u": float(np.median(rnd)) if rnd else float("nan"),
                          "radii_sorted": best["radii_sorted"], "sensors_xy": best["sensors_xy"]})
        fig_data[label] = (ring_sensors(K, rings[label]), ring_u, best, mine)
        print(f"{label:22s} ring {ring_u:.4f}; adjoint best {best['u_fine']:.4f} "
              f"({best['u_fine'] - ring_u:+.4f}) from {best['start']}; "
              f"{best_rows[-1]['random_within_0.002_of_ring']}/{len(rnd)} random starts end "
              f"within 0.002 of the ring", flush=True)
    with open(ROOT / "results" / "q3_adjoint_best.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)

    fig, axes = plt.subplots(2, len(PES), figsize=(4.3 * len(PES), 8.6),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    t = np.linspace(0, 2 * np.pi, 200)
    for col, (label, (ring, ring_u, best, mine)) in enumerate(fig_data.items()):
        ax = axes[0, col]
        ax.fill(R_OUT * np.cos(t), R_OUT * np.sin(t), color="0.95")
        ax.plot(D_RESP * np.cos(t), D_RESP * np.sin(t), color="0.5", lw=1, ls="--")
        ax.plot(*ring.T, "o", color="C0", ms=9, mfc="none", mew=1.8, label=f"best ring: u = {ring_u:.3f}")
        xy = np.array([[float(v) for v in s.split(",")] for s in best["sensors_xy"].split()])
        xy = align_to(xy, ring)
        ax.plot(*xy.T, "^", color="C3", ms=8, mfc="none", mew=1.8,
                label=f"adjoint best ({best['start']}): u = {best['u_fine']:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-R_OUT, R_OUT)
        ax.set_ylim(-R_OUT, R_OUT)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False)
        ax = axes[1, col]
        diff = 1e4 * (np.sort([r["u_fine"] for r in mine]) - ring_u)
        ax.axhspan(-1e4 * GRID_ACCURACY, 1e4 * GRID_ACCURACY, color="0.9",
                   label="coarse-grid accuracy (±5e-4)")
        ax.axhline(0.0, color="C0", ls="--", lw=1.2, label="best ring")
        ax.plot(np.arange(1, diff.size + 1), diff, ".", color="C3", label="end of each start")
        ax.set_ylim(-7, 7)
        ax.set_xlabel("start (sorted by final u)")
        ax.set_ylabel("u minus best ring (× 1e-4)")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"Gradient (adjoint) search from {len(fig_data[next(iter(PES))][3])} starts per drone "
                 f"(K = {K}, response distance {D_RESP:g}, Da = {DA:g})", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "q3_adjoint_search.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved results/q3_adjoint_search.csv, results/q3_adjoint_best.csv and "
          f"figures/q3_adjoint_search.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
