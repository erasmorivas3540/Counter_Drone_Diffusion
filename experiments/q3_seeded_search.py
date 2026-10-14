"""Question 3, part 2f: seeded evolutionary search and multi-start local search.

Two searches over free sensor positions, both on the coarse grid (200 x 128) with the
true worst-case u as the objective, final layouts re-scored on the fine grid (400 x 256):

  de      Differential evolution (DE/rand/1/bin, F dithered in [0.5, 1], CR = 0.7) with a
          60-member population SEEDED with the best ring, the best layout of each
          structured family (results/q3_structured_layouts.csv) and 55 random layouts
          (Latin hypercube), run for 100 generations, then a Powell polish. rand/1 keeps the
          population exploring instead of collapsing onto the seeded ring at once;
          selection is greedy, so the seeds are never lost. Variables: radius and angle of
          6 sensors with sensor 0 at angle 0 (11 numbers).
  powell  Powell's method from 20 random layouts per drone type (12 numbers, up to 400
          evaluations each), the same local method q3_free_layout.py used from the ring.

Setup as in q3_free_layout.py: K = 6, d = 2, Da = 1, three drone types.

Run:  PYTHONPATH=src python experiments/q3_seeded_search.py              (both, ~1.5 h, 2 cores)
      PYTHONPATH=src python experiments/q3_seeded_search.py --part de    (or --part powell)
Resumable: DE saves its whole state (population, scores, generation, random-number state)
after every generation, Powell saves every finished start; rerun the same command to
continue exactly where it stopped. --fresh starts over.
Writes results/q3_seeded_search.csv and figures/q3_seeded_search.png.
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
from scipy.stats import qmc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint, make_key  # noqa: E402
from q3_adjoint_search import align_to  # noqa: E402
from q3_best_ring import DA, PES, ROOT  # noqa: E402
from q3_free_layout import D_RESP, K, R_MAX, R_MIN, R_OUT, worst_u  # noqa: E402

from cdd import ring_sensors  # noqa: E402

WORKERS = int(os.environ.get("CDD_WORKERS", "2"))
SEED = 2028
POP, GENERATIONS, CR = 60, 100, 0.7
N_POWELL, POWELL_EVALS = 20, 400
LO = np.r_[np.full(K, R_MIN), np.zeros(K - 1)]
HI = np.r_[np.full(K, R_MAX), np.full(K - 1, 2 * np.pi)]


def de_xy(p):
    """p = [r_0..r_5, angle_1..angle_5]; sensor 0 at angle 0."""
    radii, angles = p[:K], np.r_[0.0, p[K:]]
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def xy_to_de(xy):
    """Any layout -> DE vector: rotate so the sensor with the largest radius is at angle 0."""
    r = np.maximum(np.hypot(*xy.T), R_MIN)
    a = np.arctan2(xy[:, 1], xy[:, 0])
    i = int(np.argmax(r))
    order = [i] + [j for j in range(K) if j != i]
    r, a = r[order], np.mod(a[order] - a[i], 2 * np.pi)
    return np.r_[np.minimum(r, R_MAX), a[1:]]


def repair(p):
    """Radii reflected back into bounds, angles wrapped."""
    r = p[:K].copy()
    r = np.where(r < R_MIN, 2 * R_MIN - r, r)
    r = np.where(r > R_MAX, 2 * R_MAX - r, r)
    return np.r_[np.clip(r, R_MIN, R_MAX), np.mod(p[K:], 2 * np.pi)]


def _de_eval(args):
    p, pe = args
    return worst_u(de_xy(np.asarray(p)), pe)


def _powell(args):
    label, pe, i, p0 = args
    bounds = [(R_MIN, R_MAX)] * K + [(None, None)] * K
    f = lambda q: worst_u(np.column_stack([q[:K] * np.cos(q[K:]), q[:K] * np.sin(q[K:])]), pe)  # noqa: E731
    res = minimize(f, np.asarray(p0), method="Powell", bounds=bounds,
                   options={"maxfev": POWELL_EVALS, "xtol": 1e-3, "ftol": 1e-5})
    return label, pe, i, list(map(float, res.x)), float(res.fun), int(res.nfev)


def load_seeds():
    with open(ROOT / "results" / "q3_best_ring.csv") as f:
        rings = {r["drone"]: float(r["best_radius"]) for r in csv.DictReader(f)
                 if float(r["response_distance"]) == D_RESP and int(r["K"]) == K}
    seeds = {label: {"ring": ring_sensors(K, rings[label])} for label in PES}
    path = ROOT / "results" / "q3_structured_layouts.csv"
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                xy = np.array([[float(v) for v in s.split(",")] for s in r["sensors_xy"].split()])
                seeds[r["drone"]][f"family:{r['family']}"] = xy
    else:
        print("  note: results/q3_structured_layouts.csv not found; seeding with the ring only")
    return rings, seeds


def run_de(label, pe, seeds, pool, ck, t0):
    key = make_key("de", pe)
    state = ck.get(key)
    if state is None:
        rng = np.random.default_rng([SEED, int(pe * 10)])
        seeded = [xy_to_de(xy) for xy in seeds.values()]
        lhs = qmc.LatinHypercube(d=2 * K - 1, seed=rng).random(POP - len(seeded))
        pop = np.vstack(seeded + [LO + lhs * (HI - LO)])
        energy = np.array(pool.map(_de_eval, [(p, pe) for p in pop]))
        state = {"gen": 0, "pop": pop.tolist(), "energy": energy.tolist(),
                 "rng": rng.bit_generator.state, "nfev": POP, "seed_names": list(seeds),
                 "history": [float(energy.min())]}
        ck.add(key, state)
    pop, energy = np.array(state["pop"]), np.array(state["energy"])
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    while state["gen"] < GENERATIONS:
        trials = np.empty_like(pop)
        for i in range(POP):
            a, b, c = rng.choice([j for j in range(POP) if j != i], 3, replace=False)
            F = rng.uniform(0.5, 1.0)
            mutant = pop[a] + F * (pop[b] - pop[c])
            cross = rng.random(pop.shape[1]) < CR
            cross[rng.integers(pop.shape[1])] = True
            trials[i] = repair(np.where(cross, mutant, pop[i]))
        e_trial = np.array(pool.map(_de_eval, [(p, pe) for p in trials]))
        better = e_trial <= energy
        pop[better], energy[better] = trials[better], e_trial[better]
        state = {**state, "gen": state["gen"] + 1, "pop": pop.tolist(), "energy": energy.tolist(),
                 "rng": rng.bit_generator.state, "nfev": state["nfev"] + POP,
                 "history": state["history"] + [float(energy.min())]}
        ck.add(key, state)
        if state["gen"] % 10 == 0:
            print(f"  DE {label:22s} generation {state['gen']:3d}: best u = {energy.min():.4f} "
                  f"(from seed: {_origin(state, pop)})  ({time.time() - t0:.0f} s)", flush=True)
    return state


def _origin(state, pop):
    return "yes" if np.argmin(state["energy"]) < len(state["seed_names"]) else "no"


def polish_de(label, pe, state, ck):
    key = make_key("de_polish", pe)
    if key not in ck:
        energy = np.array(state["energy"])
        p0 = np.array(state["pop"])[int(np.argmin(energy))]
        bounds = list(zip(LO, HI))
        res = minimize(lambda p: worst_u(de_xy(p), pe), p0, method="Powell", bounds=bounds,
                       options={"maxfev": POWELL_EVALS, "xtol": 1e-3, "ftol": 1e-5})
        x = res.x if res.fun < energy.min() else p0
        ck.add(key, {"p": list(map(float, x)), "nfev": int(res.nfev),
                     "best_member": int(np.argmin(energy))})
    return ck[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("de", "powell", "both"), default="both")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    ck = Checkpoint("q3_seeded_search", fresh=args.fresh)
    rings, seeds = load_seeds()
    t0 = time.time()
    rows, panels = [], {label: {} for label in PES}

    with Pool(WORKERS) as pool:
        for label, pe in PES.items():
            ring_u = ck.get(make_key("ring_fine", pe))
            if ring_u is None:
                ring_u = worst_u(ring_sensors(K, rings[label]), pe, fine=True)
                ck.add(make_key("ring_fine", pe), ring_u)
            panels[label]["best ring"] = (ring_sensors(K, rings[label]), ring_u)

            if args.part in ("de", "both"):
                print(f"DE for {label}: seeds {list(seeds[label])}", flush=True)
                state = run_de(label, pe, seeds[label], pool, ck, t0)
                pol = polish_de(label, pe, state, ck)
                fkey = make_key("de_fine", pe)
                if fkey not in ck:
                    ck.add(fkey, worst_u(de_xy(np.array(pol["p"])), pe, fine=True))
                xy, u = de_xy(np.array(pol["p"])), ck[fkey]
                from_seed = pol["best_member"] < len(state["seed_names"])
                panels[label]["seeded DE"] = (xy, u)
                rows.append({"drone": label, "Pe": pe, "method": "seeded DE + Powell",
                             "u_best_ring": ring_u, "u_found": u, "found_minus_ring": u - ring_u,
                             "detail": (f"best member came from seed "
                                        f"{state['seed_names'][pol['best_member']]}" if from_seed
                                        else "best member came from a random layout"),
                             "radii_sorted": " ".join(f"{r:.2f}" for r in np.sort(np.hypot(*xy.T))),
                             "sensors_xy": " ".join(f"{a:.3f},{b:.3f}" for a, b in xy),
                             "evaluations": state["nfev"] + pol["nfev"]})
                print(f"  DE result {label}: u = {u:.4f} vs ring {ring_u:.4f} ({u - ring_u:+.4f}); "
                      f"{rows[-1]['detail']}", flush=True)

            if args.part in ("powell", "both"):
                rng = np.random.default_rng([SEED, 7, int(pe * 10)])
                starts = [np.r_[rng.uniform(1.0, R_MAX, K), rng.uniform(0, 2 * np.pi, K)]
                          for _ in range(N_POWELL)]
                todo = [(label, pe, i, list(map(float, p))) for i, p in enumerate(starts)
                        if make_key("powell", pe, i) not in ck]
                print(f"Powell for {label}: {N_POWELL} starts, {len(todo)} to do", flush=True)
                for lab, pe_, i, x, fun, nfev in pool.imap_unordered(_powell, todo):
                    ck.add(make_key("powell", pe_, i), {"x": x, "fun": fun, "nfev": nfev})
                    print(f"  Powell {lab:22s} start {i:2d}: u = {fun:.4f} "
                          f"({time.time() - t0:.0f} s)", flush=True)
                res = [ck[make_key("powell", pe, i)] for i in range(N_POWELL)]
                funs = np.array([r["fun"] for r in res])
                best = res[int(np.argmin(funs))]
                x = np.array(best["x"])
                xy = np.column_stack([x[:K] * np.cos(x[K:]), x[:K] * np.sin(x[K:])])
                fkey = make_key("powell_fine", pe)
                if fkey not in ck:
                    ck.add(fkey, worst_u(xy, pe, fine=True))
                u = ck[fkey]
                panels[label]["multi-start Powell"] = (xy, u)
                rows.append({"drone": label, "Pe": pe, "method": f"Powell, {N_POWELL} random starts",
                             "u_best_ring": ring_u, "u_found": u, "found_minus_ring": u - ring_u,
                             "detail": (f"median start ends at {np.median(funs):.4f}; "
                                        f"{int((funs <= ring_u + 0.002).sum())}/{N_POWELL} within "
                                        f"0.002 of the ring"),
                             "radii_sorted": " ".join(f"{r:.2f}" for r in np.sort(x[:K])),
                             "sensors_xy": " ".join(f"{a:.3f},{b:.3f}" for a, b in xy),
                             "evaluations": int(sum(r["nfev"] for r in res))})
                print(f"  Powell result {label}: best u = {u:.4f} vs ring {ring_u:.4f} "
                      f"({u - ring_u:+.4f}); {rows[-1]['detail']}", flush=True)

    out = ROOT / "results" / "q3_seeded_search.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    styles = {"best ring": ("o", "C0"), "seeded DE": ("^", "C3"), "multi-start Powell": ("s", "C2")}
    fig, axes = plt.subplots(1, len(PES), figsize=(4.3 * len(PES), 5.5))
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, (label, layouts) in zip(axes, panels.items()):
        ax.fill(R_OUT * np.cos(t), R_OUT * np.sin(t), color="0.95")
        ax.plot(D_RESP * np.cos(t), D_RESP * np.sin(t), color="0.5", lw=1, ls="--")
        ring_xy = layouts["best ring"][0]
        for kind, (xy, u) in layouts.items():
            m, c = styles[kind]
            xy = align_to(xy, ring_xy)
            ax.plot(*xy.T, m, color=c, ms=8, mfc="none", mew=1.6, label=f"{kind}: u = {u:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-R_OUT, R_OUT)
        ax.set_ylim(-R_OUT, R_OUT)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(f"Seeded evolutionary search and multi-start Powell vs the best ring "
                 f"(K = {K}, response distance {D_RESP:g}, Da = {DA:g})", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(ROOT / "figures" / "q3_seeded_search.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved {out.relative_to(ROOT)} and figures/q3_seeded_search.png "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
