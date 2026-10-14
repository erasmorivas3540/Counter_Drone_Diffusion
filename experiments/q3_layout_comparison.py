"""Question 3, part 2 summary: every layout search side by side against the best ring.

Reads the result files of the earlier scripts (nothing is recomputed) and writes one table:

    q3_free_layout.csv          Powell from the ring and from a staggered ring
    q3_global_search.csv        differential evolution from random layouts (unseeded, 40 gen.)
    q3_structured_layouts.csv   exhaustive search over four structured families
    q3_seeded_search.csv        seeded DE (100 gen.) and Powell from 20 random layouts
    q3_adjoint_search.csv       adjoint-gradient L-BFGS-B from 36 starts per drone

All u values are the drone's best chance on the fine grid (400 x 256). "vs ring" is
u_found - u_ring: positive means worse than the ring. Differences under 5e-4 are within
the search grid's accuracy and count as a tie.

Run:  PYTHONPATH=src python experiments/q3_layout_comparison.py   (a few seconds)
Writes results/q3_layout_comparison.csv and results/q3_layout_comparison.md.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q3_best_ring import PES, ROOT  # noqa: E402

TIE = 5e-4


def read(name):
    path = ROOT / "results" / name
    if not path.exists():
        print(f"  missing {name}; skipped")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows = []
    ring = {r["drone"]: float(r["u_best_ring"]) for r in read("q3_free_layout.csv")}

    def add(drone, method, starts, u, evals, note=""):
        rows.append({"drone": drone, "method": method, "starts": starts, "u_found": u,
                     "vs_ring": u - ring[drone], "evaluations": evals, "note": note})

    for drone in PES:
        add(drone, "best ring (1-D search over radius)", "-", ring[drone], "-")
    for r in read("q3_free_layout.csv"):
        add(r["drone"], f"Powell from {r['start']}", 1, float(r["u_free"]), int(r["evaluations"]))
    for r in read("q3_global_search.csv"):
        add(r["drone"], "DE unseeded, 40 generations + Powell", "110 random", float(r["u_global"]),
            int(r["evaluations"]))
    for r in read("q3_structured_layouts.csv"):
        add(r["drone"], f"structured family: {r['family']}", "grid scan", float(r["u_family_best"]),
            int(r["evaluations"]), f"params {r['params']}")
    for r in read("q3_seeded_search.csv"):
        add(r["drone"], r["method"], "60 (5 seeded)" if "DE" in r["method"] else "20 random",
            float(r["u_found"]), int(r["evaluations"]), r["detail"])
    adj = read("q3_adjoint_search.csv")
    for drone in PES:
        mine = [r for r in adj if r["drone"] == drone]
        if not mine:
            continue
        u = np.array([float(r["u_fine"]) for r in mine])
        rnd = np.array([float(r["u_fine"]) for r in mine if r["start"].startswith("random")])
        add(drone, "adjoint gradient (L-BFGS-B, soft-max)", f"{len(mine)} ({len(rnd)} random)",
            float(u.min()), int(sum(int(r["evaluations"]) for r in mine)),
            f"{int((rnd <= ring[drone] + TIE).sum())}/{len(rnd)} random starts tie the ring; "
            f"worst start {u.max() - ring[drone]:+.1e}")

    with open(ROOT / "results" / "q3_layout_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Q3: does any 6-sensor layout beat the best ring?", "",
             "K = 6, response distance d = 2, Da = 1. u = drone's best chance (fine grid). "
             f"'vs ring' > 0 means worse than the ring; |vs ring| < {TIE:g} is a tie "
             "(search-grid accuracy).", ""]
    for drone in PES:
        lines += [f"## {drone}  (best ring u = {ring[drone]:.4f})", "",
                  "| Method | Starts | u found | vs ring | Evaluations | Note |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            if r["drone"] != drone:
                continue
            verdict = "tie" if abs(r["vs_ring"]) < TIE else ("better" if r["vs_ring"] < 0 else "worse")
            vs = "-" if r["method"].startswith("best ring") else f"{r['vs_ring']:+.4f} ({verdict})"
            lines.append(f"| {r['method']} | {r['starts']} | {r['u_found']:.4f} | {vs} | "
                         f"{r['evaluations']} | {r['note']} |")
        lines.append("")
    better = [r for r in rows if r["vs_ring"] < -TIE]
    lines.append("**Verdict:** " + ("no method found a layout better than the best ring beyond "
                                    "grid accuracy." if not better else
                                    f"{len(better)} result(s) beat the ring: check them on a finer grid."))
    (ROOT / "results" / "q3_layout_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
