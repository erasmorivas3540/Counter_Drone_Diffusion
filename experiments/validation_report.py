"""Validation report: every rung of the ladder with more paths than the tests use.

    python experiments/validation_report.py            # about 3 minutes
    python experiments/validation_report.py --quick    # about 1 minute, 4x fewer paths

Writes results/validation.md (tables for the paper), results/validation.json (raw
numbers) and figures/validation.{png,pdf} (grid convergence and time-step bias).
Seeds differ from the test suite's, so this is independent evidence.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cdd import (Scenario, ring_sensors, simulate_u, simulate_u_1d, solve,  # noqa: E402
                 u_1d, u_bessel, u_radial)

ROOT = Path(__file__).resolve().parents[1]


def on_circle(radius: float, angle: float) -> np.ndarray:
    return radius * np.array([np.cos(angle), np.sin(angle)])


def mc_row(res, ref) -> dict:
    return {"mc": res.estimate, "half_width": res.half_width, "z": res.z_score(ref),
            "n_paths": res.n_paths, "timed_out": res.timed_out}


def rung1(n: int) -> list[dict]:
    rows = []
    for k, sigma in enumerate((0.5, 1.0, 2.0)):
        exact = float(u_1d(1.0, 1.0, sigma, 1.0))
        res = simulate_u_1d(1.0, 1.0, sigma, 1.0, dt=1e-2, n_paths=n, seed=1000 + k)
        rows.append({"case": f"1D, sigma = {sigma}", "reference": exact, **mc_row(res, exact)})
    return rows


def step_bias(n: int) -> dict:
    """1D, sigma = 1: estimate minus exact for a plain simulator and the corrected one."""
    exact = float(u_1d(1.0, 1.0, 1.0, 1.0))
    dts = [0.1, 0.05, 0.02, 0.01]
    out = {"dt": dts, "plain": [], "plain_hw": [], "corrected": [], "corrected_hw": []}
    for k, dt in enumerate(dts):
        for key, bridge in (("plain", False), ("corrected", True)):
            res = simulate_u_1d(1.0, 1.0, 1.0, 1.0, dt=dt, n_paths=n, seed=1100 + k, bridge=bridge)
            out[key].append(res.estimate - exact)
            out[key + "_hw"].append(res.half_width)
    return out


def rung2(n: int) -> list[dict]:
    rows = []
    for k, outer in enumerate(("reflect", "absorb")):
        scn = Scenario(r0=0.5, R=4.0, v=0.0, sigma=1.0, lam0=1.0, kernel="uniform", outer=outer)
        exact = float(u_bessel(1.5, 0.5, 4.0, 1.0, 1.0, outer))
        pde = float(solve(scn, nr=200, ntheta=16)(on_circle(1.5, 0.3))[0])
        res = simulate_u(on_circle(1.5, 0.3), scn, dt=1e-2, n_paths=n, seed=1200 + k, w_min=1e-6)
        rows.append({"case": f"no drift, {outer}ing edge, r = 1.5", "reference": exact,
                     "pde": pde, **mc_row(res, exact)})
    return rows


def rung3(n: int) -> list[dict]:
    rows = []
    for k, outer in enumerate(("reflect", "absorb")):
        scn = Scenario(r0=0.5, R=4.0, v=1.0, sigma=1.0, lam0=0.5, kernel="uniform", outer=outer)
        ref = float(u_radial(2.5, 0.5, 4.0, 1.0, 1.0, 0.5, outer))
        pde = float(solve(scn, nr=200, ntheta=16)(on_circle(2.5, 2.0))[0])
        res = simulate_u(on_circle(2.5, 2.0), scn, dt=1e-2, n_paths=n, seed=1300 + k)
        rows.append({"case": f"radial drift, {outer}ing edge, r = 2.5", "reference": ref,
                     "pde": pde, **mc_row(res, ref)})
    return rows


def rung4() -> dict:
    scn = Scenario(r0=0.5, R=4.0, v=1.0, sigma=1.0, lam0=0.5, kernel="uniform", outer="reflect")
    grids = [25, 50, 100, 200, 400, 800]
    out = {"nr": grids, "h": [(scn.R - scn.r0) / nr for nr in grids]}
    for scheme in ("central", "hybrid", "upwind"):
        errs = []
        for nr in grids:
            sol = solve(scn, nr=nr, ntheta=8, scheme=scheme)
            errs.append(float(np.abs(sol.u[:, 0] - u_radial(sol.r, 0.5, 4.0, 1.0, 1.0, 0.5)).max()))
        out[scheme] = errs
        out[scheme + "_order"] = float(np.polyfit(np.log(out["h"]), np.log(errs), 1)[0])
    return out


def ring_case() -> Scenario:
    return Scenario(r0=0.5, R=5.0, v=1.0, sigma=1.0, lam0=1.0, rho=1.0,
                    sensors=ring_sensors(4, 2.0), kernel="gaussian", outer="reflect")


def rung5(n: int) -> tuple[list[dict], dict]:
    scn = ring_case()
    angles = {"over a sensor": 0.0, "pi/8 off": np.pi / 8, "between sensors": np.pi / 4}
    pts = np.array([on_circle(4.5, a) for a in angles.values()])
    grids = [(100, 128), (200, 256), (400, 512)]
    vals = [solve(scn, nr=nr, ntheta=nth)(pts) for nr, nth in grids]
    d1, d2 = np.abs(vals[0] - vals[1]), np.abs(vals[1] - vals[2])
    conv = {"grids": grids, "values": [v.tolist() for v in vals],
            "orders": np.log2(d1 / d2).tolist()}
    rows = []
    for k, (name, a) in enumerate(angles.items()):
        pde = float(vals[-1][k])
        res = simulate_u(pts[k], scn, dt=1e-2, n_paths=n, seed=1500 + k)
        rows.append({"case": f"4-sensor ring, start r = 4.5, {name}", "reference": pde,
                     "pde": pde, **mc_row(res, pde)})
    return rows, conv


def markdown(results: dict) -> str:
    lines = ["# Validation report", "",
             f"Monte Carlo: {results['n_paths']:,} paths per case, time step 0.01, "
             "95% half-widths; z = (reference - estimate) / standard error.", ""]

    def table(title, rows, with_pde):
        lines.extend([f"## {title}", ""])
        head = "| Case | Reference |" + (" PDE |" if with_pde else "") + " Monte Carlo | z |"
        lines.extend([head, "|" + "---|" * (head.count("|") - 1)])
        for r in rows:
            pde = f" {r['pde']:.5f} |" if with_pde else ""
            lines.append(f"| {r['case']} | {r['reference']:.5f} |{pde} "
                         f"{r['mc']:.5f} ± {r['half_width']:.5f} | {r['z']:+.2f} |")
        lines.append("")

    table("Rung 1: 1D closed form", results["rung1"], False)
    table("Rung 2: no drift, Bessel closed form (PDE grid 200 x 16)", results["rung2"], True)
    table("Rung 3: radial drift, solve_bvp reference (PDE grid 200 x 16)", results["rung3"], True)
    table("Rung 5: sensor ring, PDE reference (grid 400 x 512)", results["rung5"], True)

    c = results["rung4"]
    lines.extend(["## Rung 4: grid convergence (radial drift case, max error over the grid)", "",
                  "| nr | central | hybrid | upwind |", "|---|---|---|---|"])
    for i, nr in enumerate(c["nr"]):
        lines.append(f"| {nr} | {c['central'][i]:.2e} | {c['hybrid'][i]:.2e} | {c['upwind'][i]:.2e} |")
    lines.extend(["", f"Fitted orders: central {c['central_order']:.2f}, hybrid "
                  f"{c['hybrid_order']:.2f}, upwind {c['upwind_order']:.2f}.",
                  "Sensor ring self-convergence orders (three grids, three starts): "
                  + ", ".join(f"{o:.2f}" for o in results["rung5_convergence"]["orders"]) + ".", ""])

    b = results["step_bias"]
    lines.extend(["## Time-step bias, 1D, sigma = 1 (estimate minus exact)", "",
                  "| dt | plain | with bridge check and half-step refund |", "|---|---|---|"])
    for i, dt in enumerate(b["dt"]):
        lines.append(f"| {dt} | {b['plain'][i]:+.4f} ± {b['plain_hw'][i]:.4f} | "
                     f"{b['corrected'][i]:+.4f} ± {b['corrected_hw'][i]:.4f} |")
    zs = [r["z"] for k in ("rung1", "rung2", "rung3", "rung5") for r in results[k]]
    lines.extend(["", f"All {len(zs)} Monte Carlo comparisons: largest |z| = "
                  f"{max(abs(z) for z in zs):.2f}, mean z = {np.mean(zs):+.2f}.", ""])
    return "\n".join(lines)


def figure(results: dict, path: Path) -> None:
    c, b = results["rung4"], results["step_bias"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    h = np.array(c["h"])
    ax1.loglog(h, c["central"], "o-", label=f"central (order {c['central_order']:.2f})")
    ax1.loglog(h, c["upwind"], "s-", label=f"upwind (order {c['upwind_order']:.2f})")
    ax1.loglog(h, 0.3 * c["central"][0] * (h / h[0]) ** 2, "k:", lw=1, label="slope 2")
    ax1.loglog(h, 0.3 * c["upwind"][0] * (h / h[0]), "k--", lw=1, label="slope 1")
    ax1.set_xlabel("radial grid spacing h")
    ax1.set_ylabel("max |PDE - exact|")
    ax1.set_title("(a) PDE grid convergence")
    ax1.legend(fontsize=8)
    dt = np.array(b["dt"])
    ax2.errorbar(dt, b["plain"], yerr=b["plain_hw"], fmt="s-", capsize=3, label="plain Euler")
    ax2.errorbar(dt, b["corrected"], yerr=b["corrected_hw"], fmt="o-", capsize=3,
                 label="bridge check + half-step refund")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("time step dt")
    ax2.set_ylabel("Monte Carlo minus exact")
    ax2.set_title("(b) Simulator time-step bias (1D)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="4x fewer Monte Carlo paths")
    args = parser.parse_args()
    n = 50_000 if args.quick else 200_000
    results = {"n_paths": n}
    steps = [("rung1", lambda: rung1(n)), ("step_bias", lambda: step_bias(n)),
             ("rung2", lambda: rung2(n)), ("rung3", lambda: rung3(n)), ("rung4", rung4)]
    for name, fn in steps:
        t = time.time()
        results[name] = fn()
        print(f"{name:10s} done in {time.time() - t:5.1f} s", flush=True)
    t = time.time()
    results["rung5"], results["rung5_convergence"] = rung5(n)
    print(f"{'rung5':10s} done in {time.time() - t:5.1f} s", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    report = markdown(results)
    (ROOT / "results" / "validation.md").write_text(report)
    (ROOT / "results" / "validation.json").write_text(json.dumps(results, indent=2))
    figure(results, ROOT / "figures" / "validation")
    print()
    print(report)


if __name__ == "__main__":
    main()
