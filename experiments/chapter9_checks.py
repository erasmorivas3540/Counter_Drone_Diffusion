"""Checks that a large z-score in chapter9_dirichlet_demo.py is noise, not a bug.

Two tests on Part 2 (Brownian motion, g = cos(theta) on the area edge):

  1. Seed test: rerun with several seeds. A bug pushes every seed the same way;
     noise scatters the z-scores around 0, mostly between -2 and 2.
  2. Time-step test: rerun at three time steps. Error from the time stepping grows
     with the step; if "Monte Carlo - exact" shows no trend and stays inside its
     interval, that error is smaller than the noise.

Both tests also print the share of paths that exit through the site. It should match
Part 1's exact answer, ln(R/r)/ln(R/r0) = 0.4717, at any starting angle.

Run from the project folder:
    python experiments/chapter9_checks.py           # about 5 minutes
    python experiments/chapter9_checks.py --quick   # about 1 minute, 4x fewer paths
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chapter9_dirichlet_demo import R, R0, exit_point  # noqa: E402

START_R, START_ANGLE = 1.5, 0.7


def part2(dt, n, seed):
    """Monte Carlo estimate of Part 2, its standard error, and the site-exit share."""
    x0 = START_R * np.array([np.cos(START_ANGLE), np.sin(START_ANGLE)])
    exits, at_site = exit_point(x0, dt=dt, n=n, seed=seed)
    g = np.where(at_site, 0.0, np.cos(np.arctan2(exits[:, 1], exits[:, 0])))
    return g.mean(), g.std(ddof=1) / np.sqrt(n), at_site.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="4x fewer paths")
    args = parser.parse_args()
    n = 50_000 if args.quick else 200_000

    exact = (START_R - R0**2 / START_R) * np.cos(START_ANGLE) / (R - R0**2 / R)
    site_exact = np.log(R / START_R) / np.log(R / R0)
    print(f"Part 2 exact value {exact:.4f}; exact site-exit share {site_exact:.4f}; "
          f"{n:,} paths per run\n")

    print("1. Seed test (time step 0.004)")
    zs, seeds = [], (2, 11, 12, 13)
    for seed in seeds:
        t = time.time()
        est, se, site = part2(4e-3, n, seed)
        zs.append((exact - est) / se)
        print(f"   seed {seed:2d}: {est:.4f} +/- {1.96 * se:.4f}   z = {zs[-1]:+.2f}   "
              f"site share {site:.4f}   ({time.time() - t:.0f} s)")
    print(f"   mean z = {np.mean(zs):+.2f} (near 0 means no bias)\n")

    print("2. Time-step test (seed 21)")
    dts, errs, bars = (8e-3, 4e-3, 2e-3), [], []
    for dt in dts:
        t = time.time()
        est, se, site = part2(dt, n, 21)
        errs.append(est - exact)
        bars.append(1.96 * se)
        print(f"   dt {dt}: Monte Carlo - exact = {est - exact:+.4f} +/- {1.96 * se:.4f}   "
              f"site share {site:.4f}   ({time.time() - t:.0f} s)")
    print("   every interval covering 0 means any step-size bias is smaller than the noise")



if __name__ == "__main__":
    main()
