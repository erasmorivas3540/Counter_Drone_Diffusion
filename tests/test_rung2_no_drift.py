"""Rung 2: 2D, no drift, constant detection rate, ring r0 = 0.5 < r < R = 4 (Bessel).

sigma = 1, kappa = 1. Exact u(1.5) = 0.1505 (reflecting edge), 0.1502 (absorbing edge).
"""
import numpy as np
import pytest
from conftest import Z_MAX

from cdd import Scenario, simulate_u, solve, u_bessel

BASE = dict(r0=0.5, R=4.0, v=0.0, sigma=1.0, lam0=1.0, kernel="uniform")


@pytest.mark.parametrize("outer", ["reflect", "absorb"])
def test_pde_matches_bessel(outer):
    sol = solve(Scenario(**BASE, outer=outer), nr=200, ntheta=16)
    exact = u_bessel(sol.r, 0.5, 4.0, 1.0, 1.0, outer)
    assert np.abs(sol.u - exact[:, None]).max() < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("outer, seed", [("reflect", 5), ("absorb", 6)])
def test_simulator_matches_bessel(outer, seed):
    exact = float(u_bessel(1.5, 0.5, 4.0, 1.0, 1.0, outer))
    # Without drift, undetected paths can wander for a long time; dropping paths whose
    # survival weight is below 1e-6 shifts the estimate by at most 1e-6.
    res = simulate_u([0.0, 1.5], Scenario(**BASE, outer=outer), dt=1e-2, n_paths=50_000,
                     seed=seed, w_min=1e-6)
    assert abs(res.z_score(exact)) < Z_MAX, (res, exact)
