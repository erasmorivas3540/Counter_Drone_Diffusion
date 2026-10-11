"""Rung 3: 2D, drift toward the site, constant detection rate (radial ODE via solve_bvp).

r0 = 0.5, R = 4, v = 1, sigma = 1, kappa = 0.5. Reference u(2.5) = 0.3495 (reflecting).
"""
import numpy as np
import pytest
from conftest import Z_MAX, on_circle

from cdd import Scenario, simulate_u, solve, u_radial

BASE = dict(r0=0.5, R=4.0, v=1.0, sigma=1.0, lam0=0.5, kernel="uniform")


@pytest.mark.parametrize("outer", ["reflect", "absorb"])
@pytest.mark.parametrize("scheme, tol", [("central", 1e-4), ("hybrid", 1e-4), ("upwind", 5e-3)])
def test_pde_matches_radial_ode(outer, scheme, tol):
    sol = solve(Scenario(**BASE, outer=outer), nr=200, ntheta=16, scheme=scheme)
    ref = u_radial(sol.r, 0.5, 4.0, 1.0, 1.0, 0.5, outer)
    assert np.abs(sol.u - ref[:, None]).max() < tol


@pytest.mark.slow
@pytest.mark.parametrize("outer, seed", [("reflect", 7), ("absorb", 8)])
def test_simulator_matches_radial_ode(outer, seed):
    ref = float(u_radial(2.5, 0.5, 4.0, 1.0, 1.0, 0.5, outer))
    res = simulate_u(on_circle(2.5, 1.0), Scenario(**BASE, outer=outer), dt=1e-2,
                     n_paths=50_000, seed=seed)
    assert res.timed_out < 1e-3
    assert abs(res.z_score(ref)) < Z_MAX, (res, ref)
