"""Rung 1: straight 1D approach at constant detection rate (closed form).

dX = -v dt + sigma dW from x = 1, site at 0, v = 1, kappa = 1.
Exact answers: 0.4070 (sigma 0.5), 0.4809 (sigma 1), 0.6065 (sigma 2).
"""
import pytest
from conftest import Z_MAX

from cdd import simulate_u_1d, u_1d


@pytest.mark.parametrize("sigma, seed", [(0.5, 1), (1.0, 2), (2.0, 3)])
def test_simulator_matches_closed_form(sigma, seed):
    exact = float(u_1d(1.0, 1.0, sigma, 1.0))
    res = simulate_u_1d(1.0, 1.0, sigma, 1.0, dt=1e-2, n_paths=200_000, seed=seed)
    assert abs(res.z_score(exact)) < Z_MAX, (res, exact)


def test_bridge_check_removes_step_bias():
    """At a coarse step a plain simulator misses arrivals between steps and is biased
    low by about 0.05; the bridge check and half-step refund remove that."""
    exact = float(u_1d(1.0, 1.0, 1.0, 1.0))
    plain = simulate_u_1d(1.0, 1.0, 1.0, 1.0, dt=0.1, n_paths=100_000, seed=4, bridge=False)
    fixed = simulate_u_1d(1.0, 1.0, 1.0, 1.0, dt=0.1, n_paths=100_000, seed=4, bridge=True)
    assert exact - plain.estimate > 0.03
    assert plain.z_score(exact) > 10
    assert abs(fixed.z_score(exact)) < Z_MAX, (fixed, exact)
