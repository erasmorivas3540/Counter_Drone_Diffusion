"""Rung 5: four Gaussian sensors on a ring, no exact answer; PDE vs Monte Carlo.

Site r0 = 0.5, area R = 5 (reflecting), sensors at radius 2 (rho = 1, lam0 = 1),
v = 1, sigma = 1. Drones start at r = 4.5 heading in over a sensor (angle 0),
between sensors (pi/4) and halfway between those (pi/8).
"""
import numpy as np
import pytest
from conftest import Z_MAX, on_circle

from cdd import simulate_u, solve


@pytest.fixture(scope="module")
def pde(ring_scenario):
    return solve(ring_scenario, nr=200, ntheta=256)


@pytest.mark.slow
@pytest.mark.parametrize("angle, seed", [(0.0, 0), (np.pi / 8, 1), (np.pi / 4, 2)])
def test_pde_matches_simulator(ring_scenario, pde, angle, seed):
    x = on_circle(4.5, angle)
    res = simulate_u(x, ring_scenario, dt=1e-2, n_paths=50_000, seed=seed)
    assert res.timed_out < 1e-3
    assert abs(res.z_score(float(pde(x)[0]))) < Z_MAX, (res, float(pde(x)[0]))
