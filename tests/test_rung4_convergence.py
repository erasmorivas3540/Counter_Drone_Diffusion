"""Rung 4: the PDE error shrinks at the rate the scheme promises.

Central and hybrid differences should be second order (error / 4 per grid doubling),
upwind first order (error / 2). The sensor-ring case has no exact answer, so there the
check is self-convergence: successive differences between grids should shrink ~4x.
"""
import numpy as np
import pytest
from conftest import on_circle

from cdd import Scenario, solve, u_radial

DRIFT = Scenario(r0=0.5, R=4.0, v=1.0, sigma=1.0, lam0=0.5, kernel="uniform", outer="reflect")
GRIDS = (50, 100, 200, 400)


def observed_orders(errors) -> np.ndarray:
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


@pytest.mark.parametrize("scheme, lo, hi", [("central", 1.8, 2.2), ("hybrid", 1.8, 2.2),
                                            ("upwind", 0.8, 1.2)])
def test_radial_drift_convergence_order(scheme, lo, hi):
    errors = []
    for nr in GRIDS:
        sol = solve(DRIFT, nr=nr, ntheta=8, scheme=scheme)
        ref = u_radial(sol.r, 0.5, 4.0, 1.0, 1.0, 0.5, "reflect")
        errors.append(np.abs(sol.u[:, 0] - ref).max())
    orders = observed_orders(errors)
    assert np.all((orders > lo) & (orders < hi)), (errors, orders)


def test_sensor_ring_self_convergence(ring_scenario):
    points = np.array([on_circle(4.5, a) for a in (0.0, np.pi / 8, np.pi / 4)])
    values = [solve(ring_scenario, nr=nr, ntheta=nth)(points)
              for nr, nth in ((100, 128), (200, 256), (400, 512))]
    d1 = np.abs(values[0] - values[1])
    d2 = np.abs(values[1] - values[2])
    orders = np.log2(d1 / d2)
    assert np.all((orders > 1.5) & (orders < 2.6)), orders
    assert d2.max() < 1e-4
