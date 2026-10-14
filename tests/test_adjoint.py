"""The adjoint gradient of the soft-max edge objective matches finite differences."""
import numpy as np
import pytest

from cdd import Scenario, ring_sensors, solve
from cdd.adjoint import LayoutObjective, grad_xy_to_polar, polar_to_xy


def _scenario(pe, sensors):
    return Scenario(r0=2.0, R=6.5, v=1.0, sigma=np.sqrt(2.0 / pe), lam0=1.0, rho=1.0,
                    sensors=sensors, kernel="gaussian", outer="reflect")


@pytest.fixture(scope="module")
def layout():
    rng = np.random.default_rng(0)
    return ring_sensors(6, 3.0) + rng.normal(0.0, 0.4, (6, 2))


def test_true_max_matches_solver(layout):
    scn = _scenario(2.0, layout)
    obj = LayoutObjective(scn, nr=100, ntheta=64)
    assert obj.worst(layout) == pytest.approx(solve(scn, nr=100, ntheta=64).worst_entry()[0],
                                              abs=1e-12)


def test_soft_max_brackets_true_max(layout):
    obj = LayoutObjective(_scenario(2.0, layout), nr=100, ntheta=64)
    for beta in (10.0, 100.0, 1000.0):
        J, _, m = obj.value_and_grad(layout, beta)
        assert m - np.log(64) / beta - 1e-12 <= J <= m + 1e-12


@pytest.mark.parametrize("pe", [0.5, 20.0])
def test_gradient_matches_finite_differences(layout, pe):
    obj = LayoutObjective(_scenario(pe, layout), nr=100, ntheta=64)
    beta, h = 200.0, 1e-5
    _, grad, _ = obj.value_and_grad(layout, beta)
    fd = np.zeros_like(grad)
    for k in range(layout.shape[0]):
        for c in range(2):
            sp, sm = layout.copy(), layout.copy()
            sp[k, c] += h
            sm[k, c] -= h
            fd[k, c] = (obj.value_and_grad(sp, beta)[0] - obj.value_and_grad(sm, beta)[0]) / (2 * h)
    assert np.abs(grad - fd).max() < 1e-6 * max(np.abs(fd).max(), 1.0)


def test_polar_chain_rule(layout):
    obj = LayoutObjective(_scenario(2.0, layout), nr=100, ntheta=64)
    radii, angles = np.hypot(*layout.T), np.arctan2(layout[:, 1], layout[:, 0])
    _, g_xy, _ = obj.value_and_grad(polar_to_xy(radii, angles), 100.0)
    d_r, d_a = grad_xy_to_polar(g_xy, radii, angles)
    h = 1e-5
    r_p, r_m = radii.copy(), radii.copy()
    r_p[2] += h
    r_m[2] -= h
    a_p, a_m = angles.copy(), angles.copy()
    a_p[4] += h
    a_m[4] -= h
    f = lambda r, a: obj.value_and_grad(polar_to_xy(r, a), 100.0)[0]  # noqa: E731
    assert d_r[2] == pytest.approx((f(r_p, angles) - f(r_m, angles)) / (2 * h), abs=1e-7)
    assert d_a[4] == pytest.approx((f(radii, a_p) - f(radii, a_m)) / (2 * h), abs=1e-7)
