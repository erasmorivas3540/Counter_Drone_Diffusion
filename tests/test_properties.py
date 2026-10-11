"""Structural checks: probabilities stay in [0, 1], symmetries hold, the no-noise limit
is exact, and the model's inputs are validated."""
from dataclasses import replace

import numpy as np
import pytest
from conftest import on_circle
from scipy.integrate import quad

from cdd import Scenario, ring_sensors, simulate_u, solve


def straight_line_u(scn: Scenario, r: float, angle: float) -> float:
    """sigma = 0: the drone flies straight in, so u = exp(-(1/v) * integral of kappa)."""
    rate = lambda s: float(scn.detection_rate(on_circle(s, angle))[0])
    integral = quad(rate, scn.r0, r, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    return float(np.exp(-integral / scn.v))


# --- the PDE solution behaves like a probability --------------------------------------

def test_pde_values_are_probabilities(ring_scenario):
    sol = solve(ring_scenario, nr=100, ntheta=64)
    assert np.all(sol.u >= 0.0) and np.all(sol.u <= 1.0)
    assert np.all(sol.u[0] == 1.0)


def test_more_detection_lowers_u(ring_scenario):
    weak = solve(ring_scenario, nr=100, ntheta=64)
    strong = solve(replace(ring_scenario, lam0=2.0), nr=100, ntheta=64)
    assert np.all(strong.u <= weak.u + 1e-12)
    assert (weak.u - strong.u).max() > 0.05


def test_ring_symmetry(ring_scenario):
    """Four sensors at angles 0, 90, 180, 270 degrees: u repeats every quarter turn and
    is mirror-symmetric about angle 0."""
    nth = 64
    u = solve(ring_scenario, nr=100, ntheta=nth).u
    assert np.abs(u - np.roll(u, nth // 4, axis=1)).max() < 1e-10
    assert np.abs(u - u[:, (-np.arange(nth)) % nth]).max() < 1e-10


def test_rotating_the_sensors_rotates_the_solution(ring_scenario):
    nth, shift = 64, 5
    base = solve(ring_scenario, nr=100, ntheta=nth).u
    turned = replace(ring_scenario, sensors=ring_sensors(4, 2.0, phase=2 * np.pi * shift / nth))
    assert np.abs(solve(turned, nr=100, ntheta=nth).u - np.roll(base, shift, axis=1)).max() < 1e-10


def test_between_sensors_is_the_better_approach(ring_scenario):
    sol = solve(ring_scenario, nr=200, ntheta=256)
    over, between = sol(np.array([on_circle(4.5, 0.0), on_circle(4.5, np.pi / 4)]))
    assert between > over + 0.02


def test_interpolation_returns_grid_values_and_one_inside_site(ring_scenario):
    sol = solve(ring_scenario, nr=50, ntheta=32)
    i, j = 17, 9
    assert sol(on_circle(sol.r[i], sol.theta[j]))[0] == pytest.approx(sol.u[i, j], abs=1e-12)
    assert sol(np.array([0.1, -0.2]))[0] == 1.0


# --- sigma = 0: straight-line flight has an exact answer, sensors and all ------------

@pytest.fixture(scope="module")
def no_noise(ring_scenario):
    return replace(ring_scenario, sigma=0.0)


def test_straight_line_limit_pde(no_noise):
    """With no noise the solver falls back to upwind differences: first order."""
    angles = (0.0, np.pi / 8, np.pi / 4)
    exact = np.array([straight_line_u(no_noise, 4.5, a) for a in angles])
    pts = np.array([on_circle(4.5, a) for a in angles])
    err = [np.abs(solve(no_noise, nr=nr, ntheta=64)(pts) - exact) for nr in (400, 800)]
    assert err[1].max() < 1e-3
    assert np.all(np.abs(err[0] / err[1] - 2.0) < 0.2)


def test_straight_line_limit_simulator(no_noise):
    for angle in (0.0, np.pi / 8, np.pi / 4):
        exact = straight_line_u(no_noise, 4.5, angle)
        res = simulate_u(on_circle(4.5, angle), no_noise, dt=1e-3, n_paths=2)
        assert res.estimate == pytest.approx(exact, abs=2e-4)


def test_simulator_starting_inside_site_returns_one(ring_scenario):
    assert simulate_u([0.1, 0.1], ring_scenario).estimate == 1.0


# --- model inputs ---------------------------------------------------------------------

def test_detection_rate_kernels():
    sensors = np.array([[1.0, 0.0], [-1.0, 0.0]])
    x = np.array([[0.0, 0.0], [1.0, 0.5], [3.0, 3.0]])
    gauss = Scenario(sensors=sensors, lam0=2.0, rho=0.8, kernel="gaussian")
    d2 = ((x[:, None, :] - sensors[None]) ** 2).sum(-1)
    assert np.allclose(gauss.detection_rate(x), 2.0 * np.exp(-d2 / (2 * 0.8**2)).sum(1))
    disk = Scenario(sensors=sensors, lam0=2.0, rho=1.0, kernel="disk")
    assert np.allclose(disk.detection_rate(x), [4.0, 2.0, 0.0])
    assert np.allclose(Scenario(kernel="uniform", lam0=0.3).detection_rate(x), 0.3)
    assert np.allclose(Scenario().detection_rate(x), 0.0)          # no sensors


def test_scenario_checks_inputs():
    with pytest.raises(ValueError):
        Scenario(kernel="cone")
    with pytest.raises(ValueError):
        Scenario(outer="wrap")
    with pytest.raises(ValueError):
        Scenario(r0=2.0, R=1.0)
    with pytest.raises(ValueError):
        Scenario(rho=0.0)


def test_dimensionless_numbers():
    scn = Scenario(v=2.0, sigma=1.0, lam0=3.0, rho=0.5)
    assert scn.D == 0.5
    assert scn.peclet == pytest.approx(2.0 * 0.5 / 0.5)
    assert scn.detection_number == pytest.approx(3.0 * 0.5 / 2.0)
    assert Scenario(sigma=0.0).peclet == np.inf
