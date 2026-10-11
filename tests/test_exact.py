"""Checks on the exact reference answers themselves (fast, no simulation)."""
import numpy as np
import pytest

from cdd import u_1d, u_bessel, u_radial


def test_1d_solves_its_equation():
    """u = exp(m x) satisfies D u'' - v u' - kappa u = 0 with u(0) = 1."""
    v, sigma, kappa = 1.3, 0.8, 0.7
    D = 0.5 * sigma**2
    x = np.linspace(0.2, 3.0, 50)
    h = 1e-4
    u = u_1d(x, v, sigma, kappa)
    du = (u_1d(x + h, v, sigma, kappa) - u_1d(x - h, v, sigma, kappa)) / (2 * h)
    d2u = (u_1d(x + h, v, sigma, kappa) - 2 * u + u_1d(x - h, v, sigma, kappa)) / h**2
    assert np.abs(D * d2u - v * du - kappa * u).max() < 1e-5
    assert u_1d(0.0, v, sigma, kappa) == pytest.approx(1.0)


def test_1d_limits():
    """sigma -> 0 gives the straight-line answer; v = 0 gives exp(-sqrt(2 kappa) x / sigma)."""
    assert u_1d(2.0, 1.0, 0.0, 0.5) == pytest.approx(np.exp(-1.0), rel=1e-14)
    assert u_1d(2.0, 1.0, 1e-6, 0.5) == pytest.approx(np.exp(-1.0), rel=1e-9)
    assert u_1d(2.0, 0.0, 1.0, 0.5) == pytest.approx(np.exp(-2.0), rel=1e-14)


def test_1d_randomness_raises_undetected_arrival():
    """The headline 1D effect: same mean speed, more randomness, better odds for the drone."""
    u = [float(u_1d(1.0, 1.0, s, 1.0)) for s in (0.0, 0.5, 1.0, 2.0)]
    assert np.all(np.diff(u) > 0)
    assert u[0] == pytest.approx(np.exp(-1.0))
    assert u[2] == pytest.approx(0.4809, abs=1e-4)


@pytest.mark.parametrize("outer", ["reflect", "absorb"])
def test_bessel_matches_radial_ode_without_drift(outer):
    """Two independent routes to the same answer: Bessel closed form vs solve_bvp."""
    r = np.linspace(0.5, 4.0, 30)
    exact = u_bessel(r, 0.5, 4.0, 1.0, 1.0, outer)
    ode = u_radial(r, 0.5, 4.0, 0.0, 1.0, 1.0, outer)
    assert np.abs(exact - ode).max() < 1e-7


def test_bessel_boundary_conditions():
    a = np.sqrt(2.0)                       # sigma = 1, kappa = 1
    h = 1e-5
    assert u_bessel(0.5, 0.5, 4.0, 1.0, 1.0, "reflect") == pytest.approx(1.0)
    slope = (u_bessel(4.0, 0.5, 4.0, 1.0, 1.0, "reflect")
             - u_bessel(4.0 - h, 0.5, 4.0, 1.0, 1.0, "reflect")) / h
    assert abs(slope) < 1e-6 * a
    assert abs(u_bessel(4.0, 0.5, 4.0, 1.0, 1.0, "absorb")) < 1e-12


def test_bessel_large_area_approaches_open_plane():
    r = np.array([1.0, 2.0, 3.0])
    far = u_bessel(r, 0.5, 40.0, 1.0, 1.0, "reflect")
    plane = u_bessel(r, 0.5, np.inf, 1.0, 1.0)
    assert np.abs(far - plane).max() < 1e-12


def test_radial_ode_accepts_a_rate_profile():
    """kappa may depend on r; a flat profile must match the constant-rate call."""
    r = np.linspace(0.5, 4.0, 20)
    flat = u_radial(r, 0.5, 4.0, 1.0, 1.0, lambda s: np.full_like(s, 0.5))
    const = u_radial(r, 0.5, 4.0, 1.0, 1.0, 0.5)
    assert np.abs(flat - const).max() < 1e-8
    belt = u_radial(r, 0.5, 4.0, 1.0, 1.0, lambda s: np.exp(-(s - 2.0) ** 2 / 2))
    assert np.all(np.diff(belt) < 0) and belt[0] == pytest.approx(1.0)
