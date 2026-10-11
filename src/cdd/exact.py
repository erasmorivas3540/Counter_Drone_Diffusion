"""Exact and high-accuracy reference answers for checking the solver and simulator.

Case 1  1D straight approach, constant detection rate kappa (closed form).
Case 2  2D, no drift, constant rate, annulus r0 < r < R (modified Bessel functions).
Case 3  2D, radial drift, constant rate (radial ODE solved with solve_bvp).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import i0, i1, k0, k1


def u_1d(x, v: float, sigma: float, kappa: float):
    """Case 1: chance of reaching the site (x = 0) undetected from distance x.

    dX = -v dt + sigma dW, detection at constant rate kappa. The arrival time is
    inverse-Gaussian, so u(x) = exp(m x) with m = (v - sqrt(v^2 + 2 sigma^2 kappa)) / sigma^2.
    sigma = 0 gives the straight-line answer exp(-kappa x / v).
    """
    return np.exp(_decay_rate(v, sigma, kappa) * np.asarray(x, dtype=float))


def _decay_rate(v: float, sigma: float, kappa: float) -> float:
    """m = (v - sqrt(v^2 + 2 sigma^2 kappa)) / sigma^2, written without cancellation.

    Equal to -2 kappa / (v + sqrt(v^2 + 2 sigma^2 kappa)), which also covers sigma = 0
    (m = -kappa / v) and v = 0 (m = -sqrt(2 kappa) / sigma).
    """
    return -2.0 * kappa / (v + np.sqrt(v**2 + 2.0 * sigma**2 * kappa))


def u_bessel(r, r0: float, R: float, sigma: float, kappa: float, outer: str = "reflect"):
    """Case 2: no drift, constant rate kappa, site disk r0, area edge R.

    D (u'' + u'/r) - kappa u = 0 with u(r0) = 1 and u'(R) = 0 (reflect) or u(R) = 0
    (absorb). With a = sqrt(kappa / D), u = [K0(a r) + C I0(a r)] / [K0(a r0) + C I0(a r0)].
    R = inf gives the unbounded-plane answer K0(a r) / K0(a r0).
    """
    r = np.asarray(r, dtype=float)
    a = np.sqrt(2.0 * kappa) / sigma
    if np.isinf(R):
        c = 0.0
    elif outer == "reflect":
        c = k1(a * R) / i1(a * R)
    elif outer == "absorb":
        c = -k0(a * R) / i0(a * R)
    else:
        raise ValueError("outer must be 'reflect' or 'absorb'")
    return (k0(a * r) + c * i0(a * r)) / (k0(a * r0) + c * i0(a * r0))


def u_radial(r, r0: float, R: float, v: float, sigma: float, kappa, outer: str = "reflect",
             tol: float = 1e-9):
    """Case 3: radial drift -v toward the site and a radial detection rate kappa(r).

    Solves D (u'' + u'/r) - v u' - kappa(r) u = 0 on [r0, R] with u(r0) = 1 and
    u'(R) = 0 (reflect) or u(R) = 0 (absorb). kappa may be a number or a function of r.
    Returns u evaluated at r.
    """
    D = 0.5 * sigma**2
    kap = (lambda s: np.full_like(s, float(kappa))) if np.isscalar(kappa) else kappa

    def rhs(s, y):
        u, du = y
        return np.vstack([du, (v * du + kap(s) * u) / D - du / s])

    def bc(ya, yb):
        return np.array([ya[0] - 1.0, yb[1] if outer == "reflect" else yb[0]])

    mesh = np.linspace(r0, R, 400)
    m = _decay_rate(v, sigma, float(np.mean(kap(mesh))))
    guess = np.vstack([np.exp(m * (mesh - r0)), m * np.exp(m * (mesh - r0))])
    sol = solve_bvp(rhs, bc, mesh, guess, tol=tol, max_nodes=200_000)
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")
    return sol.sol(np.asarray(r, dtype=float))[0]
