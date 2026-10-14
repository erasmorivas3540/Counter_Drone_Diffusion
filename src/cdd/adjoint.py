"""Gradient of the drone's best chance with respect to sensor positions (adjoint method).

The defender wants to minimize the worst-case u over entry points on the outer edge,
max_j u(R, θ_j). A maximum is not smooth, so it is replaced by a soft maximum

    J = (1/β) log( (1/n) Σ_j exp(β u_j) ),   max_j u_j - log(n)/β <= J <= max_j u_j,

which approaches the true maximum as β grows.

The discrete system is A(s) u = b with A = A0 - diag(κ(s)): the sensors s enter only
through the detection rate on the diagonal, and b does not depend on them. So

    dJ/ds = λᵀ (∂b/∂s - ∂A/∂s u) = Σ_i λ_i u_i ∂κ_i/∂s,   where Aᵀ λ = ∂J/∂u.

One extra (transposed) solve with the same LU factors gives the gradient for every
sensor coordinate at once. For the Gaussian footprint
∂κ_i/∂s_j = λ0 exp(-|x_i - s_j|²/2ρ²) (x_i - s_j)/ρ².

The gradient is exact for the discrete problem, so it should match finite differences
of the same grid to rounding (tests/test_adjoint.py checks this).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import splu

from .model import Scenario
from .pde import build_system


class LayoutObjective:
    """Soft-max of edge u and its gradient for a fixed scenario with movable sensors."""

    def __init__(self, scn: Scenario, nr: int = 200, ntheta: int = 128,
                 scheme: str = "hybrid"):
        if scn.kernel != "gaussian":
            raise ValueError("the adjoint gradient is implemented for the gaussian kernel")
        if scn.outer != "reflect":
            raise ValueError("the edge objective needs a reflecting edge (u = 0 when absorbing)")
        self.scn = scn
        self.system = build_system(scn, nr=nr, ntheta=ntheta, scheme=scheme)
        self.ntheta = ntheta
        self.n_unknown = self.system.points.shape[0]

    def _kappa(self, sensors):
        x = self.system.points
        diff = x[:, None, :] - sensors[None, :, :]                 # (N, K, 2)
        g = np.exp(-(diff**2).sum(axis=2) / (2.0 * self.scn.rho**2))  # (N, K)
        return self.scn.lam0 * g.sum(axis=1), g, diff

    def solve(self, sensors):
        """Edge u values (ntheta,), full unknown vector, LU factors and pieces for the gradient."""
        sensors = np.asarray(sensors, float).reshape(-1, 2)
        kappa, g, diff = self._kappa(sensors)
        lu = splu(self.system.matrix(kappa))
        u = lu.solve(self.system.rhs)
        return u[-self.ntheta:], u, lu, g, diff

    def worst(self, sensors) -> float:
        """The true maximum of u over the edge (the quantity being targeted)."""
        return float(self.solve(sensors)[0].max())

    def value_and_grad(self, sensors, beta: float):
        """Soft-max J, dJ/d(sensor x, y) as a (K, 2) array, and the true max."""
        edge, u, lu, g, diff = self.solve(sensors)
        m = edge.max()
        e = np.exp(beta * (edge - m))
        J = m + np.log(e.mean()) / beta
        dJ_du = np.zeros(self.n_unknown)
        dJ_du[-self.ntheta:] = e / e.sum()
        lam = lu.solve(dJ_du, trans="T")
        w = lam * u                                                     # (N,)
        grad = self.scn.lam0 / self.scn.rho**2 * np.einsum("i,ik,ikc->kc", w, g, diff)
        return float(J), grad, float(m)


def polar_to_xy(radii, angles):
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def grad_xy_to_polar(grad_xy, radii, angles):
    """Chain rule from (x, y) gradients to (radius, angle) gradients."""
    c, s = np.cos(angles), np.sin(angles)
    d_r = grad_xy[:, 0] * c + grad_xy[:, 1] * s
    d_a = radii * (-grad_xy[:, 0] * s + grad_xy[:, 1] * c)
    return d_r, d_a
