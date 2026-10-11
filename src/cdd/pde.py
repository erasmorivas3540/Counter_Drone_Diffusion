"""Polar-grid finite-difference solver for u(x), the chance of arriving undetected.

By the Feynman-Kac formula, u solves on the ring r0 < r < R

    D (u_rr + u_r / r + u_tt / r^2) + b_r u_r - kappa u = 0,

with b_r = -v (drift toward the site), u = 1 on the site edge r = r0, and either
u_r = 0 (reflecting edge) or u = 0 (absorbing edge) at r = R. Angles are periodic.

Discretization (uniform grid in r and theta):
  * radial diffusion in conservative form, (1/r) d/dr (r du/dr), second order;
  * the drift term by central differences (second order) or one-sided "upwind"
    differences taken in the direction of the drift (first order, but always
    monotone); "hybrid" uses central wherever the cell Peclet number
    |b_r| h_r / (2 D) is at most 1 and upwind elsewhere;
  * the reflecting edge by a mirrored ghost ring, u_{N+1} = u_{N-1}.
Every off-diagonal coefficient is nonnegative, so the discrete u stays in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .model import Scenario

SCHEMES = ("central", "upwind", "hybrid")


@dataclass
class PDESolution:
    r: np.ndarray        # (nr + 1,) grid radii, r[0] = r0, r[-1] = R
    theta: np.ndarray    # (ntheta,) grid angles in [0, 2 pi)
    u: np.ndarray        # (nr + 1, ntheta) values; u[0] = 1 on the site edge
    scheme: str

    def __call__(self, x) -> np.ndarray:
        """Bilinear interpolation of u at points x of shape (n, 2) (or one point)."""
        x = np.asarray(x, dtype=float).reshape(-1, 2)
        r0, R = self.r[0], self.r[-1]
        hr = self.r[1] - self.r[0]
        nth = self.theta.size
        ht = 2.0 * np.pi / nth
        rad = np.hypot(x[:, 0], x[:, 1])
        ang = np.mod(np.arctan2(x[:, 1], x[:, 0]), 2.0 * np.pi)
        fi = (np.clip(rad, r0, R) - r0) / hr
        i0 = np.clip(np.floor(fi).astype(int), 0, self.r.size - 2)
        wr = fi - i0
        fj = ang / ht
        j0 = np.floor(fj).astype(int) % nth
        j1 = (j0 + 1) % nth
        wt = fj - np.floor(fj)
        u = self.u
        val = ((1 - wr) * (1 - wt) * u[i0, j0] + (1 - wr) * wt * u[i0, j1]
               + wr * (1 - wt) * u[i0 + 1, j0] + wr * wt * u[i0 + 1, j1])
        return np.where(rad <= r0, 1.0, val)

    def worst_entry(self) -> tuple[float, float]:
        """Largest u on the outer edge and the angle where it occurs."""
        j = int(np.argmax(self.u[-1]))
        return float(self.u[-1, j]), float(self.theta[j])


def solve(scn: Scenario, nr: int = 200, ntheta: int = 256, scheme: str = "hybrid") -> PDESolution:
    """Solve for u on an (nr + 1) x ntheta polar grid."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if nr < 2 or ntheta < 4:
        raise ValueError("need nr >= 2 and ntheta >= 4")
    r = np.linspace(scn.r0, scn.R, nr + 1)
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    hr, ht = r[1] - r[0], 2.0 * np.pi / ntheta
    reflect = scn.outer == "reflect"
    rings = r[1:] if reflect else r[1:-1]          # radii of the unknown rings
    nring = rings.size

    rr, tt = np.meshgrid(rings, theta, indexing="ij")          # (nring, ntheta)
    pts = np.column_stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()])
    kappa = scn.detection_rate(pts).reshape(nring, ntheta)
    br = np.full((nring, ntheta), -scn.v)                     # radial drift component
    D = scn.D

    dif_p = D * (rr + hr / 2) / (rr * hr**2)                 # weight on u_{i+1}
    dif_m = D * (rr - hr / 2) / (rr * hr**2)                 # weight on u_{i-1}
    dif_t = D / (rr**2 * ht**2)                              # weight on u_{j+1}, u_{j-1}
    if scheme == "central" and D > 0:
        use_central = np.ones_like(br, dtype=bool)
    elif scheme == "upwind" or D == 0:
        use_central = np.zeros_like(br, dtype=bool)
    else:
        use_central = np.abs(br) * hr <= 2.0 * D
    adv_p = np.where(use_central, br / (2 * hr), np.maximum(br, 0.0) / hr)
    adv_m = np.where(use_central, -br / (2 * hr), np.maximum(-br, 0.0) / hr)
    P = dif_p + adv_p
    M = dif_m + adv_m
    T = dif_t
    diag = -(P + M + 2 * T) - kappa

    a_idx, j_idx = np.meshgrid(np.arange(nring), np.arange(ntheta), indexing="ij")
    row = (a_idx * ntheta + j_idx).ravel()
    rows, cols, vals = [row], [row], [diag.ravel()]
    for dj in (1, -1):                                       # periodic angle neighbours
        rows.append(row)
        cols.append((a_idx * ntheta + (j_idx + dj) % ntheta).ravel())
        vals.append(T.ravel())
    rhs = np.zeros(nring * ntheta)

    inner = a_idx > 0                                        # u_{i-1} is an unknown
    rows.append(row[inner.ravel()])
    cols.append(((a_idx - 1) * ntheta + j_idx)[inner])
    vals.append(M[inner])
    rhs[row[(~inner).ravel()]] -= M[~inner] * 1.0            # u = 1 on the site edge

    outer = a_idx < nring - 1                                # u_{i+1} is an unknown
    rows.append(row[outer.ravel()])
    cols.append(((a_idx + 1) * ntheta + j_idx)[outer])
    vals.append(P[outer])
    if reflect:                                              # ghost ring: u_{N+1} = u_{N-1}
        last = ~outer
        rows.append(row[last.ravel()])
        cols.append(((a_idx - 1) * ntheta + j_idx)[last])
        vals.append(P[last])
    # absorbing edge: u_{N} = 0 contributes nothing

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(nring * ntheta, nring * ntheta)).tocsc()
    sol = spsolve(A, rhs).reshape(nring, ntheta)

    u = np.empty((nr + 1, ntheta))
    u[0] = 1.0
    if reflect:
        u[1:] = sol
    else:
        u[1:-1] = sol
        u[-1] = 0.0
    return PDESolution(r=r, theta=theta, u=u, scheme=scheme)
