"""Finite-difference solution of the Feynman-Kac equation on a polar grid.

    D Δu - v ∂u/∂r - κ(x) u = 0   on r0 < r < R,
    u = 1 at r = r0,  ∂u/∂r = 0 (reflect) or u = 0 (absorb) at r = R.

`build_system` assembles the sparse linear system A u = b without solving it. The sensors
enter only through -κ on the diagonal of A, which is what the adjoint gradient in
`cdd.adjoint` relies on. `solve` assembles and solves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from .model import Scenario

SCHEMES = ("central", "upwind", "hybrid")


@dataclass
class PDESolution:
    r: np.ndarray
    theta: np.ndarray
    u: np.ndarray
    scheme: str

    def __call__(self, x):
        """u at points x (bilinear in r and θ); 1 inside the site."""
        x = np.asarray(x, float).reshape(-1, 2)
        r0, R = self.r[0], self.r[-1]
        hr = self.r[1] - self.r[0]
        nth = self.theta.size
        ht = 2 * np.pi / nth
        rad = np.hypot(x[:, 0], x[:, 1])
        ang = np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
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

    def worst_entry(self):
        """The drone's best entry point on the outer edge: (u, angle)."""
        j = int(np.argmax(self.u[-1]))
        return float(self.u[-1, j]), float(self.theta[j])


@dataclass
class PDESystem:
    """The assembled system A u = rhs for the unknown rings (sensor-free part kept apart).

    A = A0 - diag(kappa). `points` are the (x, y) positions of the unknowns, in the same
    order as the unknown vector (ring-major, then angle).
    """
    r: np.ndarray
    theta: np.ndarray
    A0: csc_matrix
    rhs: np.ndarray
    points: np.ndarray
    reflect: bool
    scheme: str

    @property
    def nring(self) -> int:
        return self.points.shape[0] // self.theta.size

    def matrix(self, kappa: np.ndarray) -> csc_matrix:
        from scipy.sparse import diags
        return (self.A0 - diags(kappa)).tocsc()

    def to_grid(self, sol: np.ndarray) -> np.ndarray:
        nth = self.theta.size
        u = np.empty((self.r.size, nth))
        u[0] = 1.0
        if self.reflect:
            u[1:] = sol.reshape(-1, nth)
        else:
            u[1:-1] = sol.reshape(-1, nth)
            u[-1] = 0.0
        return u


def build_system(scn: Scenario, nr: int = 200, ntheta: int = 256,
                 scheme: str = "hybrid") -> PDESystem:
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if nr < 2 or ntheta < 4:
        raise ValueError("need nr >= 2 and ntheta >= 4")
    r = np.linspace(scn.r0, scn.R, nr + 1)
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    hr, ht = r[1] - r[0], 2 * np.pi / ntheta
    reflect = scn.outer == "reflect"
    rings = r[1:] if reflect else r[1:-1]
    nring = rings.size
    rr, tt = np.meshgrid(rings, theta, indexing="ij")
    pts = np.column_stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()])

    br = np.full((nring, ntheta), -scn.v)
    D = scn.D
    dif_p = D * (rr + hr / 2) / (rr * hr**2)
    dif_m = D * (rr - hr / 2) / (rr * hr**2)
    dif_t = D / (rr**2 * ht**2)
    if scheme == "central" and D > 0:
        use_central = np.ones_like(br, bool)
    elif scheme == "upwind" or D == 0:
        use_central = np.zeros_like(br, bool)
    else:
        use_central = np.abs(br) * hr <= 2.0 * D
    adv_p = np.where(use_central, br / (2 * hr), np.maximum(br, 0.0) / hr)
    adv_m = np.where(use_central, -br / (2 * hr), np.maximum(-br, 0.0) / hr)
    P = dif_p + adv_p
    M = dif_m + adv_m
    T = dif_t
    diag = -(P + M + 2 * T)

    a_idx, j_idx = np.meshgrid(np.arange(nring), np.arange(ntheta), indexing="ij")
    row = (a_idx * ntheta + j_idx).ravel()
    rows, cols, vals = [row], [row], [diag.ravel()]
    for dj in (1, -1):
        rows.append(row)
        cols.append((a_idx * ntheta + (j_idx + dj) % ntheta).ravel())
        vals.append(T.ravel())
    rhs = np.zeros(nring * ntheta)
    inner = a_idx > 0
    rows.append(row[inner.ravel()])
    cols.append(((a_idx - 1) * ntheta + j_idx)[inner])
    vals.append(M[inner])
    rhs[row[(~inner).ravel()]] -= M[~inner] * 1.0
    outer = a_idx < nring - 1
    rows.append(row[outer.ravel()])
    cols.append(((a_idx + 1) * ntheta + j_idx)[outer])
    vals.append(P[outer])
    if reflect:
        last = ~outer
        rows.append(row[last.ravel()])
        cols.append(((a_idx - 1) * ntheta + j_idx)[last])
        vals.append(P[last])
    n = nring * ntheta
    A0 = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                    shape=(n, n)).tocsc()
    return PDESystem(r=r, theta=theta, A0=A0, rhs=rhs, points=pts, reflect=reflect,
                     scheme=scheme)


def solve(scn: Scenario, nr: int = 200, ntheta: int = 256, scheme: str = "hybrid") -> PDESolution:
    """Solve for u on an (nr + 1) x ntheta polar grid."""
    system = build_system(scn, nr=nr, ntheta=ntheta, scheme=scheme)
    kappa = scn.detection_rate(system.points)
    sol = spsolve(system.matrix(kappa), system.rhs)
    return PDESolution(r=system.r, theta=system.theta, u=system.to_grid(sol), scheme=scheme)
