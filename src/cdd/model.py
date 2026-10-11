"""Geometry, drone motion and sensor detection for the counter-drone diffusion model.

Units are dimensionless by default: lengths in sensor ranges (rho = 1) and time in
range-crossings at full speed (v = 1). Any consistent units work.

Drone motion (Brownian motion with drift toward the site at the origin):
    dX = -v X/|X| dt + sigma dW,      D = sigma^2 / 2
Detection: a Poisson process with position-dependent rate kappa(X_t).
The site is the disk |x| <= r0; the area edge is the circle |x| = R.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KERNELS = ("gaussian", "disk", "uniform")
OUTER_EDGES = ("reflect", "absorb")


def ring_sensors(k: int, radius: float, phase: float = 0.0) -> np.ndarray:
    """K sensors evenly spaced on a ring; returns an array of shape (k, 2)."""
    angles = phase + 2.0 * np.pi * np.arange(k) / k
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


@dataclass
class Scenario:
    """One defended site, one drone model and one sensor layout.

    kernel:
        "gaussian": each sensor detects at lam0 * exp(-d^2 / (2 rho^2))
        "disk":     each sensor detects at lam0 within distance rho, 0 outside
        "uniform":  detection rate lam0 everywhere (sensors ignored); exact cases
    outer:
        "reflect":  the drone cannot leave the area (u_r = 0 at R)
        "absorb":   leaving the area means the drone gives up (u = 0 at R)
    """

    r0: float = 0.5
    R: float = 10.0
    v: float = 1.0
    sigma: float = 1.0
    lam0: float = 1.0
    rho: float = 1.0
    sensors: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    kernel: str = "gaussian"
    outer: str = "reflect"

    def __post_init__(self) -> None:
        self.sensors = np.asarray(self.sensors, dtype=float).reshape(-1, 2)
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        if self.outer not in OUTER_EDGES:
            raise ValueError(f"outer must be one of {OUTER_EDGES}")
        if not 0.0 < self.r0 < self.R:
            raise ValueError("need 0 < r0 < R")
        if self.sigma < 0 or self.v < 0 or self.lam0 < 0 or self.rho <= 0:
            raise ValueError("sigma, v, lam0 must be >= 0 and rho > 0")

    @property
    def D(self) -> float:
        """Diffusion coefficient sigma^2 / 2."""
        return 0.5 * self.sigma**2

    @property
    def peclet(self) -> float:
        """Pe = v rho / D: how directed the approach is (inf if sigma = 0)."""
        return np.inf if self.D == 0 else self.v * self.rho / self.D

    @property
    def detection_number(self) -> float:
        """Da = lam0 rho / v: expected detections per range crossed at full speed."""
        return np.inf if self.v == 0 else self.lam0 * self.rho / self.v

    def drift(self, x: np.ndarray) -> np.ndarray:
        """Drift -v x/|x| for points x of shape (n, 2)."""
        x = np.asarray(x, dtype=float)
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        return -self.v * x / np.maximum(r, 1e-300)

    def detection_rate(self, x: np.ndarray) -> np.ndarray:
        """Total detection rate kappa(x) for points x of shape (n, 2); returns (n,)."""
        x = np.asarray(x, dtype=float).reshape(-1, 2)
        if self.kernel == "uniform":
            return np.full(x.shape[0], self.lam0)
        if self.sensors.shape[0] == 0:
            return np.zeros(x.shape[0])
        s = self.sensors
        d2 = (x**2).sum(axis=1)[:, None] - 2.0 * (x @ s.T) + (s**2).sum(axis=1)[None, :]
        d2 = np.maximum(d2, 0.0)                     # squared distances, shape (n, k)
        if self.kernel == "gaussian":
            return self.lam0 * np.exp(-d2 / (2.0 * self.rho**2)).sum(axis=1)
        return self.lam0 * (d2 <= self.rho**2).sum(axis=1).astype(float)
