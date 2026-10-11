"""Monte Carlo estimates of u, the chance of reaching the site undetected.

Each path carries a survival weight: the probability that no detection has happened
so far, exp(-integral of kappa). A path stops when it reaches the site.

Two corrections keep large steps accurate:
  * Bridge check: a path that stays outside the site at both ends of a step may
    still have touched it in between. Treating the site edge locally as a straight
    line, that happens with probability exp(-2 d_old d_new / (sigma^2 dt)), where
    d is the distance to the edge (the Brownian-bridge formula).
  * Half-step refund: an arriving path got there partway through its last step,
    so it is charged half of that step's detection hazard rather than all of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Scenario


@dataclass
class MCResult:
    estimate: float
    half_width: float      # 95% confidence half-width
    n_paths: int
    timed_out: float       # share of paths still travelling at t_max (should be ~0)

    @property
    def interval(self) -> tuple[float, float]:
        return self.estimate - self.half_width, self.estimate + self.half_width

    @property
    def std_error(self) -> float:
        return self.half_width / 1.96

    def contains(self, value: float) -> bool:
        lo, hi = self.interval
        return lo <= value <= hi

    def z_score(self, value: float) -> float:
        """(value - estimate) in standard errors; the tests require |z| < 3.5."""
        return (value - self.estimate) / self.std_error if self.std_error > 0 else 0.0


def _summarize(weights: np.ndarray, still_travelling: np.ndarray) -> MCResult:
    n = weights.size
    return MCResult(
        estimate=float(weights.mean()),
        half_width=float(1.96 * weights.std(ddof=1) / np.sqrt(n)),
        n_paths=n,
        timed_out=float(still_travelling.mean()),
    )


def simulate_u_1d(x: float, v: float, sigma: float, kappa: float, dt: float = 1e-2,
                  n_paths: int = 200_000, seed: int = 0, t_max: float | None = None,
                  bridge: bool = True) -> MCResult:
    """1D check: dX = -v dt + sigma dW from X0 = x, constant rate kappa, site at 0."""
    rng = np.random.default_rng(seed)
    t_max = t_max or 20.0 * x / max(v, 1e-12)
    pos = np.full(n_paths, float(x))
    weight = np.ones(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    step_survival = np.exp(-kappa * dt)
    for _ in range(int(np.ceil(t_max / dt))):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        old = pos[idx]
        new = old - v * dt + sigma * np.sqrt(dt) * rng.standard_normal(idx.size)
        weight[idx] *= step_survival
        hit = new <= 0.0
        if bridge and sigma > 0:
            p_cross = np.exp(-2.0 * old * np.maximum(new, 0.0) / (sigma**2 * dt))
            hit |= rng.random(idx.size) < p_cross
        weight[idx[hit]] /= np.sqrt(step_survival)
        alive[idx[hit]] = False
        pos[idx] = new
    weight[alive] = 0.0
    return _summarize(weight, alive)


def simulate_u(x0, scn: Scenario, dt: float = 5e-3, n_paths: int = 50_000, seed: int = 0,
               t_max: float | None = None, bridge: bool = True,
               w_min: float = 1e-12) -> MCResult:
    """2D estimate of u(x0) for a scenario.

    Euler-Maruyama steps; survival weights use the trapezoid rule for the hazard;
    the outer edge reflects by mirroring (or absorbs, per scn.outer); paths still
    travelling at t_max count as not arriving (check MCResult.timed_out). A path whose
    survival weight falls below w_min is dropped, which lowers the estimate by at
    most about w_min. Only live paths are kept in the working arrays.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float).reshape(2)
    if np.hypot(x0[0], x0[1]) <= scn.r0:
        return MCResult(1.0, 0.0, n_paths, 0.0)
    if t_max is None:
        travel = (scn.R - scn.r0) / scn.v if scn.v > 0 else scn.R**2 / max(scn.D, 1e-12)
        t_max = 5.0 * travel + 20.0
    reflect = scn.outer == "reflect"
    noise = scn.sigma * np.sqrt(dt)
    bridge = bridge and scn.sigma > 0
    bridge_scale = 2.0 / (scn.sigma**2 * dt) if bridge else 0.0

    pos = np.tile(x0, (n_paths, 1))              # live paths only
    r_old = np.hypot(pos[:, 0], pos[:, 1])
    w = np.ones(n_paths)
    k_old = scn.detection_rate(pos)
    total = total_sq = 0.0                       # sums of finished paths' weights
    for _ in range(int(np.ceil(t_max / dt))):
        n = w.size
        if n == 0:
            break
        # drift -v x/|x| (Scenario.drift), using the radius already in hand
        p_new = pos * (1.0 - scn.v * dt / r_old)[:, None] + noise * rng.standard_normal((n, 2))
        r_new = np.hypot(p_new[:, 0], p_new[:, 1])
        keep = np.ones(n, dtype=bool)

        left = r_new > scn.R
        if left.any():
            if reflect:
                mirrored = 2.0 * scn.R - r_new[left]
                p_new[left] *= (mirrored / r_new[left])[:, None]
                r_new[left] = mirrored
            else:                                # absorb: the drone gives up
                keep &= ~left

        k_new = scn.detection_rate(p_new)
        decay = np.exp(-0.5 * (k_old + k_new) * dt)
        w *= decay

        hit = r_new <= scn.r0
        if bridge:
            gap = (r_old - scn.r0) * np.maximum(r_new - scn.r0, 0.0)
            hit |= rng.random(n) < np.exp(-bridge_scale * gap)
        hit &= keep
        if hit.any():
            done = w[hit] / np.sqrt(decay[hit])  # half-step refund
            total += float(done.sum())
            total_sq += float((done**2).sum())

        keep &= ~hit & (w >= w_min)
        pos, r_old, w, k_old = p_new[keep], r_new[keep], w[keep], k_new[keep]

    mean = total / n_paths
    var = max(total_sq - n_paths * mean**2, 0.0) / (n_paths - 1)
    return MCResult(estimate=mean, half_width=float(1.96 * np.sqrt(var / n_paths)),
                    n_paths=n_paths, timed_out=w.size / n_paths)
