"""Question 4: sensors that go down and come back (two-state Markov switching).

Each sensor j is ON or OFF, independently of the others and of the drone:

    ON  -> OFF at rate alpha (failure, jamming, maintenance)
    OFF -> ON  at rate beta  (repair)

Parameterized by availability p = beta / (alpha + beta) (long-run fraction of time ON) and
switching rate gamma = alpha + beta (1/gamma is the correlation time of a sensor's state).
The drone arrives at a random time, so each sensor starts in its stationary law (ON with
probability p). Only ON sensors detect:

    kappa(x, Z) = sum_j Z_j * lam0 * g(|x - s_j| / rho),   Z_j in {0, 1}.

The drone's chance of arriving undetected is u(x) = E[exp(-int_0^tau kappa(X_t, Z_t) dt)],
averaging over both the flight and the sensor states.

Two limits are exact PDE problems (both use cdd.pde.solve):

  fast switching (gamma -> infinity): each sensor acts as if ON at the reduced rate p*lam0,
      so u solves the usual PDE with lam0 replaced by p*lam0.
  frozen (gamma -> 0): the sensors keep their starting states for the whole flight, so
      u = sum over subsets S of ON sensors of p^|S| (1-p)^(K-|S|) u_S, with u_S the
      no-downtime solution for the sensors in S.

By Jensen's inequality, path by path, E_Z exp(-int kappa) >= exp(-E_Z int kappa), so the
frozen limit always gives the drone at least as good a chance as fast switching. The
simulator covers everything in between.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from .model import Scenario
from .montecarlo import MCResult
from .pde import solve


def switching_rates(p: float, gamma: float):
    """(alpha, beta): ON->OFF and OFF->ON rates for availability p and switching rate gamma."""
    if not 0.0 < p <= 1.0 or gamma < 0:
        raise ValueError("need 0 < p <= 1 and gamma >= 0")
    return (1.0 - p) * gamma, p * gamma


def u_fast(scn: Scenario, p: float, nr=200, ntheta=256, scheme="hybrid"):
    """Fast-switching limit: the usual problem with every sensor's rate scaled by p."""
    return solve(replace(scn, lam0=p * scn.lam0), nr=nr, ntheta=ntheta, scheme=scheme)


def frozen_states(K: int, p: float):
    """All 2^K ON/OFF patterns with their probabilities."""
    for z in itertools.product((0, 1), repeat=K):
        z = np.array(z, bool)
        yield z, p ** z.sum() * (1 - p) ** (K - z.sum())


def u_frozen_parts(scn: Scenario, p: float, nr=200, ntheta=256, scheme="hybrid"):
    """Frozen limit: list of (pattern, probability, u on the outer edge) for every pattern.

    The edge values let callers form both drone models:
      unaware drone (does not know which sensors are down): max_theta sum_S P(S) u_S(theta)
      aware drone (sees the outage, then picks its entry):  sum_S P(S) max_theta u_S(theta)
    """
    parts, cache = [], {}
    for z, prob in frozen_states(len(scn.sensors), p):
        key = z.tobytes()
        if key not in cache:
            sol = solve(replace(scn, sensors=scn.sensors[z]), nr=nr, ntheta=ntheta, scheme=scheme)
            cache[key] = sol.u[-1].copy()
        parts.append((z, prob, cache[key]))
    return parts


def u_frozen_parts_ring(scn: Scenario, p: float, nr=200, ntheta=264, scheme="hybrid"):
    """Same as u_frozen_parts for a regular ring with sensor 0 at angle 0, using symmetry.

    Rotating an ON/OFF pattern by k sensors rotates its solution by 2 pi k / K, i.e. rolls
    the edge values by k * ntheta / K grid points, so only one pattern per rotation class is
    solved (14 instead of 64 for K = 6). Needs ntheta divisible by K."""
    K = len(scn.sensors)
    if ntheta % K:
        raise ValueError("ntheta must be divisible by the number of sensors")
    shift = ntheta // K
    solved, parts = {}, []
    for z, prob in frozen_states(K, p):
        rots = [tuple(np.roll(z, k)) for k in range(K)]
        canon = min(rots)
        if canon not in solved:
            zc = np.array(canon, bool)
            sol = solve(replace(scn, sensors=scn.sensors[zc]), nr=nr, ntheta=ntheta, scheme=scheme)
            solved[canon] = sol.u[-1].copy()
        k = rots.index(canon)            # z = roll(canon, -k): sensors shifted back by k
        parts.append((z, prob, np.roll(solved[canon], -k * shift)))
    return parts


def u_frozen_parts_ring_truncated(scn: Scenario, p: float, nr=200, ntheta=264, tail_tol=1e-3,
                                  scheme="hybrid"):
    """Ring frozen limit keeping only the likeliest outage patterns.

    Patterns are kept in order of how many sensors are OFF (0, 1, 2, ...) until the
    probability of all remaining patterns is at most tail_tol. Returns (parts, dropped),
    where dropped is that leftover probability. Since 0 <= u <= 1, counting every dropped
    pattern as u = 1 gives an upper bound on the drone's chance that is off by at most
    `dropped` (see frozen_bounds)."""
    from math import comb
    K = len(scn.sensors)
    if ntheta % K:
        raise ValueError("ntheta must be divisible by the number of sensors")
    tail, max_off = 1.0, -1
    while tail > tail_tol and max_off < K:
        max_off += 1
        tail -= comb(K, max_off) * p ** (K - max_off) * (1 - p) ** max_off
    tail = max(tail, 0.0)
    shift = ntheta // K
    solved, parts = {}, []
    for z, prob in frozen_states(K, p):
        if K - z.sum() > max_off:
            continue
        rots = [tuple(np.roll(z, k)) for k in range(K)]
        canon = min(rots)
        if canon not in solved:
            zc = np.array(canon, bool)
            sol = solve(replace(scn, sensors=scn.sensors[zc]), nr=nr, ntheta=ntheta, scheme=scheme)
            solved[canon] = sol.u[-1].copy()
        k = rots.index(canon)
        parts.append((z, prob, np.roll(solved[canon], -k * shift)))
    return parts, tail


def frozen_bounds(parts, dropped):
    """(unaware upper bound, aware upper bound): dropped patterns counted as u = 1."""
    unaware, aware = frozen_edge(parts)
    return float(unaware.max()) + dropped, aware + dropped


def frozen_edge(parts):
    """(unaware drone's edge profile, aware drone's chance) from u_frozen_parts output."""
    mean_edge = sum(prob * edge for _, prob, edge in parts)
    aware = sum(prob * edge.max() for _, prob, edge in parts)
    return mean_edge, float(aware)


def simulate_u_switching(x0, scn: Scenario, p: float, gamma: float, dt=5e-3, n_paths=50_000,
                         seed=0, t_max=None, bridge=True, w_min=1e-12) -> MCResult:
    """Monte Carlo estimate of u(x0) with Markov ON/OFF sensors.

    Same scheme as cdd.montecarlo.simulate_u (Euler-Maruyama, trapezoid survival weights,
    Brownian-bridge arrival check, half-step refund, mirror reflection), plus an exact
    two-state transition for every sensor on every path and step:
        P(ON -> OFF in dt)  = (1 - p) (1 - e^{-gamma dt}),
        P(OFF -> ON in dt)  = p (1 - e^{-gamma dt}).
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float).reshape(2)
    if np.hypot(*x0) <= scn.r0:
        return MCResult(1.0, 0.0, n_paths, 0.0)
    if scn.kernel != "gaussian":
        raise ValueError("switching simulator implemented for the gaussian kernel")
    if t_max is None:
        travel = (scn.R - scn.r0) / scn.v if scn.v > 0 else scn.R**2 / max(scn.D, 1e-12)
        t_max = 5.0 * travel + 20.0
    S = scn.sensors
    K = S.shape[0]
    jump = 1.0 - np.exp(-gamma * dt)
    p_off, p_on = (1.0 - p) * jump, p * jump
    reflect = scn.outer == "reflect"
    noise = scn.sigma * np.sqrt(dt)
    bridge = bridge and scn.sigma > 0
    bridge_scale = 2.0 / (scn.sigma**2 * dt) if bridge else 0.0

    def per_sensor(pts):
        d2 = ((pts[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)
        return scn.lam0 * np.exp(-d2 / (2.0 * scn.rho**2))          # (n, K)

    pos = np.tile(x0, (n_paths, 1))
    r_old = np.hypot(pos[:, 0], pos[:, 1])
    z = rng.random((n_paths, K)) < p                                  # stationary start
    w = np.ones(n_paths)
    k_old = (per_sensor(pos) * z).sum(axis=1)
    total = total_sq = 0.0
    for _ in range(int(np.ceil(t_max / dt))):
        n = w.size
        if n == 0:
            break
        p_new = pos * (1.0 - scn.v * dt / r_old)[:, None] + noise * rng.standard_normal((n, 2))
        r_new = np.hypot(p_new[:, 0], p_new[:, 1])
        keep = np.ones(n, bool)
        left = r_new > scn.R
        if left.any():
            if reflect:
                mirrored = 2.0 * scn.R - r_new[left]
                p_new[left] *= (mirrored / r_new[left])[:, None]
                r_new[left] = mirrored
            else:
                keep &= ~left
        if gamma > 0:
            u01 = rng.random((n, K))
            z = np.where(z, u01 >= p_off, u01 < p_on)
        k_new = (per_sensor(p_new) * z).sum(axis=1)
        decay = np.exp(-0.5 * (k_old + k_new) * dt)
        w *= decay
        hit = r_new <= scn.r0
        if bridge:
            gap = (r_old - scn.r0) * np.maximum(r_new - scn.r0, 0.0)
            hit |= rng.random(n) < np.exp(-bridge_scale * gap)
        hit &= keep
        if hit.any():
            done = w[hit] / np.sqrt(decay[hit])
            total += float(done.sum())
            total_sq += float((done**2).sum())
        keep &= ~hit & (w >= w_min)
        pos, r_old, w, k_old, z = p_new[keep], r_new[keep], w[keep], k_new[keep], z[keep]
    mean = total / n_paths
    var = max(total_sq - n_paths * mean**2, 0.0) / (n_paths - 1)
    return MCResult(mean, float(1.96 * np.sqrt(var / n_paths)), n_paths, w.size / n_paths)
