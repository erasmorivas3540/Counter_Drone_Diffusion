"""Branch-and-bound certificate for sensor layouts in the straight-line limit (D -> 0).

Claim certified: for every layout of K sensors,

    V(s) = min_theta sum_j h(|s_j|, theta - arg s_j)  <=  V_ring + eps,

where V_ring is the best regular ring's exposure and eps > 0 a chosen tolerance. With
eps small this shows no layout beats the ring by more than eps; it does not separate the
ring from layouts within eps of it (that is the job of a local certificate).

Model (rho = v = 1, Da scales out of the layout problem): annulus d < r < R, one sensor at
polar position (a, phi) exposes the ray at angle theta to

    h(a, t) = int_d^R exp(-(r^2 + a^2 - 2 r a cos t)/2) dr
            = sqrt(pi/2) exp(-a^2 sin^2 t / 2) [erf((R - a cos t)/sqrt2) - erf((d - a cos t)/sqrt2)],

with t = theta - phi. Facts used (all elementary):
  (M) for a >= 0, h(a, t) is nonincreasing in |t| on [0, pi] (the integrand is).
  (C) |d^2 h / d a^2| <= L2 = 5.5 for all a, t (integrand bound (w^2 + 1) e^{-w^2/2} <= 1.22
      times the interval length 4.5).
  Minimizers lie in the closed disk |s_j| <= R (moving a sensor inward to R raises
  coverage everywhere in the annulus), so radii range over [0, R].

Upper bound for a box B (each sensor j in [a_lo, a_hi] x [phi_lo, phi_hi]) and any finite
set of test angles T:

    V(s) <= min_{theta in T} sum_j  sup_{a, phi in B_j} h(a, theta - phi).

The sup over phi is taken at the angular distance t* closest to 0 (fact M), rounded DOWN to
a table node (still an upper bound by M). The sup over a uses the table's nodes covering
[a_lo, a_hi] plus the interpolation slack L2 * delta^2 / 8 (fact C). Floating-point error
of erf (~1e-15) is covered by a 1e-12 safety term.

Symmetry: rotate so the sensor with the largest radius sits at angle 0 and order the others
counterclockwise: phi_0 = 0, 0 <= phi_1 <= ... <= phi_{K-1} < 2 pi, a_j <= a_0.

Run:  python experiments/bnb_straight.py --K 4 --eps 0.01
Resumable: open boxes and counters are saved to results/checkpoints/bnb_K{K}_eps{eps}.npz
every minute; rerun the same command to continue.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
D_IN, R_OUT = 2.0, 6.5
L2 = 5.5
SAFETY = 1e-12
EXACT_NODES = 64          # radial ranges spanning fewer table nodes are evaluated exactly in angle
# Second-order (Taylor + LP) bound constants, per sensor, in Cartesian sensor coordinates:
#   |grad_y g| <= G_BOUND = max_c (2 + c sqrt(2 pi)) e^{-c^2/2} = 2.940... -> 2.95
#   ||Hess_y g|| <= L_BOUND = max_c sqrt(2 pi)(2 + c^2) e^{-c^2/2} = 2 sqrt(2 pi) = 5.0133 -> 5.02
# where g(y) = int_d^R exp(-|r e^{i theta} - y|^2/2) dr (integrals bounded over the whole line).
G_BOUND, L_BOUND = 2.95, 5.02
LP_MAX_HALFWIDTH = 0.35   # use the Taylor/LP bound once every half-width (radius, arc) is below this


def h_exact(a, t, d=D_IN, R=R_OUT):
    a = np.asarray(a, float)
    t = np.asarray(t, float)
    c, s = a * np.cos(t), a * np.sin(t)
    return np.sqrt(np.pi / 2) * np.exp(-s**2 / 2) * (erf((R - c) / np.sqrt(2)) - erf((d - c) / np.sqrt(2)))


def exposure(radii, angles, thetas):
    """I(theta) for a layout, exact (closed form)."""
    return h_exact(radii[None, :], thetas[:, None] - angles[None, :]).sum(axis=1)


class Table:
    """h on a grid: radii a_i = i*delta in [0, R], angles t_k = k*dt in [0, pi]."""

    def __init__(self, delta=0.005, nt=4096):
        self.delta, self.nt = delta, nt
        self.dt = np.pi / nt
        self.a = np.arange(0.0, R_OUT + delta / 2, delta)
        self.t = np.arange(nt + 1) * self.dt
        self.H = h_exact(self.a[:, None], self.t[None, :])
        self.slack = L2 * delta**2 / 8 + SAFETY

    def row_max(self, a_lo, a_hi):
        """max over a in [a_lo, a_hi] of h(a, t_k), for every table angle t_k (upper bound)."""
        i0 = max(int(np.floor(a_lo / self.delta)), 0)
        i1 = min(int(np.ceil(a_hi / self.delta)), self.a.size - 1)
        return self.H[i0:i1 + 1].max(axis=0) + self.slack

    def t_index(self, tstar):
        """Largest node <= t* (so h(t_node) >= h(t*) by monotonicity)."""
        return np.minimum(np.floor(tstar / self.dt).astype(int), self.nt)


def h_and_grad(a, t, d=D_IN, R=R_OUT):
    """h(a, t), dh/da and dh/dt in closed form."""
    c, s_ = a * np.cos(t), a * np.sin(t)
    uR, ud = R - c, d - c
    E = np.exp(-s_**2 / 2)
    Erf = erf(uR / np.sqrt(2)) - erf(ud / np.sqrt(2))
    Gd = np.exp(-ud**2 / 2) - np.exp(-uR**2 / 2)
    sp = np.sqrt(np.pi / 2)
    h = sp * E * Erf
    h_a = E * (np.cos(t) * Gd - a * np.sin(t)**2 * sp * Erf)
    h_t = -a * np.sin(t) * E * (Gd + c * sp * Erf)
    return h, h_a, h_t


def lp_bound(box, thetas):
    """Second-order bound: Taylor expansion in each sensor's (radius, angle) about the box
    center with a rigorous remainder, combined over test angles with LP dual weights.

    For theta in T:  I_s(theta) <= c_theta + g_theta . delta + Q   for all s in the box,
    delta = s - center (polar). For ANY weights y >= 0 summing to 1,
        min_theta I_s(theta) <= sum_theta y_theta (c_theta + Q) + sum_k |sum_theta y_theta g_theta,k| w_k,
    with w_k the half-widths. The LP picks good weights; the bound is valid for whatever
    weights it returns, so solver tolerances do not affect rigor."""
    from scipy.optimize import linprog
    K = len(box)
    a_c = (box[:, 0] + box[:, 1]) / 2
    p_c = (box[:, 2] + box[:, 3]) / 2
    hr = (box[:, 1] - box[:, 0]) / 2
    hp = (box[:, 3] - box[:, 2]) / 2
    r_hi = box[:, 1]
    t = thetas[:, None] - p_c[None, :]
    h, h_a, h_t = h_and_grad(a_c[None, :], t)
    c = h.sum(axis=1)                                         # (nT,)
    grads = np.concatenate([h_a, -h_t], axis=1)               # d/dphi = -d/dt ; (nT, 2K)
    halfw = np.concatenate([hr, hp])
    # Remainder per test angle: a sensor at perpendicular distance >= cmin from the line of
    # the ray has ||Hess|| <= sqrt(2 pi)(2 + c^2)e^{-c^2/2} and |grad| <= (2 + c sqrt(2 pi))e^{-c^2/2}
    # (the first is decreasing in c >= 0; the second is used at max(c, 0.6767), its maximizer).
    cmin = box[None, :, 0] * min_abs_sin(thetas, box[:, 2], box[:, 3])       # (nT, K)
    Lc = np.sqrt(2 * np.pi) * (2 + cmin**2) * np.exp(-cmin**2 / 2) * 1.0005
    cg = np.maximum(cmin, 0.6767)
    Gc = np.minimum((2 + cg * np.sqrt(2 * np.pi)) * np.exp(-cg**2 / 2) * 1.0005, G_BOUND)
    Q = 0.5 * np.sum(Lc * hr**2 + 2 * (r_hi * Lc + Gc) * hr * hp
                     + (r_hi**2 * Lc + r_hi * Gc) * hp**2, axis=1)                # (nT,)
    nT, n = grads.shape
    # maximize t  s.t.  t - g_theta . delta <= c_theta + Q,  |delta_k| <= halfw_k
    A = np.hstack([np.ones((nT, 1)), -grads])
    res = linprog(np.r_[-1.0, np.zeros(n)], A_ub=A, b_ub=c + Q,
                  bounds=[(None, None)] + [(-w, w) for w in halfw], method="highs")
    if res.status != 0:
        return np.inf
    y = np.clip(-res.ineqlin.marginals, 0, None)
    if y.sum() <= 0:
        return np.inf
    y = y / y.sum()
    return float(y @ (c + Q) + np.abs(y @ grads) @ halfw + SAFETY)


def min_abs_sin(theta, p_lo, p_hi):
    """min over phi in [p_lo, p_hi] of |sin(phi - theta)|, for theta (nT,) and K ranges -> (nT, K)."""
    th = theta[:, None]
    # distance from the angle range to the nearest multiple of pi (line through the origin)
    lo, hi = th - p_hi[None, :], th - p_lo[None, :]
    m = np.ceil(lo / np.pi)
    contains = (m * np.pi <= hi) | (hi - lo >= np.pi)
    dl = np.abs(lo - np.pi * np.round(lo / np.pi))
    dh = np.abs(hi - np.pi * np.round(hi / np.pi))
    dist = np.where(contains, 0.0, np.minimum(dl, dh))
    return np.sin(np.minimum(dist, np.pi / 2))


def closest_angle(theta, p_lo, p_hi):
    """min over phi in [p_lo, p_hi] of the angular distance |theta - phi| (mod 2 pi), in [0, pi]."""
    lo, hi = theta - p_hi, theta - p_lo                 # t ranges over [lo, hi]
    two_pi = 2 * np.pi
    m = np.ceil(lo / two_pi)
    contains = m * two_pi <= hi
    dl = np.abs(lo - two_pi * np.round(lo / two_pi))
    dh = np.abs(hi - two_pi * np.round(hi / two_pi))
    out = np.minimum(dl, dh)
    out = np.where(contains | (hi - lo >= two_pi), 0.0, out)
    return np.minimum(out, np.pi)


def upper_bound(box, table, thetas):
    """box: array (K, 4) rows [a_lo, a_hi, phi_lo, phi_hi]. Returns (U, argmin theta)."""
    total = np.zeros(thetas.size)
    for a_lo, a_hi, p_lo, p_hi in box:
        tstar = closest_angle(thetas, p_lo, p_hi)
        i0 = max(int(np.floor(a_lo / table.delta)), 0)
        i1 = min(int(np.ceil(a_hi / table.delta)), table.a.size - 1)
        if i1 - i0 < EXACT_NODES:
            # small radial range: evaluate h exactly at the closest angle t* (no angle rounding)
            nodes = table.a[i0:i1 + 1]
            total += h_exact(nodes[None, :], tstar[:, None]).max(axis=1) + table.slack
        else:
            rm = table.row_max(a_lo, a_hi)
            total += rm[table.t_index(tstar)]
    k = int(np.argmin(total))
    return total[k], thetas[k]


def feasible(box):
    """Ordering constraints: a_j <= a_0 and phi_1 <= ... <= phi_{K-1}."""
    if np.any(box[1:, 0] > box[0, 1]):
        return False
    lo, hi = box[1:, 2], box[1:, 3]
    for j in range(len(lo) - 1):
        if lo[j] > hi[j + 1]:
            return False
    return True


def test_angles(box, base):
    """Fixed grid plus the minima of the exposure of the box's center layout."""
    a = (box[:, 0] + box[:, 1]) / 2
    p = (box[:, 2] + box[:, 3]) / 2
    fine = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    I = exposure(a, p, fine)
    idx = np.argsort(I)[:12]
    return np.concatenate([base, fine[idx]])


def use_lp(box):
    hr = (box[:, 1] - box[:, 0]) / 2
    arc = (box[:, 3] - box[:, 2]) / 2 * np.maximum(box[:, 1], 0.25)
    return max(hr.max(), arc[1:].max() if len(box) > 1 else 0) <= LP_MAX_HALFWIDTH


def split(box):
    a_w = box[:, 1] - box[:, 0]
    arc = (box[:, 3] - box[:, 2]) * np.maximum(box[:, 1], 0.25)
    widths = np.concatenate([a_w, arc])
    k = int(np.argmax(widths))
    j, col = (k, 0) if k < len(box) else (k - len(box), 2)
    mid = (box[j, col] + box[j, col + 1]) / 2
    b1, b2 = box.copy(), box.copy()
    b1[j, col + 1] = mid
    b2[j, col] = mid
    return b1, b2


def ring_value(K):
    from scipy.optimize import minimize_scalar
    f = lambda a: -exposure(np.full(K, a), 2 * np.pi * np.arange(K) / K, np.array([np.pi / K]))[0]  # noqa
    grid = np.linspace(0.05, R_OUT, 400)
    i = int(np.argmin([f(a) for a in grid]))
    res = minimize_scalar(f, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 399)]), method="bounded",
                          options={"xatol": 1e-9})
    return res.x, -res.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    K, eps = args.K, args.eps
    a_ring, v_ring = ring_value(K)
    target = v_ring + eps
    table = Table()
    base = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ck = ROOT / "results" / "checkpoints" / f"bnb_K{K}_eps{eps:g}.npz"
    ck.parent.mkdir(parents=True, exist_ok=True)
    if ck.exists() and not args.fresh:
        z = np.load(ck)
        stack = list(z["stack"])
        stats = json.loads(str(z["stats"]))
        print(f"Resuming: {len(stack)} open boxes, {stats['processed']} processed", flush=True)
    else:
        root = np.zeros((K, 4))
        root[:, 1] = R_OUT
        root[1:, 3] = 2 * np.pi            # phi_0 = 0 fixed
        stack = [root]
        stats = {"processed": 0, "pruned": 0, "infeasible": 0, "max_depth_volume": 1.0,
                 "best_center_V": 0.0, "seconds": 0.0}
    print(f"K = {K}: best ring radius {a_ring:.6f}, V_ring = {v_ring:.7f}; certifying V <= "
          f"{target:.7f} (eps = {eps:g}); table slack {table.slack:.1e}", flush=True)
    t0, t_ck = time.time(), time.time()
    base_seconds = stats["seconds"]
    while stack:
        box = stack.pop()
        stats["processed"] += 1
        if not feasible(box):
            stats["infeasible"] += 1
            continue
        T = test_angles(box, base)
        U, _ = upper_bound(box, table, T)
        if U > target and use_lp(box):
            U = min(U, lp_bound(box, T))
        if U <= target:
            stats["pruned"] += 1
            continue
        a = (box[:, 0] + box[:, 1]) / 2
        p = (box[:, 2] + box[:, 3]) / 2
        vc = float(exposure(a, p, np.linspace(0, 2 * np.pi, 2048, endpoint=False)).min())
        if vc > stats["best_center_V"]:
            stats["best_center_V"] = vc
        if vc > target:
            print(f"!! layout with V = {vc:.6f} > target found: radii {a}, angles {p}", flush=True)
        stack.extend(split(box))
        now = time.time()
        if now - t_ck > 60 or not stack:
            stats["seconds"] = base_seconds + now - t0
            np.savez(ck, stack=np.array(stack) if stack else np.zeros((0, K, 4)),
                     stats=json.dumps(stats))
            t_ck = now
            print(f"  {stats['processed']:>10,} boxes, {len(stack):>8,} open, pruned "
                  f"{stats['pruned']:,}, best center V {stats['best_center_V']:.5f} "
                  f"({stats['seconds'] / 60:.1f} min)", flush=True)
            if stats["seconds"] / 60 > args.max_minutes:
                print("Time limit reached; rerun to continue.")
                return
    stats["seconds"] = base_seconds + time.time() - t0
    print(f"\nCERTIFIED: no {K}-sensor layout has V > {target:.7f} "
          f"(= ring {v_ring:.7f} + {eps:g}). {stats['processed']:,} boxes, "
          f"{stats['seconds'] / 60:.1f} min.", flush=True)
    out = ROOT / "results" / f"bnb_K{K}_eps{eps:g}.json"
    out.write_text(json.dumps({"K": K, "eps": eps, "ring_radius": a_ring, "V_ring": v_ring,
                               "certified_upper_bound": target, **stats}, indent=2))


if __name__ == "__main__":
    main()
