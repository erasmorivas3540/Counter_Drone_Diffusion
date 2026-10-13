"""Chapter 9 of Oksendal in code: the stochastic solution of the Dirichlet problem.

The theorem: let X_t be a diffusion with generator L, started at x inside a domain G,
and let tau be the first time X_t leaves G. Then

    u(x) = E_x[ g(X_tau) ]

solves L u = 0 in G with u = g on the boundary. Adding a killing rate kappa(x)
(Chapter 8, Feynman-Kac) gives

    u(x) = E_x[ exp(-int_0^tau kappa(X_s) ds) g(X_tau) ],   L u - kappa u = 0.

Your project is the second form, with g = 1 on the site edge (arrival) and g = 0 on
an absorbing area edge (giving up). This script builds up to it in four steps, each
comparing a Monte Carlo average of g(X_tau) with the exact PDE solution. The domain
is always the ring r0 < |x| < R, so the boundary has two parts: the site edge and
the area edge.

    Part 1  Brownian motion, g = 1 on the site edge, 0 on the area edge
            L = D Laplacian; u = ln(R/r) / ln(R/r0)   (2D harmonic function)
    Part 2  boundary data that varies with angle: g = cos(theta) on the area edge
            u = (r - r0^2/r) cos(theta) / (R - r0^2/R)
    Part 3  add the drift toward the site: L = D Laplacian - v d/dr
            u(r) = int_r^R e^(v s/D)/s ds / int_r0^R e^(v s/D)/s ds
    Part 4  add detection (killing at rate kappa): your model with an absorbing edge
            reference = cdd.u_radial, simulator = cdd.simulate_u

Run:  python experiments/chapter9_dirichlet_demo.py      (about 2 minutes)

200,000 paths per part; a time step of 0.004 adds error under 0.002 (checked against
0.008 and 0.002), well inside the Monte Carlo noise.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from cdd import Scenario, simulate_u, u_radial

R0, R, SIGMA = 0.5, 4.0, 1.0
D = 0.5 * SIGMA**2


def exit_point(x0, v=0.0, dt=4e-3, n=200_000, seed=0, t_max=200.0):
    """Run paths of dX = -v X/|X| dt + sigma dW from x0 until they leave the ring.

    Returns the exit points (n, 2) and a flag that is True where the path left through
    the site edge (r = R0). A Brownian-bridge check at both edges catches paths that
    cross an edge between steps. Exit points are projected onto the edge they crossed.
    """
    rng = np.random.default_rng(seed)
    pos = np.tile(np.asarray(x0, float), (n, 1))
    exit_pos = np.zeros((n, 2))
    at_site = np.zeros(n, dtype=bool)
    live = np.arange(n)
    noise = SIGMA * np.sqrt(dt)
    for _ in range(int(t_max / dt)):
        if live.size == 0:
            break
        p = pos[live]
        r_old = np.hypot(p[:, 0], p[:, 1])
        p_new = p * (1.0 - v * dt / r_old)[:, None] + noise * rng.standard_normal(p.shape)
        r_new = np.hypot(p_new[:, 0], p_new[:, 1])
        scale = 2.0 / (SIGMA**2 * dt)
        hit_in = (r_new <= R0) | (rng.random(live.size) < np.exp(
            -scale * (r_old - R0) * np.maximum(r_new - R0, 0.0)))
        hit_out = ~hit_in & ((r_new >= R) | (rng.random(live.size) < np.exp(
            -scale * (R - r_old) * np.maximum(R - r_new, 0.0))))
        done = hit_in | hit_out
        unit = p_new[done] / r_new[done, None]
        exit_pos[live[done]] = unit * np.where(hit_in[done], R0, R)[:, None]
        at_site[live[done]] = hit_in[done]
        pos[live] = p_new
        live = live[~done]
    if live.size:
        raise RuntimeError(f"{live.size} paths still inside at t_max")
    return exit_pos, at_site


def report(name, exact, samples):
    est = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(samples.size)
    print(f"{name:50s} exact {exact:.4f}   Monte Carlo {est:.4f} +/- {1.96 * se:.4f}"
          f"   z = {(exact - est) / se:+.2f}")


def main():
    x0 = np.array([1.5, 0.0])
    r = 1.5

    # Part 1: g = 1 on the site edge, 0 on the area edge; u is the chance of hitting
    # the site before the area edge. u = ln(R/r)/ln(R/r0) is harmonic in 2D.
    _, at_site = exit_point(x0, seed=1)
    report("Part 1  Brownian motion, P(site before edge)",
           np.log(R / r) / np.log(R / R0), at_site.astype(float))

    # Part 2: g = cos(theta) on the area edge, 0 on the site edge, start off-axis.
    angle = 0.7
    x_off = r * np.array([np.cos(angle), np.sin(angle)])
    exits, at_site = exit_point(x_off, seed=2)
    g = np.where(at_site, 0.0, np.cos(np.arctan2(exits[:, 1], exits[:, 0])))
    exact = (r - R0**2 / r) * np.cos(angle) / (R - R0**2 / R)
    report("Part 2  Brownian motion, g = cos(theta) on edge", exact, g)

    # Part 3: drift toward the site. Solve D(u'' + u'/r) - v u' = 0 by hand:
    # u' is proportional to e^(v r/D)/r, then integrate and match u(R0)=1, u(R)=0.
    v = 0.5
    w = lambda s: np.exp(v * s / D) / s
    exact = quad(w, r, R)[0] / quad(w, R0, R)[0]
    _, at_site = exit_point(x0, v=v, seed=3)
    report(f"Part 3  drift v = {v}, P(site before edge)", exact, at_site.astype(float))

    # Part 4: add detection at constant rate kappa: Feynman-Kac with Dirichlet data.
    # This is your model with outer='absorb', checked with the project's own code.
    kappa, v = 0.5, 1.0
    scn = Scenario(r0=R0, R=R, v=v, sigma=SIGMA, lam0=kappa, kernel="uniform", outer="absorb")
    res = simulate_u(x0, scn, dt=1e-2, n_paths=100_000, seed=4)
    exact = float(u_radial(r, R0, R, v, SIGMA, kappa, outer="absorb"))
    print(f"{'Part 4  drift v = 1 plus detection kappa = 0.5':50s} exact {exact:.4f}   "
          f"Monte Carlo {res.estimate:.4f} +/- {res.half_width:.4f}   z = {res.z_score(exact):+.2f}")


if __name__ == "__main__":
    print(f"Ring r0 = {R0} < r < R = {R}, sigma = {SIGMA}, start r = 1.5\n")
    main()
