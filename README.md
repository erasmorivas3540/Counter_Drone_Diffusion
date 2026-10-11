# Counter-drone diffusion

Code for the project *Does random maneuvering help a drone slip past sensors? A diffusion
model for counter-UAS sensor layouts.*

It computes **u(x): the chance that a drone starting at x reaches a defended site without
being detected**, two independent ways (a finite-difference PDE solver and a Monte Carlo
simulator), and checks both against exact answers.

## The model

- **Drone:** Brownian motion with drift toward the site at the origin,
  dX = −v X/|X| dt + σ dW (diffusion coefficient D = σ²/2).
- **Sensors:** detection is a Poisson process with rate κ(x), the sum of one bump per sensor:
  Gaussian, λ₀ exp(−d²/2ρ²); disk, λ₀ within range ρ; or a uniform rate λ₀ for the exact cases.
- **Geometry:** the site is the disk |x| ≤ r₀; the area edge |x| = R either reflects the
  drone or counts as giving up.
- **Feynman–Kac:** u(x) = E[exp(−∫₀^τ κ(X_s) ds)] solves
  D Δu − v ∂u/∂r − κ u = 0, with u = 1 on the site edge and ∂u/∂r = 0 (reflecting) or
  u = 0 (absorbing) at R.
- **Dimensionless numbers:** Péclet Pe = vρ/D (how directed the approach is) and
  detection number Da = λ₀ρ/v (expected detections per sensor range crossed at full speed).
  `Scenario.peclet` and `Scenario.detection_number` report them.

Default units: lengths in sensor ranges (ρ = 1), time in range-crossings at full speed (v = 1).

## Set up

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda env create -f environment.yml
pip install -e ".[test]"
```

Needs Python 3.10+, NumPy, SciPy and Matplotlib.

## Check that it works

```bash
pytest -q                                   # all 43 checks, about 40 s
pytest -q -m "not slow"                     # skip the Monte Carlo checks, about 10 s
python experiments/validation_report.py     # tables + figure, about 3 min (--quick: 1 min)
```

The report writes `results/validation.md`, `results/validation.json` and
`figures/validation.png` / `.pdf`.

## Use it

```python
import numpy as np
from cdd import Scenario, ring_sensors, solve, simulate_u

scn = Scenario(r0=0.5, R=5.0, v=1.0, sigma=1.0, lam0=1.0, rho=1.0,
               sensors=ring_sensors(4, 2.0), kernel="gaussian", outer="reflect")

sol = solve(scn, nr=200, ntheta=256)     # u on the whole area, under a second
sol(np.array([4.5, 0.0]))                # chance of arriving undetected from (4.5, 0): 0.162
sol.worst_entry()                        # the drone's best entry point on the edge: (u, angle)
simulate_u([4.5, 0.0], scn)              # Monte Carlo estimate with a 95% interval
```

## What is in the repository

| Path | Contents |
|---|---|
| `src/cdd/model.py` | `Scenario` (site, drone, sensors, edge), detection kernels, `ring_sensors` |
| `src/cdd/exact.py` | Exact references: 1D closed form, 2D Bessel solution, radial ODE (`solve_bvp`) |
| `src/cdd/montecarlo.py` | `simulate_u` (2D) and `simulate_u_1d`, returning an estimate and 95% interval |
| `src/cdd/pde.py` | `solve`: polar-grid finite differences and a sparse direct solve |
| `tests/` | The validation ladder (rungs 1–5) plus structural checks |
| `experiments/validation_report.py` | Reruns the ladder with more paths; tables and figure for the paper |
| `results/`, `figures/` | Generated outputs |
| `configs/`, `paper/` | For Phase 2 experiment settings and the manuscript |

## Validation ladder

| Rung | Case | Reference | Checked |
|---|---|---|---|
| 1 | 1D straight approach, constant rate | closed form (inverse-Gaussian Laplace transform) | simulator |
| 2 | 2D, no drift, constant rate, both edge types | modified Bessel functions | PDE, simulator |
| 3 | 2D, radial drift, constant rate, both edge types | radial ODE, `solve_bvp` | PDE (3 schemes), simulator |
| 4 | Grid refinement | exact (rung 3) and self-convergence (rung 5) | order 2 central, order 1 upwind |
| 5 | Four Gaussian sensors on a ring | PDE on a fine grid | simulator at 3 starting angles |

Also checked: zero-noise flight against direct integration (sensors included), the ring's
symmetries, rotation of the layout, more detection always lowering u, and input validation.
The latest full report is in `results/validation.md`.

## Numerical methods (for the Methods section)

**Simulator.** Euler–Maruyama steps with survival weights exp(−∫κ) (trapezoid rule in
time). Two corrections keep coarse steps accurate: a Brownian-bridge check for paths that
touch the site between steps, with probability exp(−2 d_old d_new / σ²Δt), and a
half-step refund of the detection hazard on the arriving step. The area edge reflects by
mirroring. Paths whose weight falls below `w_min` (default 1e-12) are dropped, which
changes u by at most `w_min`.

**PDE solver.** Uniform polar grid; radial diffusion in conservative form; the drift term
by central differences (second order), upwind differences (first order, always monotone)
or a hybrid that uses central wherever the cell Péclet number v h / 2D is at most 1.
A mirrored ghost ring imposes the reflecting edge; angles are periodic. All off-diagonal
coefficients are nonnegative, so the discrete u stays in [0, 1]. The sparse system is
solved directly (SuperLU through SciPy): a 400 × 512 grid takes about 3 s.

**Known limits.** The simulator's remaining time-step bias is first order, about
0.04·Δt on rung 3 (under 0.0005 at Δt = 0.01, inside the Monte Carlo noise). With
σ = 0 the PDE falls back to first-order upwind. The disk kernel's jump in κ will lower
the PDE's order; check convergence before using it in experiments.
