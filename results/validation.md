# Validation report

Monte Carlo: 200,000 paths per case, time step 0.01, 95% half-widths; z = (reference - estimate) / standard error.

## Rung 1: 1D closed form

| Case | Reference | Monte Carlo | z |
|---|---|---|---|
| 1D, sigma = 0.5 | 0.40698 | 0.40753 ± 0.00069 | -1.55 |
| 1D, sigma = 1.0 | 0.48092 | 0.48068 ± 0.00107 | +0.44 |
| 1D, sigma = 2.0 | 0.60653 | 0.60646 ± 0.00132 | +0.11 |

## Rung 2: no drift, Bessel closed form (PDE grid 200 x 16)

| Case | Reference | PDE | Monte Carlo | z |
|---|---|---|---|---|
| no drift, reflecting edge, r = 1.5 | 0.15052 | 0.15054 | 0.15006 ± 0.00107 | +0.84 |
| no drift, absorbing edge, r = 1.5 | 0.15023 | 0.15025 | 0.14976 ± 0.00107 | +0.86 |

## Rung 3: radial drift, solve_bvp reference (PDE grid 200 x 16)

| Case | Reference | PDE | Monte Carlo | z |
|---|---|---|---|---|
| radial drift, reflecting edge, r = 2.5 | 0.34952 | 0.34954 | 0.34882 ± 0.00093 | +1.48 |
| radial drift, absorbing edge, r = 2.5 | 0.34065 | 0.34066 | 0.34163 ± 0.00097 | -1.98 |

## Rung 5: sensor ring, PDE reference (grid 400 x 512)

| Case | Reference | PDE | Monte Carlo | z |
|---|---|---|---|---|
| 4-sensor ring, start r = 4.5, over a sensor | 0.16206 | 0.16206 | 0.16201 ± 0.00058 | +0.17 |
| 4-sensor ring, start r = 4.5, pi/8 off | 0.18358 | 0.18358 | 0.18296 ± 0.00065 | +1.87 |
| 4-sensor ring, start r = 4.5, between sensors | 0.20544 | 0.20544 | 0.20548 ± 0.00071 | -0.10 |

## Rung 4: grid convergence (radial drift case, max error over the grid)

| nr | central | hybrid | upwind |
|---|---|---|---|
| 25 | 9.80e-04 | 9.80e-04 | 1.45e-02 |
| 50 | 2.49e-04 | 2.49e-04 | 7.38e-03 |
| 100 | 6.26e-05 | 6.26e-05 | 3.72e-03 |
| 200 | 1.57e-05 | 1.57e-05 | 1.87e-03 |
| 400 | 3.92e-06 | 3.92e-06 | 9.35e-04 |
| 800 | 9.79e-07 | 9.79e-07 | 4.68e-04 |

Fitted orders: central 1.99, hybrid 1.99, upwind 0.99.
Sensor ring self-convergence orders (three grids, three starts): 1.99, 1.99, 1.99.

## Time-step bias, 1D, sigma = 1 (estimate minus exact)

| dt | plain | with bridge check and half-step refund |
|---|---|---|
| 0.1 | -0.0499 ± 0.0011 | -0.0001 ± 0.0011 |
| 0.05 | -0.0393 ± 0.0011 | -0.0009 ± 0.0011 |
| 0.02 | -0.0263 ± 0.0011 | -0.0002 ± 0.0011 |
| 0.01 | -0.0196 ± 0.0011 | -0.0004 ± 0.0011 |

All 10 Monte Carlo comparisons: largest |z| = 1.98, mean z = +0.21.
