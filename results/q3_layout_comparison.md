# Q3: does any 6-sensor layout beat the best ring?

K = 6, response distance d = 2, Da = 1. u = drone's best chance (fine grid). 'vs ring' > 0 means worse than the ring; |vs ring| < 0.0005 is a tie (search-grid accuracy).

## wandering (Pe = 0.5)  (best ring u = 0.4440)

| Method | Starts | u found | vs ring | Evaluations | Note |
|---|---|---|---|---|---|
| best ring (1-D search over radius) | - | 0.4440 | - | - |  |
| Powell from ring | 1 | 0.4441 | +0.0001 (tie) | 590 |  |
| Powell from staggered | 1 | 0.4654 | +0.0214 (worse) | 600 |  |
| DE unseeded, 40 generations + Powell | 110 random | 0.4900 | +0.0460 (worse) | 5310 |  |
| structured family: staggered | grid scan | 0.4441 | +0.0001 (tie) | 1050 | params 4.025 4.025 1.047 |
| structured family: zigzag | grid scan | 0.4440 | +0.0000 (tie) | 750 | params 4.026 4.026 |
| structured family: five_plus | grid scan | 0.5208 | +0.0768 (worse) | 1170 | params 3.886 2.682 0.628 |
| structured family: weighted | grid scan | 0.5469 | +0.1029 (worse) | 1350 | params 3.748 3.294 0.001 |
| seeded DE + Powell | 60 (5 seeded) | 0.4441 | +0.0001 (tie) | 6409 | best member came from seed ring |
| Powell, 20 random starts | 20 random | 0.5122 | +0.0682 (worse) | 8000 | median start ends at 0.6033; 0/20 within 0.002 of the ring |
| adjoint gradient (L-BFGS-B, soft-max) | 36 (30 random) | 0.4440 | -0.0000 (tie) | 3389 | 30/30 random starts tie the ring; worst start +2.0e-05 |

## mixed (Pe = 2)  (best ring u = 0.3101)

| Method | Starts | u found | vs ring | Evaluations | Note |
|---|---|---|---|---|---|
| best ring (1-D search over radius) | - | 0.3101 | - | - |  |
| Powell from ring | 1 | 0.3102 | +0.0001 (tie) | 600 |  |
| Powell from staggered | 1 | 0.3519 | +0.0419 (worse) | 600 |  |
| DE unseeded, 40 generations + Powell | 110 random | 0.4971 | +0.1870 (worse) | 5310 |  |
| structured family: staggered | grid scan | 0.3102 | +0.0002 (tie) | 1050 | params 3.252 3.251 1.047 |
| structured family: zigzag | grid scan | 0.3101 | +0.0000 (tie) | 750 | params 3.252 3.252 |
| structured family: five_plus | grid scan | 0.4004 | +0.0903 (worse) | 1170 | params 3.054 0.001 0.628 |
| structured family: weighted | grid scan | 0.4661 | +0.1561 (worse) | 1350 | params 2.829 2.450 0.001 |
| seeded DE + Powell | 60 (5 seeded) | 0.3101 | +0.0000 (tie) | 6426 | best member came from seed ring |
| Powell, 20 random starts | 20 random | 0.6725 | +0.3624 (worse) | 8000 | median start ends at 0.8102; 0/20 within 0.002 of the ring |
| adjoint gradient (L-BFGS-B, soft-max) | 36 (30 random) | 0.3101 | +0.0000 (tie) | 4258 | 30/30 random starts tie the ring; worst start +3.9e-05 |

## directed (Pe = 20)  (best ring u = 0.2710)

| Method | Starts | u found | vs ring | Evaluations | Note |
|---|---|---|---|---|---|
| best ring (1-D search over radius) | - | 0.2710 | - | - |  |
| Powell from ring | 1 | 0.2713 | +0.0004 (tie) | 600 |  |
| Powell from staggered | 1 | 0.3362 | +0.0653 (worse) | 600 |  |
| DE unseeded, 40 generations + Powell | 110 random | 0.4368 | +0.1658 (worse) | 5310 |  |
| structured family: staggered | grid scan | 0.2712 | +0.0002 (tie) | 1050 | params 2.592 2.591 1.047 |
| structured family: zigzag | grid scan | 0.2710 | +0.0000 (tie) | 750 | params 2.597 2.597 |
| structured family: five_plus | grid scan | 0.3817 | +0.1107 (worse) | 1170 | params 2.350 0.001 0.628 |
| structured family: weighted | grid scan | 0.4758 | +0.2049 (worse) | 1350 | params 2.060 1.738 0.001 |
| seeded DE + Powell | 60 (5 seeded) | 0.2714 | +0.0004 (tie) | 6409 | best member came from seed family:zigzag |
| Powell, 20 random starts | 20 random | 0.6504 | +0.3794 (worse) | 8000 | median start ends at 0.7288; 0/20 within 0.002 of the ring |
| adjoint gradient (L-BFGS-B, soft-max) | 36 (30 random) | 0.2711 | +0.0001 (tie) | 6166 | 30/30 random starts tie the ring; worst start +1.8e-04 |

**Verdict:** no method found a layout better than the best ring beyond grid accuracy.
