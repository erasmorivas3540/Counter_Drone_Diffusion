# Real-world scenarios mapped onto the model

Six sensors on a ring; response line at 2 footprints (2ρ), area edge at 6.5ρ. Assumed inputs (time to an actionable detection, maneuver spread σ_v and correlation time τ) are illustrative; speeds and ranges are sourced.

| ID | Drone | Sensor | v (m/s) | ρ (m) | Pe | Da | Best ring (m) | Straight-line ring (m) | Drone's chance | With straight-line ring | Flight time (s) | Long outage if > (s) | Brief if < (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | quadcopter, direct approach | short-range radar, open terrain | 15 | 1000 | 2.3e+02 | 13 | 2453 | 2453 | 0.000 | 0.000 | 300 | 300 | 10 |
| A2 | quadcopter, direct approach | short-range radar, heavy clutter | 15 | 1000 | 2.3e+02 | 1.1 | 2453 | 2453 | 0.235 | 0.235 | 300 | 300 | 10 |
| A3 | quadcopter, evasive weaving | short-range radar, heavy clutter | 10 | 1000 | 6.7 | 1.7 | 2807 | 2453 | 0.130 | 0.139 | 452 | 452 | 15 |
| B1 | loitering munition (Switchblade-class) | short-range radar, RF-cued classification | 28 | 1000 | 1e+03 | 1.8 | 2453 | 2453 | 0.097 | 0.097 | 161 | 161 | 5 |
| C1 | one-way attack drone (Shahed-class) | long-range radar | 50 | 3000 | 5.6e+03 | 3 | 7358 | 7358 | 0.020 | 0.020 | 270 | 270 | 9 |
| D1 | quadcopter, slow evasive approach | EO/IR camera network | 8 | 300 | 3.8 | 3.8 | 883 | 736 | 0.025 | 0.030 | 170 | 170 | 6 |
| D2 | quadcopter, searching/loitering | EO/IR camera network | 3 | 300 | 0.7 | 10 | 1008 | 736 | 0.013 | 0.025 | 401 | 401 | 13 |
| E1 | quadcopter, searching/loitering | acoustic array | 3 | 60 | 0.14 | 10 | 241 | 147 | 0.120 | 0.227 | 34 | 34 | 1 |
| E2 | quadcopter, slow evasive approach | acoustic array | 8 | 60 | 0.75 | 3.8 | 212 | 147 | 0.089 | 0.142 | 31 | 31 | 1 |
