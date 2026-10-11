"""Shared settings for the validation ladder.

Monte Carlo checks pass when the reference value lies within Z_MAX standard errors of
the simulated estimate. Seeds are fixed, so every run gives the same numbers; Z_MAX is
set above the usual 1.96 so that a different seed does not fail the suite by chance
(|z| > 3.5 happens about 1 time in 2,000 when nothing is wrong). Real bugs show up
as errors of many standard errors. experiments/validation_report.py reruns the same
comparisons with more paths and reports the z-scores themselves.
"""
import numpy as np
import pytest

from cdd import Scenario, ring_sensors

Z_MAX = 3.5


@pytest.fixture(scope="session")
def ring_scenario() -> Scenario:
    """Rung 5 test case: four Gaussian sensors on a ring of radius 2 around the site."""
    return Scenario(r0=0.5, R=5.0, v=1.0, sigma=1.0, lam0=1.0, rho=1.0,
                    sensors=ring_sensors(4, 2.0), kernel="gaussian", outer="reflect")


def on_circle(radius: float, angle: float) -> np.ndarray:
    return radius * np.array([np.cos(angle), np.sin(angle)])
