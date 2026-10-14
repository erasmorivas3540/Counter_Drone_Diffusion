"""Question 4: the ON/OFF sensor simulator matches its exact limits."""
import numpy as np
import pytest

from cdd import Scenario, ring_sensors, solve
from cdd.downtime import (frozen_edge, simulate_u_switching, switching_rates, u_fast,
                          u_frozen_parts)

Z_MAX = 3.5


@pytest.fixture(scope="module")
def scn():
    return Scenario(r0=0.5, R=5.0, v=1.0, sigma=1.0, lam0=1.0, rho=1.0,
                    sensors=ring_sensors(4, 2.0), kernel="gaussian", outer="reflect")


X0 = np.array([4.5, 0.3])
P = 0.7


def test_rates():
    a, b = switching_rates(0.9, 2.0)
    assert a == pytest.approx(0.2) and b == pytest.approx(1.8)
    with pytest.raises(ValueError):
        switching_rates(0.0, 1.0)


def test_limits_are_ordered(scn):
    """Always-on <= fast switching <= frozen (Jensen), pointwise on the edge."""
    on = solve(scn, nr=100, ntheta=64).u[-1]
    fast = u_fast(scn, P, nr=100, ntheta=64).u[-1]
    unaware, aware = frozen_edge(u_frozen_parts(scn, P, nr=100, ntheta=64))
    assert np.all(on <= fast + 1e-12)
    assert np.all(fast <= unaware + 1e-12)
    assert unaware.max() <= aware + 1e-12


def test_frozen_probabilities_sum_to_one(scn):
    parts = u_frozen_parts(scn, P, nr=40, ntheta=16)
    assert sum(prob for _, prob, _ in parts) == pytest.approx(1.0)
    assert len(parts) == 2 ** len(scn.sensors)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["always_on", "frozen", "fast"])
def test_simulator_matches_limits(scn, case):
    if case == "always_on":
        ref = float(solve(scn, nr=200, ntheta=256)(X0)[0])
        p, gamma = 1.0, 0.0
    elif case == "fast":
        ref = float(u_fast(scn, P, nr=200, ntheta=256)(X0)[0])
        p, gamma = P, 1e4
    else:
        ref = 0.0
        for z, prob, _ in u_frozen_parts(scn, P, nr=200, ntheta=256):
            sub = Scenario(r0=0.5, R=5.0, v=1.0, sigma=1.0, lam0=1.0, rho=1.0,
                           sensors=scn.sensors[z])
            ref += prob * float(solve(sub, nr=200, ntheta=256)(X0)[0])
        p, gamma = P, 0.0
    r = simulate_u_switching(X0, scn, p, gamma, dt=0.01, n_paths=60_000, seed=11)
    assert abs(r.z_score(ref)) < Z_MAX


@pytest.mark.slow
def test_intermediate_switching_lies_between_limits(scn):
    fast = float(u_fast(scn, P, nr=200, ntheta=256)(X0)[0])
    r = simulate_u_switching(X0, scn, P, 1.0, dt=0.01, n_paths=60_000, seed=12)
    assert fast < r.estimate < 0.36          # frozen limit is 0.349 here


def test_ring_symmetry_and_truncation_match_full_sum():
    """Rotation-class solver equals the full 2^K sum; truncation stays within its bound."""
    from cdd.downtime import (frozen_bounds, u_frozen_parts_ring,
                              u_frozen_parts_ring_truncated)
    ring = Scenario(r0=2.0, R=6.5, v=1.0, sigma=1.0, lam0=1.0, rho=1.0,
                    sensors=ring_sensors(6, 3.2))
    full = u_frozen_parts(ring, 0.8, nr=60, ntheta=48)
    sym = u_frozen_parts_ring(ring, 0.8, nr=60, ntheta=48)
    assert max(np.abs(a[2] - b[2]).max() for a, b in zip(full, sym)) < 1e-12
    exact_unaware, exact_aware = frozen_edge(full)
    parts, dropped = u_frozen_parts_ring_truncated(ring, 0.8, nr=60, ntheta=48, tail_tol=0.02)
    ub_unaware, ub_aware = frozen_bounds(parts, dropped)
    assert 0 < dropped <= 0.02
    assert exact_aware <= ub_aware <= exact_aware + dropped + 1e-12
    assert exact_unaware.max() <= ub_unaware <= exact_unaware.max() + dropped + 1e-12
