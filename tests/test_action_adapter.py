"""The §4.3 action contract, pinned with hand-computed values.

These are the fixtures C1 (`dronevla_core`) will have to reproduce when it replaces
`dronevla.action_adapter`.
"""
import math

import numpy as np
import pytest

from dronevla.action_adapter import (ActionLimits, adapt, body_to_world_velocity,
                                     world_to_body_velocity)


def test_inside_caps_is_unchanged():
    a = adapt([0.3, -0.2, 0.0, 0.0, -1.0])
    assert a.rejected is None
    np.testing.assert_array_equal(a.applied, [0.3, -0.2, 0.0, 0.0])
    assert not any(a.flags.values())
    assert a.stop_positive is False


def test_horizontal_cap_scales_the_vector_not_each_axis():
    # a per-axis clip would give (0.5, 0.5); the contract caps the norm and keeps direction
    a = adapt([3.0, 4.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(a.applied[:2], [0.3, 0.4], atol=1e-12)
    assert a.flags["horizontal_capped"]


def test_horizontal_cap_boundary_is_inclusive():
    a = adapt([0.5, 0.0, 0.0, 0.0, 0.0])
    assert a.applied[0] == 0.5 and not a.flags["horizontal_capped"]


def test_planar_mode_zeroes_vz_and_yaw_rate_and_says_so():
    a = adapt([0.1, 0.0, 0.2, 0.3, 0.0], planar=True)
    np.testing.assert_array_equal(a.applied, [0.1, 0.0, 0.0, 0.0])
    assert a.flags["planar_zeroed"]


def test_non_planar_caps_vz_and_yaw_rate():
    a = adapt([0.0, 0.0, -1.0, 2.0, 0.0], planar=False)
    np.testing.assert_array_equal(a.applied, [0.0, 0.0, -0.3, 0.5])
    assert a.flags["vertical_capped"] and a.flags["yaw_capped"]


@pytest.mark.parametrize("bad", [
    [np.nan, 0, 0, 0, 0], [0, 0, 0, 0, np.inf], [0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
def test_bad_actions_are_rejected_not_raised(bad):
    a = adapt(bad)
    assert a.applied is None and a.rejected


def test_stop_is_positive_strictly_above_zero():
    assert adapt([0, 0, 0, 0, 0.0]).stop_positive is False
    assert adapt([0, 0, 0, 0, 1e-9]).stop_positive is True


def test_custom_limits():
    a = adapt([1.0, 0.0, 0.0, 0.0, 0.0], ActionLimits(horizontal_mps=0.2))
    assert a.applied[0] == pytest.approx(0.2)


def test_body_frame_at_90_degrees_yaw():
    # FLU: body +x (forward) at yaw +90 deg points along world +y; body +y (left) along -x
    np.testing.assert_allclose(body_to_world_velocity([1, 0, 0], math.pi / 2), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(body_to_world_velocity([0, 1, 0], math.pi / 2), [-1, 0, 0], atol=1e-12)


def test_body_world_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        v, yaw = rng.normal(size=3), rng.uniform(-math.pi, math.pi)
        np.testing.assert_allclose(world_to_body_velocity(body_to_world_velocity(v, yaw), yaw),
                                   v, atol=1e-12)
