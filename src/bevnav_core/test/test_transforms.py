"""Planar pose helpers."""

import math

from bevnav_core.transforms import Pose2D, relative_pose, wrap_angle
import pytest


def test_wrap_angle_range() -> None:
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)


def test_relative_pose_of_origin_is_zero() -> None:
    p = Pose2D(-3.64, -6.5, math.pi / 2)
    r = relative_pose(p, p)
    assert (r.x, r.y, r.yaw) == pytest.approx((0.0, 0.0, 0.0))


def test_relative_pose_rotated_origin() -> None:
    # Origin faces +y in W; a point 1 m further along +y is 1 m straight ahead (+x).
    origin = Pose2D(2.0, 1.0, math.pi / 2)
    r = relative_pose(origin, Pose2D(2.0, 2.0, math.pi))
    assert (r.x, r.y, r.yaw) == pytest.approx((1.0, 0.0, math.pi / 2))
    # A point 1 m along -x in W is on the origin's left (+y).
    r = relative_pose(origin, Pose2D(1.0, 1.0, math.pi / 2))
    assert (r.x, r.y) == pytest.approx((0.0, 1.0))


def test_compose_inverts_relative_pose() -> None:
    from bevnav_core.transforms import compose
    origin, pose = Pose2D(2.0, 1.0, 0.7), Pose2D(-1.0, 3.0, -2.5)
    back = compose(origin, relative_pose(origin, pose))
    assert (back.x, back.y, back.yaw) == pytest.approx((pose.x, pose.y, pose.yaw))


def test_to_and_from_frame_are_inverse() -> None:
    import numpy as np
    from bevnav_core.transforms import from_frame, to_frame
    origin = Pose2D(2.0, 1.0, math.pi / 2)
    pts = np.array([[2.0, 2.0], [1.0, 1.0]])
    np.testing.assert_allclose(to_frame(origin, pts), [[1.0, 0.0], [0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(from_frame(origin, to_frame(origin, pts)), pts, atol=1e-12)
