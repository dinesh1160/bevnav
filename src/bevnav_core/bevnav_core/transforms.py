"""Planar pose helpers: x, y in metres, yaw in radians (REP 103, CCW positive)."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Pose2D:
    """A pose on the floor plane: position (m) and heading (rad)."""

    x: float
    y: float
    yaw: float


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to (-pi, pi]."""
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped == -math.pi else wrapped


def relative_pose(origin: Pose2D, pose: Pose2D) -> Pose2D:
    """Express pose (given in some frame W) in the frame of origin (also given in W)."""
    dx, dy = pose.x - origin.x, pose.y - origin.y
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return Pose2D(c * dx + s * dy, -s * dx + c * dy, wrap_angle(pose.yaw - origin.yaw))
