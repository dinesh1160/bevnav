"""Planar pose helpers: x, y in metres, yaw in radians (REP 103, CCW positive)."""

from dataclasses import dataclass
import math

import numpy as np


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


def compose(origin: Pose2D, local: Pose2D) -> Pose2D:
    """Express local (given in origin's frame) in the frame origin is given in."""
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return Pose2D(origin.x + c * local.x - s * local.y, origin.y + s * local.x + c * local.y,
                  wrap_angle(origin.yaw + local.yaw))


def to_frame(origin: Pose2D, points: np.ndarray) -> np.ndarray:
    """(N, 2) points given in some frame W -> the frame of origin (also given in W)."""
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    d = np.asarray(points, float) - (origin.x, origin.y)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def from_frame(origin: Pose2D, points: np.ndarray) -> np.ndarray:
    """(N, 2) points given in origin's frame -> the frame origin is given in."""
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    p = np.asarray(points, float)
    return np.stack([origin.x + c * p[:, 0] - s * p[:, 1], origin.y + s * p[:, 0] + c * p[:, 1]],
                    axis=1)


def jittered_start(start: Pose2D, seed: int, lateral_m: float, yaw_rad: float) -> Pose2D:
    """
    Trial start pose: start shifted sideways by up to lateral_m and turned by up to yaw_rad.

    Uniform, reproducible per seed; the same function places the robot (bringup) and
    evaluates the trial (bevnav_eval).
    """
    rng = np.random.default_rng(seed)
    side, turn = rng.uniform(-lateral_m, lateral_m), rng.uniform(-yaw_rad, yaw_rad)
    return Pose2D(start.x - side * math.sin(start.yaw), start.y + side * math.cos(start.yaw),
                  wrap_angle(start.yaw + turn))
