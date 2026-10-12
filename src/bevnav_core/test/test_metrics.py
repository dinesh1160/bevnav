"""Section 7 trial outcomes and footprint clearance."""

from bevnav_core.metrics import (classify_trial, clearance_map, COLLISION, footprint_clearance,
                                 OutcomeRules, RUNNING, STALL, SUCCESS, TIMEOUT)
from bevnav_core.transforms import jittered_start, Pose2D
import numpy as np
import pytest

RULES = OutcomeRules(0.5, 30.0, 0.02, 0.10, 10.0)
GOAL = np.array([10.0, 0.0])


def drive(speed: float, seconds: float, rate: float = 10.0):
    t = np.arange(0, seconds, 1 / rate)
    return t, np.stack([speed * t, np.zeros_like(t)], 1)


def test_success_and_running() -> None:
    t, xy = drive(0.45, 25)
    assert classify_trial(t, xy, 0, GOAL, RULES) == SUCCESS
    assert classify_trial(t[:50], xy[:50], 0, GOAL, RULES, finished=False) == RUNNING


def test_collision_comes_first() -> None:
    t, xy = drive(0.45, 25)
    assert classify_trial(t, xy, 3, GOAL, RULES) == COLLISION
    assert classify_trial(t, xy, 0, GOAL, RULES, clearance=np.full(len(t), 0.01)) == COLLISION


def test_timeout_and_stall() -> None:
    t, xy = drive(0.2, 31)                     # too slow: 6.2 m by the 30 s limit
    assert classify_trial(t, xy, 0, GOAL, RULES) == TIMEOUT
    t, xy = drive(0.0, 15)                     # standing still for 15 s
    assert classify_trial(t, xy, 0, GOAL, RULES, finished=False) == STALL
    scanning = np.ones(len(t), bool)           # ... but scanning the whole time is not a stall
    assert classify_trial(t, xy, 0, GOAL, RULES, scanning=scanning, finished=False) == RUNNING


def test_footprint_clearance() -> None:
    occupied = np.zeros((100, 100), bool)
    occupied[:, 60:] = True                    # wall from x = 3.0 m (0.05 m cells, origin 0, 0)
    cmap = clearance_map(occupied, 0.05)
    xy, yaw = np.array([[2.0, 2.5]]), np.array([0.0])
    c = footprint_clearance(xy, yaw, np.array([0.4, 0.4]), cmap, np.array([0.0, 0.0]), 0.05, 0.05)
    assert c[0] == pytest.approx(0.8, abs=0.06)   # front edge at x = 2.2, wall at 3.0


def test_jittered_start_is_reproducible_and_bounded() -> None:
    start = Pose2D(-3.64, -6.5, np.pi / 2)
    a, b = jittered_start(start, 7, 0.2, 0.1745), jittered_start(start, 7, 0.2, 0.1745)
    assert a == b
    for seed in range(20):
        p = jittered_start(start, seed, 0.2, 0.1745)
        assert abs(p.y - start.y) < 1e-9          # facing +y: the offset is purely along x
        assert abs(p.x - start.x) <= 0.2 and abs(p.yaw - start.yaw) <= 0.1745
