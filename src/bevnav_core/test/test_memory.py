"""Stage 6: persistent map, window cut-out, blind-disc fill and scan triggers (synthetic)."""

import math

from bevnav_core.bev import BevGrid, build_lut, FREE, inflate, OCCUPIED, UNKNOWN
from bevnav_core.camera_model import base_from_optical_matrix, PinholeCamera
from bevnav_core.fusion import FusedGrid
from bevnav_core.memory import BevMap, drivable_window, grid_points, MapParams, route_subgoal_xy
from bevnav_core.raycast import build_ray_bins, fill_blind_disc, raycast
from bevnav_core.scan import ScanTrigger
from bevnav_core.transforms import from_frame, Pose2D
import numpy as np
import pytest

F = 320 / math.tan(0.61)
CAM = PinholeCamera(F, F, 320, 240, 640, 480, base_from_optical_matrix(0.45, 0.2618))
BEV = BevGrid(480, 480, (0.0, 5.3), (-3.0, 3.0))
LUT, RAYS = build_lut(BEV, CAM), build_ray_bins(BEV, (0.0, 0.0), 0.0087)
PARAMS = MapParams(0.05, (600, 600), (-15.0, -15.0), -0.7, 0.85, -2.0, 3.5)


def corridor_with_box(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Floor of a 3 m wide corridor along +x (odom) with a 0.5 m box on its left at x = 2 m."""
    box = (np.abs(x - 2.25) < 0.25) & (np.abs(y - 0.9) < 0.25)
    return (np.abs(y) < 1.5) & ~box


def view_at(pose: Pose2D, is_floor) -> np.ndarray:
    """Ray-cast tri-state forward view (BEV grid) of a flat odom world seen from pose."""
    x, y = BEV.cell_centres()
    world = from_frame(pose, np.stack([x.ravel(), y.ravel()], axis=1))
    floor = is_floor(world[:, 0], world[:, 1]).reshape(x.shape)
    seed = np.where(LUT.valid, np.where(floor, FREE, OCCUPIED), UNKNOWN).astype(np.uint8)
    return raycast(seed, RAYS)


def integrate_view(bev_map: BevMap, pose: Pose2D, view: np.ndarray) -> None:
    x, y = BEV.cell_centres()
    bev_map.integrate(pose, *grid_points(x, y, view, BEV.cell_m, PARAMS.resolution_m))


def test_box_is_kept_after_it_leaves_the_view() -> None:
    bev_map = BevMap(PARAMS)
    for x in np.arange(0.0, 4.01, 0.25):                 # drive past the box
        pose = Pose2D(float(x), 0.0, 0.0)
        integrate_view(bev_map, pose, view_at(pose, corridor_with_box))
    near_face = bev_map.sample(Pose2D(0.0, 0.0, 0.0), np.array([2.02]), np.array([0.9]))
    assert near_face[0] > 0                                 # still occupied, now behind the robot
    back = drivable_window(bev_map, Pose2D(4.0, 0.0, math.pi), BEV, 0.65, 0.2)
    x, y = BEV.cell_centres()                               # looking back: box ahead, on the right
    box_in_window = (np.abs(x - 1.75) < 0.25) & (np.abs(y + 0.9) < 0.25)
    assert not back[box_in_window].any()
    assert back[(np.abs(x - 1.0) < 0.3) & (np.abs(y) < 0.3)].all()   # free floor before it


def test_window_matches_a_direct_view_at_the_same_pose() -> None:
    pose = Pose2D(1.0, -2.0, 0.7)
    world = lambda x, y: corridor_with_box(*from_frame(Pose2D(0.0, -2.0, 0.7),  # noqa: E731
                                                       np.stack([x, y], 1)).T)
    view = view_at(pose, world)
    bev_map = BevMap(PARAMS)
    integrate_view(bev_map, pose, view)
    window = drivable_window(bev_map, pose, BEV, 0.65, 0.2)
    direct = (view == FREE) & ~inflate(view == OCCUPIED, BEV, 0.2)
    known = view != UNKNOWN
    assert (window[known] == direct[known]).mean() > 0.97   # differences only at 5 cm cell edges
    assert not window[~known].any()                         # unknown is never drivable


def test_points_outside_the_map_are_dropped() -> None:
    bev_map = BevMap(PARAMS)
    state = np.array([FREE, FREE], np.uint8)
    assert bev_map.integrate(Pose2D(14.9, 0.0, 0.0), np.array([0.0, 1.0]), np.zeros(2),
                             state) == 1
    assert (bev_map.log_odds < 0).sum() == 1


def test_blind_disc_is_filled_only_where_the_floor_continues() -> None:
    fused = FusedGrid(0.05, 2.0)
    x, y = fused.cell_centres()
    r = np.hypot(x, y)
    state = np.where(r > 0.5, FREE, UNKNOWN).astype(np.uint8)
    obstacle = (r > 0.5) & (r < 0.6) & (np.abs(np.arctan2(y, x)) < 0.3)   # just ahead
    state[obstacle] = OCCUPIED
    filled = fill_blind_disc(state, x, y, fused.resolution_m)
    disc = r < 0.45
    behind = disc & (np.abs(np.arctan2(y, x)) > 0.5)
    ahead = disc & (r > 0.1) & (np.abs(np.arctan2(y, x)) < 0.2)
    assert (filled[behind] == FREE).all()
    assert (filled[ahead] == UNKNOWN).all()
    assert (filled[r > 0.5] == state[r > 0.5]).all()


def test_scan_trigger() -> None:
    trigger = ScanTrigger(2.0, 0.1, 5.0)
    assert trigger.update(0.0, (0.0, 0.0), 1) == 'start'
    trigger.scanned(0.0, (0.0, 0.0))
    assert trigger.update(1.0, (0.0, 0.0), 1) is None          # no path, but not moved since
    assert trigger.update(2.0, (1.0, 0.0), 10) is None
    assert trigger.update(3.0, (2.0, 0.0), 10) == 'interval'
    trigger.scanned(3.0, (2.0, 0.0))
    assert trigger.update(4.0, (2.5, 0.0), 1) == 'blocked'     # path ended after moving
    trigger.scanned(4.0, (2.5, 0.0))
    assert trigger.update(6.0, (2.52, 0.0), 10) is None
    assert trigger.update(9.0, (2.53, 0.0), 10) == 'stalled'   # < 0.1 m in 5 s


def test_route_subgoal_leaves_a_dead_end() -> None:
    bev_map = BevMap(PARAMS)
    x, y = bev_map.cell_centres()
    walls = ((np.abs(x + 1.0) < 0.05) | (np.abs(x - 1.0) < 0.05)) & (y > -3.0) & (y < 6.0)
    walls |= (np.abs(y - 6.0) < 0.05) & (np.abs(x) < 1.05)     # corridor along +y, closed north
    bev_map.log_odds[walls] = 3.5
    bev_map.log_odds[(np.abs(x) < 0.95) & (y > -3.0) & (y < 5.95)] = -2.0
    robot = Pose2D(0.0, 3.0, math.pi / 2)                       # in the corridor, facing the end
    sub = route_subgoal_xy(bev_map, robot, (0.0, 9.0), 0.275, 3.0, 2.0)
    assert sub is not None and sub[1] < robot.y - 2.5           # back out southwards
    sub_open = route_subgoal_xy(BevMap(PARAMS), robot, (0.0, 9.0), 0.275, 3.0, 2.0)
    assert sub_open == pytest.approx((0.0, 6.0), abs=0.1)       # nothing known: straight on
    oblique = route_subgoal_xy(BevMap(PARAMS), robot, (1.3, 13.0), 0.275, 3.0, 2.0)
    norm = math.hypot(1.3, 10.0)                                # straight at the goal
    assert oblique == pytest.approx((3.0 * 1.3 / norm, 3.0 + 3.0 * 10.0 / norm), abs=0.05)
