"""Stage 3 geometry on synthetic data: projection signs, checkerboard IPM, left/right, ray-cast."""

import math
import time

from bevnav_core.bev import BevGrid, build_lut, FREE, OCCUPIED, UNKNOWN, warp_floor_mask
from bevnav_core.camera_model import base_from_optical_matrix, PinholeCamera
from bevnav_core.raycast import build_ray_bins, raycast
import numpy as np
import pytest

# Robot camera (robot.yaml): 640 x 480, HFOV 1.22 rad, lens 0.45 m, pitch 0.2618 rad.
W, H, HFOV, HEIGHT, PITCH = 640, 480, 1.22, 0.45, 0.2618
F = (W / 2) / math.tan(HFOV / 2)
CAM = PinholeCamera(F, F, W / 2, H / 2, W, H, base_from_optical_matrix(HEIGHT, PITCH))
GRID = BevGrid(480, 480, (0.0, 5.3), (-3.0, 3.0))
STEP = math.radians(0.5)


def render_floor(is_floor) -> np.ndarray:
    """Image mask (H, W) uint8 of a flat world: 255 where the pixel's floor point is floor."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ray_opt = np.stack([(u - CAM.cx) / CAM.fx, (v - CAM.cy) / CAM.fy, np.ones_like(u, float)], -1)
    ray = ray_opt @ CAM.base_from_optical[:3, :3].T
    origin = CAM.base_from_optical[:3, 3]
    down = ray[..., 2] < 0
    t = np.where(down, -origin[2] / np.where(down, ray[..., 2], -1), 0)
    x, y = origin[0] + t * ray[..., 0], origin[1] + t * ray[..., 1]
    return np.where(down & is_floor(x, y), 255, 0).astype(np.uint8)


def cell(x: float, y: float):
    dx, dy = GRID.cell_m
    return int((GRID.x_range_m[1] - x) / dx), int((GRID.y_range_m[1] - y) / dy)


def test_projection_signs() -> None:
    axis_hit = HEIGHT / math.tan(PITCH)   # where the optical axis meets the floor (1.68 m)
    u, v, vis = CAM.project(np.array([[axis_hit, 0.0, 0.0], [1.0, 0.0, 0.0],
                                      [2.0, 1.0, 0.0], [2.0, -1.0, 0.0]]))
    assert vis.all()
    assert (u[0], v[0]) == pytest.approx((CAM.cx, CAM.cy))   # pitch sign and height
    assert v[1] > CAM.cy                                     # nearer floor: lower in the image
    assert u[2] < CAM.cx < u[3]                              # left in the world: left in image
    u, v, vis = CAM.project(np.array([[1000.0, 0.0, HEIGHT]]))  # far point at lens height
    assert v[0] == pytest.approx(CAM.cy - F * math.tan(PITCH), abs=0.5)  # horizon above centre


def test_matrix_matches_robot_tf() -> None:
    # tf2_echo base_footprint camera_optical_frame: RPY (-105, 0, -90) deg, xyz (0, 0, 0.45).
    r, y = math.radians(-105), math.radians(-90)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    np.testing.assert_allclose(CAM.base_from_optical[:3, :3], rz @ rx, atol=1e-6)


def test_checkerboard_lands_on_known_cells() -> None:
    square = 0.25
    board = lambda x, y: (np.floor(x / square) + np.floor(y / square)) % 2 == 0  # noqa: E731
    seed = warp_floor_mask(render_floor(board), build_lut(GRID, CAM))
    x, y = GRID.cell_centres()
    interior = ((np.abs(x / square - np.round(x / square)) > 0.1)
                & (np.abs(y / square - np.round(y / square)) > 0.1))
    check = (seed != UNKNOWN) & interior & (x < 3.0)
    assert check.sum() > 20000
    agree = (seed[check] == FREE) == board(x[check], y[check])
    assert agree.mean() > 0.995


def test_blind_zone_and_frustum_are_unknown() -> None:
    seed = warp_floor_mask(render_floor(lambda x, y: np.ones_like(x, bool)), build_lut(GRID, CAM))
    blind = HEIGHT / math.tan(PITCH + math.atan(H / 2 / F))
    assert seed[cell(blind - 0.05, 0.0)] == UNKNOWN
    assert seed[cell(blind + 0.05, 0.0)] == FREE
    assert seed[cell(1.0, 2.0)] == UNKNOWN          # outside the 35 deg half field of view


def test_left_box_stays_left() -> None:
    box = lambda x, y: (x > 2.0) & (x < 2.5) & (y > 0.5) & (y < 1.0)  # noqa: E731
    mask = render_floor(lambda x, y: ~box(x, y))
    view = raycast(warp_floor_mask(mask, build_lut(GRID, CAM)), build_ray_bins(GRID, (0, 0), STEP))
    assert view[cell(2.02, 0.75)] == OCCUPIED       # near face of the box, on the left (y > 0)
    assert view[cell(2.02, -0.75)] == FREE          # mirror position on the right is floor
    assert view[cell(1.5, 0.75)] == FREE            # floor in front of the box
    assert view[cell(3.0, 1.5)] == UNKNOWN          # hidden behind the box


def test_raycast_l_corridor() -> None:
    # Floor: 1 m wide corridor straight ahead to x = 3, then a left turn (2 < x < 3, y up to 3).
    floor = lambda x, y: ((np.abs(y) < 0.5) & (x < 3.0)) | ((x > 2.0) & (x < 3.0) & (y > 0) & (y < 3.0))  # noqa: E731,E501
    x, y = GRID.cell_centres()
    seed = np.where(floor(x, y), FREE, OCCUPIED).astype(np.uint8)
    view = raycast(seed, build_ray_bins(GRID, (0, 0), STEP))
    assert view[cell(1.0, 0.0)] == FREE
    assert view[cell(3.005, 0.0)] == OCCUPIED       # end wall: first contact straight ahead
    assert view[cell(3.5, 0.0)] == UNKNOWN          # behind the end wall
    assert view[cell(2.5, 2.5)] == UNKNOWN          # floor around the corner is not visible
    assert view[cell(1.0, 0.6)] == UNKNOWN          # wall behind the corridor's side contact


def test_warp_and_raycast_under_20_ms() -> None:
    mask = render_floor(lambda x, y: (x < 4.0) | (np.abs(y) > 1.0))
    lut, rays = build_lut(GRID, CAM), build_ray_bins(GRID, (0, 0), STEP)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        raycast(warp_floor_mask(mask, lut), rays)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 0.020
