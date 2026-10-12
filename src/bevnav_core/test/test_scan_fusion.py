"""Stage 5: pan-scan sequencing and six-view fusion on synthetic data."""

import math

from bevnav_core.bev import BevGrid, build_lut, FREE, OCCUPIED, UNKNOWN
from bevnav_core.camera_model import base_from_optical_matrix, PinholeCamera
from bevnav_core.fusion import fuse_views, FusedGrid, log_odds_to_tristate, view_to_fused
from bevnav_core.raycast import build_ray_bins, raycast
from bevnav_core.scan import PanScan
import numpy as np
import pytest

F = 320 / math.tan(0.61)
CAM = PinholeCamera(F, F, 320, 240, 640, 480, base_from_optical_matrix(0.45, 0.2618))
BEV = BevGrid(480, 480, (0.0, 5.3), (-3.0, 3.0))
FUSED = FusedGrid(0.05, 5.3)
STEP = math.radians(60)
LUT, RAYS = build_lut(BEV, CAM), build_ray_bins(BEV, (0.0, 0.0), 0.0087)


def view_of(is_floor, pan: float) -> np.ndarray:
    """Tri-state view at pan angle of a flat world (is_floor(x, y) in base_footprint)."""
    xv, yv = BEV.cell_centres()                 # view (pan) frame
    c, s = math.cos(pan), math.sin(pan)
    floor = is_floor(c * xv - s * yv, s * xv + c * yv)
    seed = np.where(LUT.valid, np.where(floor, FREE, OCCUPIED), UNKNOWN).astype(np.uint8)
    return raycast(seed, RAYS)


def test_pan_scan_sequence() -> None:
    scan = PanScan(6, STEP, 0.0087, 0.2, 15.0)
    assert scan.start(0.0, 2 * math.pi + 0.001) == pytest.approx(2 * math.pi)   # whole turns
    scan.on_joint(1.0, 2 * math.pi)                    # arrived at heading 0 at t = 1.0
    assert not scan.take(1.1) and scan.take(1.25)      # only frames after the settle time
    targets = [scan.advance()]
    for k in range(1, 6):
        scan.on_joint(2.0 + k, targets[-1])
        assert scan.take(2.3 + k)
        targets.append(scan.advance())
    assert np.diff(targets) == pytest.approx([STEP] * 5)   # always one short step forward
    assert targets[-1] == pytest.approx(4 * math.pi) and not scan.active  # home: full turn
    assert not scan.timed_out(100.0)


def test_six_views_fuse_into_square_room() -> None:
    room = lambda x, y: (np.abs(x) < 1.5) & (np.abs(y) < 1.5)  # noqa: E731
    views = [(view_of(room, k * STEP), k * STEP) for k in range(6)]
    state = log_odds_to_tristate(fuse_views(views, FUSED, BEV, (0.0, 0.0), -0.4, 0.85, 3.0))
    x, y = FUSED.cell_centres()
    r = np.hypot(x, y)
    inside = (np.abs(x) < 1.35) & (np.abs(y) < 1.35) & (r > 0.6)
    assert (state[inside] == FREE).mean() > 0.95                      # 360 degrees of floor
    edge = np.maximum(np.abs(x), np.abs(y))
    wall = (edge > 1.5) & (edge < 1.55)                              # first cells beyond the floor
    assert (state[wall] == OCCUPIED).mean() > 0.8                     # the four walls
    assert (state[(np.abs(x) > 1.7) | (np.abs(y) > 1.7)] == FREE).sum() == 0  # nothing beyond


def test_one_degree_pan_error_moves_a_5m_cell_by_9cm() -> None:
    target = np.zeros((BEV.rows, BEV.cols), np.uint8)
    xv, yv = BEV.cell_centres()
    target[np.unravel_index(np.argmin(np.hypot(xv - 5.0, yv)), xv.shape)] = OCCUPIED
    x, y = FUSED.cell_centres()
    where = []
    for pan in (0.0, math.radians(1.0)):
        hit = view_to_fused(target, pan, FUSED, BEV, (0.0, 0.0)) == OCCUPIED
        where.append((x[hit].mean(), y[hit].mean()))
    # Moves by 5 m x 1 deg = 8.7 cm, seen through 5 cm cells.
    assert math.dist(*where) == pytest.approx(5.0 * math.radians(1.0), abs=0.05)
