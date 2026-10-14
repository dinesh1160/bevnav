"""Stage 4 on synthetic masks: centre-line, cost map, greedy planner, pure pursuit."""

from bevnav_core.bev import BevGrid
from bevnav_core.centerline import fit_centerline, Line, row_midpoints
from bevnav_core.costmap import cost_map, CostParams
from bevnav_core.planner import cost_at, plan_greedy
from bevnav_core.pursuit import pursuit_command, PursuitParams
import numpy as np
import pytest

GRID = BevGrid(480, 480, (0.0, 5.3), (-3.0, 3.0))
X, Y = GRID.cell_centres()
COST = CostParams(3.0, 0.5, 2.5, 0.05, 151)   # costmap.yaml values
PURSUIT = PursuitParams(0.75, 0.45, 1.0, 2.0)
AHEAD = (5.3, 0.0)    # D3 default goal


def plan(drivable: np.ndarray, goal=AHEAD) -> np.ndarray:
    xs, ys = row_midpoints(drivable, GRID)
    total, _ = cost_map(drivable, GRID, goal, fit_centerline(xs, ys, 40, 0.10, 100), COST)
    return plan_greedy(total, GRID, 0.10, 10)


def test_centerline_of_offset_corridor() -> None:
    drivable = np.abs(Y - 0.3) < 0.6
    line = fit_centerline(*row_midpoints(drivable, GRID), 40, 0.10, 100)
    assert line.m == pytest.approx(0.0, abs=0.01) and line.b == pytest.approx(0.3, abs=0.02)


def test_centerline_in_a_camera_wedge() -> None:
    # Real masks: the drivable area starts as a narrow wedge at the robot (field of view).
    wedge = np.abs(Y) < 0.7 * X
    drivable = wedge & (np.abs(Y - 0.2) < 0.6)
    drivable[-3:, :] = False                       # apex rows not drivable at all
    line = fit_centerline(*row_midpoints(drivable, GRID), 40, 0.10, 100)
    # Near the robot the field-of-view edges bound the rows, so check the line at range.
    assert line is not None and line.m * 4.0 + line.b == pytest.approx(0.2, abs=0.05)


def test_centerline_needs_40_rows() -> None:
    drivable = (np.abs(Y) < 0.5) & (X < 0.3)        # only ~27 rows have both edges
    assert fit_centerline(*row_midpoints(drivable, GRID), 40, 0.10, 100) is None


def test_cost_terms() -> None:
    drivable = np.abs(Y) < 1.0
    total, terms = cost_map(drivable, GRID, AHEAD, Line(0.0, 0.0), COST)
    assert np.isinf(total[~drivable]).all() and np.isfinite(total[drivable]).all()
    for t in terms.values():
        assert t[drivable].min() == pytest.approx(0.0) and t[drivable].max() == pytest.approx(1.0)
    row = 240
    assert terms['oa'][row, np.argmin(np.abs(Y[row] - 0.95))] > terms['oa'][row, 240]  # near wall
    assert terms['g'][0, 240] < terms['g'][479, 240]                          # far row nearer goal


def test_straight_corridor_goes_straight_to_the_far_edge() -> None:
    path = plan(np.abs(Y) < 0.5)
    assert path[-1, 0] > 5.1                        # no early oscillation stop
    assert np.abs(path[:, 1]).max() < 0.25


def test_bend_is_followed() -> None:
    drivable = ((np.abs(Y) < 0.5) & (X < 3.0)) | ((X > 2.0) & (X < 3.0) & (Y > -0.5))
    path = plan(drivable, goal=(2.5, 3.0))
    assert path[-1, 1] > 2.5                        # turned left into the bend
    free = np.where(drivable, 0.0, np.inf)
    assert np.isfinite(cost_at(free, GRID, path[1:, 0], path[1:, 1])).all()


def test_blocked_corridor_stops_before_the_wall() -> None:
    drivable = (np.abs(Y) < 0.5) & (X < 2.0)
    path = plan(drivable)
    assert path[:, 0].max() < 2.0
    assert pursuit_command(path[-1:] - path[-1], PURSUIT) == (0.0, 0.0)  # at the path end: stop


def _cells(path: np.ndarray):
    i = np.floor((GRID.x_range_m[1] - path[:, 0]) / GRID.cell_m[0]).astype(int)
    j = np.floor((GRID.y_range_m[1] - path[:, 1]) / GRID.cell_m[1]).astype(int)
    ok = (i >= 0) & (i < GRID.rows) & (j >= 0) & (j < GRID.cols)
    return i[ok], j[ok]


def test_box_beside_the_line_is_passed() -> None:
    box = (X > 2.0) & (X < 2.5) & (Y > 0.15) & (Y < 0.8)   # narrows the corridor on the left
    path = plan((np.abs(Y) < 1.0) & ~box)
    assert path[-1, 0] > 3.0                        # got past the box
    assert not box[_cells(path)].any()


def test_box_across_the_line_stops_safely() -> None:
    # Known limitation of the greedy planner (design spec Section 9): an obstacle across the
    # straight line to the goal is a local minimum. The path must stop before it, not hit it.
    box = (X > 2.0) & (X < 2.5) & (Y > -0.15) & (Y < 0.6)
    path = plan((np.abs(Y) < 1.0) & ~box)
    assert path[:, 0].max() < 2.0
    assert not box[_cells(path)].any()


def test_narrow_gap_is_passed() -> None:
    # Scenario D: two blocks leave a 0.6 m gap in a 2.2 m corridor.
    blocks = (X > 2.0) & (X < 2.5) & (np.abs(Y) > 0.3)
    path = plan((np.abs(Y) < 1.1) & ~blocks)
    assert path[-1, 0] > 3.0
    at_gap = (path[:, 0] > 2.0) & (path[:, 0] < 2.5)
    assert np.abs(path[at_gap, 1]).max() < 0.3


def test_pursuit_signs_and_limits() -> None:
    straight = np.stack([np.linspace(0, 3, 31), np.zeros(31)], 1)
    assert pursuit_command(straight, PURSUIT) == pytest.approx((0.45, 0.0))
    left = np.stack([np.linspace(0, 1, 11), np.linspace(0, 1, 11)], 1)
    v, w = pursuit_command(left, PURSUIT)
    assert w > 0 and v < 0.45 and abs(w) <= 1.0
    assert pursuit_command(np.zeros((0, 2)), PURSUIT) == (0.0, 0.0)
    assert pursuit_command(np.array([[-0.5, 0.2]]), PURSUIT) == (0.0, 0.0)   # behind: stop
