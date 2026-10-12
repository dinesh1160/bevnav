"""
Persistent BEV map in odom and the Galvis window cut from it (CLAUDE.md Stage 6).

The map is a log-odds grid laid out like nav_msgs/OccupancyGrid: log_odds[iy, ix] covers
x = origin_x + (ix + 0.5) * resolution, y = origin_y + (iy + 0.5) * resolution (odom, metres).
Observations are tri-state cells at known positions in base_footprint; each observed map cell
adds l_occ if any observation in it is occupied, else l_free if any is free, clamped to
[l_min, l_max]. Unknown observations add nothing. route_subgoal_xy gives the cost map a
sub-goal along the shortest route on this map (E5).
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from bevnav_core.bev import BevGrid, FREE, inflate, OCCUPIED, UNKNOWN
from bevnav_core.planner import farthest_visible, route_distance, route_path
from bevnav_core.transforms import from_frame, Pose2D
import cv2
import numpy as np


@dataclass(frozen=True)
class MapParams:
    """memory.yaml values: geometry (metres, cells) and log-odds increments and limits."""

    resolution_m: float
    size_cells: Tuple[int, int]     # (columns along x, rows along y)
    origin_m: Tuple[float, float]   # odom position of the lower-left corner
    l_free: float
    l_occ: float
    l_min: float
    l_max: float


class BevMap:
    """Log-odds occupancy grid: 0 = unknown, < 0 = free, > 0 = occupied."""

    def __init__(self, p: MapParams) -> None:
        self.p = p
        self.log_odds = np.zeros((p.size_cells[1], p.size_cells[0]), np.float32)

    def reset(self) -> None:
        """Forget everything."""
        self.log_odds[:] = 0.0

    def integrate(self, pose: Pose2D, x: np.ndarray, y: np.ndarray, state: np.ndarray) -> int:
        """
        Add one observation.

        pose: base_footprint in odom. x, y: (N,) metres in base_footprint; state: (N,) tri-state.
        Several observations in one map cell count once (occupied wins). Returns the number of
        known observations outside the map (dropped, never wrapped).
        """
        known = state != UNKNOWN
        points = from_frame(pose, np.stack([x[known], y[known]], axis=1))
        ix = np.floor((points[:, 0] - self.p.origin_m[0]) / self.p.resolution_m).astype(int)
        iy = np.floor((points[:, 1] - self.p.origin_m[1]) / self.p.resolution_m).astype(int)
        rows, cols = self.log_odds.shape
        inside = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        flat = (iy * cols + ix)[inside]
        observed = state[known][inside]
        occupied = np.zeros(self.log_odds.size, bool)
        free = np.zeros(self.log_odds.size, bool)
        occupied[flat[observed == OCCUPIED]] = True
        free[flat[observed == FREE]] = True
        lo = self.log_odds.reshape(-1)
        lo[occupied] += self.p.l_occ
        lo[free & ~occupied] += self.p.l_free
        np.clip(lo, self.p.l_min, self.p.l_max, out=lo)
        return int((~inside).sum())

    def sample(self, pose: Pose2D, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log-odds of the map cells under points x, y (metres in base_footprint); 0 outside."""
        points = from_frame(pose, np.stack([x.ravel(), y.ravel()], axis=1))
        ix = np.floor((points[:, 0] - self.p.origin_m[0]) / self.p.resolution_m).astype(int)
        iy = np.floor((points[:, 1] - self.p.origin_m[1]) / self.p.resolution_m).astype(int)
        rows, cols = self.log_odds.shape
        inside = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        out = np.zeros(len(points), np.float32)
        out[inside] = self.log_odds[iy[inside], ix[inside]]
        return out.reshape(x.shape)

    def tristate(self) -> np.ndarray:
        """(rows, cols) uint8 tri-state of the whole map (sign of the log-odds)."""
        out = np.full(self.log_odds.shape, UNKNOWN, np.uint8)
        out[self.log_odds < 0] = FREE
        out[self.log_odds > 0] = OCCUPIED
        return out

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """x, y: (rows, cols) odom metres of every map cell centre."""
        rows, cols = self.log_odds.shape
        x = self.p.origin_m[0] + (np.arange(cols) + 0.5) * self.p.resolution_m
        y = self.p.origin_m[1] + (np.arange(rows) + 0.5) * self.p.resolution_m
        return np.meshgrid(x, y)


def grid_points(x: np.ndarray, y: np.ndarray, state: np.ndarray, cell_m: Tuple[float, float],
                target_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points of a tri-state grid dense enough that no target cell falls between them.

    x, y: (R, C) cell centres (metres) of a grid with cell size cell_m = (along rows, along
    columns); state: (R, C). Each cell is split into k x k sub-cells with spacing at most
    target_m / sqrt(2), so a grid of any rotation covers every target_m cell under it. Returns
    flat x, y, state of the known sub-cells only.
    """
    k = max(1, math.ceil(max(cell_m) * math.sqrt(2) / target_m))
    known = state != UNKNOWN
    xs, ys, ss = x[known], y[known], state[known]
    if k == 1:
        return xs, ys, ss
    offsets = (np.arange(k) + 0.5) / k - 0.5
    ox, oy = np.meshgrid(offsets * cell_m[0], offsets * cell_m[1], indexing='ij')
    return ((xs[:, None] + ox.ravel()).ravel(), (ys[:, None] + oy.ravel()).ravel(),
            np.repeat(ss, k * k))


def drivable_window(bev_map: BevMap, pose: Pose2D, grid: BevGrid, p_free_threshold: float,
                    inflation_m: float) -> np.ndarray:
    """
    Galvis window cut from the map at pose: (rows, cols) bool, True = drivable.

    Drivable = P(free) > p_free_threshold and farther than inflation_m from any map cell that is
    more likely occupied than free. Unknown is not drivable. Nearest neighbour, no blending.
    An occupied map cell fills its whole square in the finer window while the obstacle's edge
    lies, on average, at the cell centre, so the window grows it by inflation_m less half a cell.
    """
    x, y = grid.cell_centres()
    log_odds = bev_map.sample(pose, x, y)
    free = log_odds < math.log((1.0 - p_free_threshold) / p_free_threshold)
    grow_m = max(inflation_m - bev_map.p.resolution_m / 2, 0.0)
    return free & ~inflate(log_odds > 0, grid, grow_m)


def route_subgoal_xy(bev_map: BevMap, pose: Pose2D, goal_xy: Tuple[float, float],
                     inflation_m: float, along_m: float,
                     margin_m: float) -> Optional[Tuple[float, float]]:
    """
    Sub-goal toward the goal along the shortest route on the map (odom metres, E5).

    The route is pulled straight: the sub-goal lies along_m metres from the robot toward the
    farthest route cell it can see in a straight line (or on that cell, if nearer). The route
    runs over the map cropped to the known cells, robot and goal plus margin_m.
    Cells within inflation_m of an occupied cell (as in drivable_window) block it; unknown
    cells do not, so the route explores. Discs of that radius around the robot and the goal
    stay open (the robot is there; a goal beside a shelf is still a goal). None when the goal
    cannot be reached inside the crop.
    """
    res = bev_map.p.resolution_m
    rows, cols = bev_map.log_odds.shape

    def cell(x: float, y: float) -> Tuple[int, int]:
        return (min(max(int((y - bev_map.p.origin_m[1]) / res), 0), rows - 1),
                min(max(int((x - bev_map.p.origin_m[0]) / res), 0), cols - 1))

    start, goal = cell(pose.x, pose.y), cell(*goal_xy)
    known_r, known_c = np.nonzero(bev_map.log_odds)
    pad = int(math.ceil(margin_m / res))
    r0 = max(min([start[0], goal[0], known_r.min(initial=rows)]) - pad, 0)
    r1 = min(max([start[0], goal[0], known_r.max(initial=0)]) + pad + 1, rows)
    c0 = max(min([start[1], goal[1], known_c.min(initial=cols)]) - pad, 0)
    c1 = min(max([start[1], goal[1], known_c.max(initial=0)]) + pad + 1, cols)
    grow = max(int(round((inflation_m - res / 2) / res)), 0)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    blocked = cv2.dilate((bev_map.log_odds[r0:r1, c0:c1] > 0).astype(np.uint8), disc) > 0
    for r, c in (start, goal):
        open_disc = np.zeros(blocked.shape, np.uint8)
        open_disc[r - r0, c - c0] = 1
        blocked &= cv2.dilate(open_disc, disc) == 0
    dist = route_distance(blocked, (goal[0] - r0, goal[1] - c0), res)
    path = route_path(dist, (start[0] - r0, start[1] - c0))
    if len(path) == 0:
        return None
    aim = path[farthest_visible(path, blocked)] + (r0, c0)
    aim_xy = np.array([bev_map.p.origin_m[0] + (aim[1] + 0.5) * res,
                       bev_map.p.origin_m[1] + (aim[0] + 0.5) * res])
    offset = aim_xy - (pose.x, pose.y)
    reach = float(np.hypot(*offset))
    if len(path) == 1 or reach <= along_m:
        return float(aim_xy[0]), float(aim_xy[1])
    sub = np.array([pose.x, pose.y]) + offset * (along_m / reach)
    return float(sub[0]), float(sub[1])
