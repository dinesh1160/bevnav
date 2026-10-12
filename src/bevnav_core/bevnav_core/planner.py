"""
Galvis greedy planner (Algorithm 1) on the BEV cost map.

From the robot (0, 0), repeatedly step step_m to the cheapest of the 8 neighbours. The path
stops when it reaches the grid boundary (ahead or sideways), when every neighbour is
non-drivable, or when it would revisit a cell from the last oscillation_window steps (D4).
"""

from collections import deque
import math
from typing import Tuple

from bevnav_core.bev import BevGrid
import numpy as np

_DIRECTIONS = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)])


def cost_at(cost: np.ndarray, grid: BevGrid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cost of the cells containing points (x, y) metres; +inf outside the grid."""
    dx, dy = grid.cell_m
    i = np.floor((grid.x_range_m[1] - x) / dx).astype(int)
    j = np.floor((grid.y_range_m[1] - y) / dy).astype(int)
    inside = (i >= 0) & (i < grid.rows) & (j >= 0) & (j < grid.cols)
    out = np.full(np.shape(x), np.inf, np.float32)
    out[inside] = cost[i[inside], j[inside]]
    return out


def plan_greedy(cost: np.ndarray, grid: BevGrid, step_m: float, oscillation_window: int,
                start_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Return the path as (N, 2) metres in base_footprint, starting at start_xy."""
    lattice = np.array(start_xy, float)               # positions are start + k * step
    pos = np.zeros(2, int)                            # in steps from the start
    recent = deque([tuple(pos)], maxlen=oscillation_window)
    path = [lattice.copy()]
    max_steps = math.ceil((grid.x_range_m[1] - grid.x_range_m[0]) / step_m
                          * (grid.y_range_m[1] - grid.y_range_m[0]) / step_m)
    for _ in range(max_steps):
        candidates = pos + _DIRECTIONS
        xy = lattice + candidates * step_m
        costs = cost_at(cost, grid, xy[:, 0], xy[:, 1])
        k = int(np.argmin(costs))
        if not np.isfinite(costs[k]) or tuple(candidates[k]) in recent:
            break
        pos = candidates[k]
        recent.append(tuple(pos))
        path.append(xy[k])
        x, y = xy[k]
        if (x > grid.x_range_m[1] - step_m or y > grid.y_range_m[1] - step_m
                or y < grid.y_range_m[0] + step_m):
            break                                     # reached the far or a side boundary
    return np.array(path)
