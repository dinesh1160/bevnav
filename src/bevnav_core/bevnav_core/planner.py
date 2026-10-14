"""
Galvis greedy planner (Algorithm 1) on the BEV cost map, and the route sub-goal (E5).

From the robot (0, 0), repeatedly step step_m to the cheapest of the 8 neighbours. The path
stops when it reaches the grid boundary (ahead or sideways), when every neighbour is
non-drivable, or when it would revisit a cell from the last oscillation_window steps (D4).

Route (E5, Stage 6): the shortest 8-connected route to the goal over a grid of
blocked cells (Dijkstra), and the point a given distance along it, used as the cost map's goal.
"""

from collections import deque
import math
from typing import Tuple

from bevnav_core.bev import BevGrid
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

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


def route_distance(blocked: np.ndarray, goal_rc: Tuple[int, int], cell_m: float) -> np.ndarray:
    """
    Shortest-route distance (metres) from every cell to goal_rc over the unblocked cells.

    blocked: (R, C) bool. 8-connected moves of cell_m or cell_m * sqrt(2). Returns (R, C)
    float: +inf where the goal cannot be reached (and on blocked cells).
    """
    rows, cols = blocked.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    src, dst, weight = [], [], []
    free = ~blocked
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):       # each neighbour pair once
        lo, hi = max(0, -dc), cols - max(0, dc)       # columns where a + (dr, dc) stays inside
        a = (slice(0, rows - dr), slice(lo, hi))
        b = (slice(dr, rows), slice(lo + dc, hi + dc))
        ok = free[a] & free[b]
        src.append(index[a][ok])
        dst.append(index[b][ok])
        weight.append(np.full(int(ok.sum()), cell_m * math.hypot(dr, dc)))
    graph = coo_matrix((np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))),
                       shape=(rows * cols, rows * cols)).tocsr()
    dist = dijkstra(graph, directed=False, indices=int(index[goal_rc]))
    dist[blocked.ravel()] = np.inf
    return dist.reshape(rows, cols)


def route_path(dist: np.ndarray, start_rc: Tuple[int, int]) -> np.ndarray:
    """
    Cells from start_rc down a route_distance() field to the goal: (N, 2) int (row, col).

    Steps to the 8-neighbour with the smallest distance; empty if start_rc cannot reach the goal.
    """
    if not np.isfinite(dist[start_rc]):
        return np.zeros((0, 2), int)
    rows, cols = dist.shape
    path = [start_rc]
    while dist[path[-1]] > 0:
        r, c = path[-1]
        best = min(((r + dr, c + dc) for dr, dc in _DIRECTIONS
                    if 0 <= r + dr < rows and 0 <= c + dc < cols), key=lambda n: dist[n])
        if dist[best] >= dist[path[-1]]:
            break
        path.append(best)
    return np.array(path, int)


def farthest_visible(path: np.ndarray, blocked: np.ndarray) -> int:
    """
    Index of the farthest path cell seen from path[0] along a straight unblocked line.

    path: (N, 2) cells (row, col), N >= 1; blocked: (R, C) bool. The line is sampled every half
    cell. Stops at the first path cell that is not visible (string pulling).
    """
    best = 0
    for k in range(1, len(path)):
        n = int(np.abs(path[k] - path[0]).max()) * 2 + 1
        line = np.rint(np.linspace(path[0], path[k], n)).astype(int)
        if blocked[line[:, 0], line[:, 1]].any():
            break
        best = k
    return best
