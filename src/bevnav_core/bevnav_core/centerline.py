"""
Aisle centre-line from drivable-edge midpoints and a RANSAC line fit (deviation D2).

Rows are scanned from the robot outward, starting at the first row where the robot's column
is drivable. In each row the drivable run containing the tracked column is taken; if it has a
non-drivable cell on both sides (a left and a right edge inside the grid), its midpoint is kept
and becomes the tracked column for the next row. Losing the run ends the scan.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from bevnav_core.bev import BevGrid
import numpy as np


@dataclass(frozen=True)
class Line:
    """Centre-line y = m * x + b in base_footprint (metres)."""

    m: float
    b: float


def row_midpoints(drivable: np.ndarray, grid: BevGrid) -> Tuple[np.ndarray, np.ndarray]:
    """drivable: (rows, cols) bool. Returns x, y (N,) metres of the midpoints found."""
    x, y = grid.cell_centres()
    track = int(np.argmin(np.abs(y[0])))           # column nearest y = 0
    xs, ys = [], []
    started = False
    for i in range(grid.rows - 1, -1, -1):          # near (bottom) to far (top)
        row = drivable[i]
        if not row[track]:
            if started:
                break                               # the corridor ahead is lost
            continue                                # not reached it yet (e.g. a narrow apex)
        started = True
        blocked = np.flatnonzero(~row)
        left, right = blocked[blocked < track], blocked[blocked > track]
        if len(left) and len(right):
            first, last = left[-1] + 1, right[0] - 1
            xs.append(x[i, 0])
            ys.append((y[0, first] + y[0, last]) / 2)
            track = (first + last) // 2
    return np.array(xs), np.array(ys)


def fit_centerline(xs: np.ndarray, ys: np.ndarray, min_rows: int, inlier_m: float,
                   iterations: int, seed: int = 0) -> Optional[Line]:
    """RANSAC fit of y = m x + b; None when fewer than min_rows midpoints (then W_C = 0)."""
    if len(xs) < min_rows:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        i, j = rng.choice(len(xs), 2, replace=False)
        if xs[i] == xs[j]:
            continue
        m = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - m * xs[i]
        inliers = np.abs(ys - (m * xs + b)) / np.hypot(1.0, m) < inlier_m
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 2:
        return None
    m, b = np.polyfit(xs[best], ys[best], 1)
    return Line(float(m), float(b))
