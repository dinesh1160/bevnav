"""
Fuse the six views of a pan scan into one 360-degree tri-state grid (Stage 5).

Each view is a ray-cast tri-state grid on the Galvis BEV grid, expressed in the pan frame: the
frame base_footprint would have if the camera faced forward. A view taken at pan angle theta is
that grid rotated by theta about the lens's ground point. Views add in log-odds with weight
exp(-r / r0); unknown cells add nothing, so they never overwrite free or occupied.
"""

from dataclasses import dataclass
from typing import List, Tuple

from bevnav_core.bev import BevGrid, FREE, OCCUPIED, UNKNOWN
import numpy as np


@dataclass(frozen=True)
class FusedGrid:
    """Square grid centred on base_footprint: size x size cells of resolution_m (metres)."""

    resolution_m: float
    radius_m: float

    @property
    def size(self) -> int:
        """Cells per side."""
        return int(round(2 * self.radius_m / self.resolution_m))

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """x, y: (size, size) metres; row 0 = largest x, column 0 = largest y (driver's view)."""
        c = self.radius_m - (np.arange(self.size) + 0.5) * self.resolution_m
        return np.meshgrid(c, c, indexing='ij')


def view_to_fused(view: np.ndarray, pan_rad: float, fused: FusedGrid, bev: BevGrid,
                  lens_xy: Tuple[float, float]) -> np.ndarray:
    """
    Resample one view into the fused grid by forward mapping every view cell.

    view: (bev.rows, bev.cols) tri-state in the pan frame. Returns (size, size) tri-state: a fused
    cell is OCCUPIED if any view cell landing in it is, else FREE if any is, else UNKNOWN. Forward
    mapping keeps the thin contact cells that sampling the fine grid at coarse centres would miss.
    """
    xv, yv = bev.cell_centres()
    c, s = np.cos(pan_rad), np.sin(pan_rad)
    dx, dy = xv - lens_xy[0], yv - lens_xy[1]
    x, y = c * dx - s * dy + lens_xy[0], s * dx + c * dy + lens_xy[1]
    row = np.floor((fused.radius_m - x) / fused.resolution_m).astype(int)
    col = np.floor((fused.radius_m - y) / fused.resolution_m).astype(int)
    n = fused.size
    inside = (row >= 0) & (row < n) & (col >= 0) & (col < n)
    flat = row * n + col
    occupied, free = np.zeros(n * n, bool), np.zeros(n * n, bool)
    occupied[flat[inside & (view == OCCUPIED)]] = True
    free[flat[inside & (view == FREE)]] = True
    out = np.where(occupied, OCCUPIED, np.where(free, FREE, UNKNOWN)).astype(np.uint8)
    return out.reshape(n, n)


def fuse_views(views: List[Tuple[np.ndarray, float]], fused: FusedGrid, bev: BevGrid,
               lens_xy: Tuple[float, float], l_free: float, l_occ: float,
               r0_m: float) -> np.ndarray:
    """
    Sum the views in log-odds.

    views: [(tri-state (bev.rows, bev.cols) uint8, pan angle rad)]. Returns (size, size) float32
    log-odds: negative = free, positive = occupied, 0 = unknown.
    """
    x, y = fused.cell_centres()
    weight = np.exp(-np.hypot(x - lens_xy[0], y - lens_xy[1]) / r0_m)
    log_odds = np.zeros(x.shape, np.float32)
    for view, pan in views:
        state = view_to_fused(view, pan, fused, bev, lens_xy)
        log_odds += weight * np.where(state == FREE, l_free,
                                      np.where(state == OCCUPIED, l_occ, 0.0))
    return log_odds.astype(np.float32)


def log_odds_to_tristate(log_odds: np.ndarray) -> np.ndarray:
    """(H, W) log-odds -> uint8 tri-state: FREE < 0, OCCUPIED > 0, UNKNOWN == 0."""
    out = np.full(log_odds.shape, UNKNOWN, np.uint8)
    out[log_odds < 0] = FREE
    out[log_odds > 0] = OCCUPIED
    return out
