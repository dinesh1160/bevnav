"""
Ray-cast a BEV view into free / occupied / unknown (CLAUDE.md Stage 3).

Rays leave the camera's ground point every step_rad. Each cell belongs to the ray (angular
bin) through its centre. Along a ray, floor cells before the first non-floor cell are free,
that contact cell is occupied, and everything behind it is unknown: IPM smears anything
above the floor outward, and only its contact with the floor is placed correctly.
"""

from dataclasses import dataclass
import math
from typing import Tuple

from bevnav_core.bev import BevGrid, FREE, OCCUPIED, UNKNOWN
import numpy as np


@dataclass(frozen=True)
class RayBins:
    """Per-cell ray index and range: bin_index (rows, cols) int; range_m (rows, cols) float."""

    bin_index: np.ndarray
    range_m: np.ndarray
    n_bins: int
    contact_band_m: float


def build_ray_bins(grid: BevGrid, origin_xy: Tuple[float, float], step_rad: float) -> RayBins:
    """Precompute which ray each cell lies on, seen from origin_xy (m, base_footprint)."""
    x, y = grid.cell_centres()
    dx, dy = x - origin_xy[0], y - origin_xy[1]
    n_bins = int(math.ceil(2 * math.pi / step_rad))
    bins = np.floor((np.arctan2(dy, dx) + math.pi) / step_rad).astype(np.int64) % n_bins
    return RayBins(bins, np.hypot(dx, dy), n_bins, float(np.hypot(*grid.cell_m)))


def raycast(seed: np.ndarray, rays: RayBins) -> np.ndarray:
    """
    Apply the first-contact rule to a warped view.

    seed: (rows, cols) uint8 tri-state from warp_floor_mask (FREE = sees floor, OCCUPIED =
    sees non-floor, UNKNOWN = not visible). Returns (rows, cols) uint8 tri-state.
    """
    hit = seed == OCCUPIED
    contact = np.full(rays.n_bins, np.inf)
    np.minimum.at(contact, rays.bin_index[hit], rays.range_m[hit])
    contact_of_cell = contact[rays.bin_index]
    out = np.full(seed.shape, UNKNOWN, np.uint8)
    out[(seed == FREE) & (rays.range_m < contact_of_cell)] = FREE
    out[hit & (rays.range_m < contact_of_cell + rays.contact_band_m)] = OCCUPIED
    return out
