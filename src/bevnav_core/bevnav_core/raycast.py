"""
Ray-cast a BEV view into free / occupied / unknown (Stage 3).

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


@dataclass(frozen=True)
class BlindZone:
    """
    Cells hidden below the camera's view, in front of the robot (computed once from the LUT).

    mask: (rows, cols) bool, not visible but nearer than the first visible cell on their ray.
    first_cells, first_bins: flat indices and ray bins of the first visible cells of each ray.
    """

    mask: np.ndarray
    first_cells: np.ndarray
    first_bins: np.ndarray


def blind_zone(valid: np.ndarray, rays: RayBins) -> BlindZone:
    """Find the blind zone; valid: (rows, cols) bool from the LUT. FOV-outside rays excluded."""
    nearest = np.full(rays.n_bins, np.inf)
    np.minimum.at(nearest, rays.bin_index[valid], rays.range_m[valid])
    near_of_cell = nearest[rays.bin_index]
    mask = ~valid & np.isfinite(near_of_cell) & (rays.range_m < near_of_cell)
    first = valid & (rays.range_m < near_of_cell + rays.contact_band_m)
    return BlindZone(mask, np.flatnonzero(first), rays.bin_index[first])


def drivable_blind_cells(view: np.ndarray, rays: RayBins, blind: BlindZone) -> np.ndarray:
    """
    Blind cells assumed drivable for this view: (rows, cols) bool.

    A blind cell counts only when the first visible cells of its ray are free, i.e. the floor
    seen just beyond the blind zone continues into it; an obstacle there blocks the whole ray.
    """
    blocked = np.zeros(rays.n_bins, bool)
    np.logical_or.at(blocked, blind.first_bins, view.ravel()[blind.first_cells] != FREE)
    return blind.mask & ~blocked[rays.bin_index]


def fill_blind_disc(state: np.ndarray, x: np.ndarray, y: np.ndarray,
                    cell_m: float) -> np.ndarray:
    """
    Fill the unseen disc around the origin of a 360-degree grid (Stage 6).

    state: (rows, cols) tri-state, e.g. a fused scan; x, y: (rows, cols) cell centres (metres)
    around the ray origin (0, 0); cell_m: cell size. Rays are one cell wide at the nearest known
    cell, so every ray meets the ring of known cells around the disc. On each ray, the unknown
    cells nearer than its first known cell become FREE when that cell is FREE (the floor seen
    just beyond the disc continues into it), as drivable_blind_cells does for one view. Rays
    with no known cell stay unknown. Returns a new array.
    """
    known = state != UNKNOWN
    r = np.hypot(x, y)
    if not known.any():
        return state.copy()
    step = cell_m / max(r[known].min(), cell_m)
    n_bins = int(math.ceil(2 * math.pi / step))
    bins = np.floor((np.arctan2(y, x) + math.pi) / step).astype(np.int64) % n_bins
    nearest = np.full(n_bins, np.inf)
    np.minimum.at(nearest, bins[known], r[known])
    near_of_cell = nearest[bins]
    first = known & (r < near_of_cell + cell_m)
    blocked = np.zeros(n_bins, bool)
    np.logical_or.at(blocked, bins[first], state[first] != FREE)
    out = state.copy()
    out[~known & np.isfinite(near_of_cell) & (r < near_of_cell) & ~blocked[bins]] = FREE
    return out
