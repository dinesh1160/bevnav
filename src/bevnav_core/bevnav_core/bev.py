"""
Lookup-table inverse perspective mapping onto the Galvis BEV grid (CLAUDE.md Section 5).

Grid convention: row i -> x = x_max - (i + 0.5) * dx, column j -> y = y_max - (j + 0.5) * dy,
so row 0 is the far edge and column 0 the left edge (a driver's view), in base_footprint.
"""

from dataclasses import dataclass
from typing import Tuple

from bevnav_core.camera_model import PinholeCamera
import cv2
import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2   # tri-state values used in core


@dataclass(frozen=True)
class BevGrid:
    """BEV grid geometry: rows x cols cells over x_range_m x y_range_m (metres)."""

    rows: int
    cols: int
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]

    @property
    def cell_m(self) -> Tuple[float, float]:
        """Cell size (dx along x, dy along y) in metres."""
        return ((self.x_range_m[1] - self.x_range_m[0]) / self.rows,
                (self.y_range_m[1] - self.y_range_m[0]) / self.cols)

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return x, y: (rows, cols) metres of every cell centre in base_footprint."""
        dx, dy = self.cell_m
        x = self.x_range_m[1] - (np.arange(self.rows) + 0.5) * dx
        y = self.y_range_m[1] - (np.arange(self.cols) + 0.5) * dy
        return np.meshgrid(x, y, indexing='ij')


@dataclass(frozen=True)
class BevLut:
    """Source pixel of each BEV cell: map_u, map_v (rows, cols) float32; valid (rows, cols)."""

    map_u: np.ndarray
    map_v: np.ndarray
    valid: np.ndarray


def build_lut(grid: BevGrid, camera: PinholeCamera) -> BevLut:
    """Project every cell centre on the floor (z = 0) into the image; invalid = not seen."""
    x, y = grid.cell_centres()
    u, v, visible = camera.project(np.stack([x, y, np.zeros_like(x)], axis=-1))
    map_u = np.where(visible, u, -1).astype(np.float32)
    map_v = np.where(visible, v, -1).astype(np.float32)
    return BevLut(map_u, map_v, visible)


def warp_floor_mask(mask: np.ndarray, lut: BevLut) -> np.ndarray:
    """
    Resample an image floor mask onto the BEV grid.

    mask: (H, W) uint8, 255 = floor. Returns (rows, cols) uint8 tri-state seed: FREE where a
    visible cell sees floor, OCCUPIED where it sees non-floor, UNKNOWN where not visible.
    Nearest neighbour only, so no in-between values appear.
    """
    sampled = cv2.remap(mask, lut.map_u, lut.map_v, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = np.full(sampled.shape, UNKNOWN, np.uint8)
    out[lut.valid] = np.where(sampled[lut.valid] == 255, FREE, OCCUPIED)
    return out


DEBUG_COLOURS = np.array([[40, 40, 40],      # UNKNOWN: dark grey
                          [235, 235, 235],   # FREE: near white
                          [220, 60, 40]],    # OCCUPIED: red
                         np.uint8)
DEBUG_GRID_COLOUR = np.array([70, 130, 220], np.uint8)


def grid_lines(grid: BevGrid, spacing_m: float) -> np.ndarray:
    """(rows, cols) bool: cells on a metric grid line every spacing_m (computed once)."""
    x, y = grid.cell_centres()
    dx, dy = grid.cell_m
    on_x = np.abs(x / spacing_m - np.round(x / spacing_m)) * spacing_m < dx / 2
    on_y = np.abs(y / spacing_m - np.round(y / spacing_m)) * spacing_m < dy / 2
    return on_x | on_y


def debug_image(view: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """(rows, cols) tri-state -> (rows, cols, 3) uint8 RGB, grid_lines() drawn on top."""
    img = DEBUG_COLOURS[view]
    img[lines] = DEBUG_GRID_COLOUR
    return img


def inflate(cells: np.ndarray, grid: BevGrid, radius_m: float) -> np.ndarray:
    """(rows, cols) bool: cells within radius_m (elliptical in cells, round in metres)."""
    dx, dy = grid.cell_m
    ry, rx = max(int(round(radius_m / dx)), 0), max(int(round(radius_m / dy)), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
    return cv2.dilate(cells.astype(np.uint8), kernel) > 0
