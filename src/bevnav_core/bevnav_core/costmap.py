"""
Galvis cost map, Eq. 1-4, on the BEV grid (base_footprint, metres).

J_OA: Gaussian blur of the drivable-region contours, window border excluded (D8).
J_C:  |-m x + y - b| / sqrt(m^2 + 1), distance to the centre-line (D5: absolute value).
J_G:  Euclidean distance to the goal.
Each term is min-max normalised to [0, 1] over drivable cells (D1), then
J = w_oa J_OA + w_c J_C + w_g J_G; non-drivable cells cost +inf (D7).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from bevnav_core.bev import BevGrid
from bevnav_core.centerline import Line
import cv2
import numpy as np


@dataclass(frozen=True)
class CostParams:
    """Weights and blur from costmap.yaml."""

    w_oa: float
    w_c: float
    w_g: float
    gauss_sigma_m: float
    gauss_kernel_px: int


def obstacle_term(drivable: np.ndarray, grid: BevGrid, sigma_m: float,
                  kernel_px: int) -> np.ndarray:
    """
    (rows, cols) float32: blurred contour image of the drivable region (before D1).

    Contour pixels on the grid border are dropped: the edge of the BEV window is not an
    obstacle (D8). Field-of-view edges inside the window stay, as in the paper.
    """
    contours, _ = cv2.findContours(drivable.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    edges = np.zeros(drivable.shape, np.float32)
    cv2.drawContours(edges, contours, -1, 1.0, 1)
    edges[[0, -1], :] = 0.0
    edges[:, [0, -1]] = 0.0
    dx, dy = grid.cell_m
    return cv2.GaussianBlur(edges, (kernel_px, kernel_px),
                            sigmaX=sigma_m / dy, sigmaY=sigma_m / dx)


def centerline_term(grid: BevGrid, line: Line) -> np.ndarray:
    """(rows, cols) metres from each cell centre to the centre-line."""
    x, y = grid.cell_centres()
    return np.abs(-line.m * x + y - line.b) / np.hypot(line.m, 1.0)


def goal_term(grid: BevGrid, goal_xy: Tuple[float, float]) -> np.ndarray:
    """(rows, cols) metres from each cell centre to the goal."""
    x, y = grid.cell_centres()
    return np.hypot(x - goal_xy[0], y - goal_xy[1])


def _normalise(term: np.ndarray, drivable: np.ndarray) -> np.ndarray:
    values = term[drivable]
    if values.size == 0 or values.max() == values.min():
        return np.zeros_like(term, np.float32)
    return ((term - values.min()) / (values.max() - values.min())).astype(np.float32)


def cost_map(drivable: np.ndarray, grid: BevGrid, goal_xy: Tuple[float, float],
             line: Optional[Line], p: CostParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Total cost (rows, cols) float32 (+inf where not drivable) and the normalised terms.

    drivable: (rows, cols) bool. goal_xy: metres in base_footprint (may lie outside the grid).
    line: centre-line, or None (W_C = 0 for this frame).
    """
    terms = {'oa': _normalise(obstacle_term(drivable, grid, p.gauss_sigma_m,
                                            p.gauss_kernel_px), drivable),
             'g': _normalise(goal_term(grid, goal_xy), drivable)}
    terms['c'] = (_normalise(centerline_term(grid, line), drivable) if line is not None
                  else np.zeros_like(terms['g']))
    total = p.w_oa * terms['oa'] + p.w_c * terms['c'] + p.w_g * terms['g']
    total[~drivable] = np.inf
    return total.astype(np.float32), terms
