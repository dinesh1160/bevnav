"""
Evaluation metrics as defined in CLAUDE.md Section 7, from ground-truth data.

Used both live (to end a trial) and on recorded bags (the reported results), so a trial is
judged the same way in both places.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

COLLISION, TIMEOUT, STALL, SUCCESS, RUNNING = 'collision', 'timeout', 'stall', 'success', 'running'


@dataclass(frozen=True)
class OutcomeRules:
    """Section 7 thresholds (metres, seconds) from trials.yaml."""

    goal_tolerance_m: float
    time_limit_s: float
    collision_clearance_m: float
    stall_dist_m: float
    stall_time_s: float


def _stalled(t: np.ndarray, xy: np.ndarray, scanning: np.ndarray, rules: OutcomeRules) -> bool:
    """Any stall_time_s window, not scanning, in which the robot stays within stall_dist_m."""
    end = np.searchsorted(t, t + rules.stall_time_s)
    for i in np.flatnonzero(end < len(t)):
        window = slice(i, end[i] + 1)
        if scanning[window].any():
            continue
        if np.hypot(*(xy[window] - xy[i]).T).max() < rules.stall_dist_m:
            return True
    return False


def classify_trial(t: np.ndarray, xy: np.ndarray, n_contacts: int, goal_xy: np.ndarray,
                   rules: OutcomeRules, clearance: Optional[np.ndarray] = None,
                   scanning: Optional[np.ndarray] = None, finished: bool = True) -> str:
    """
    One outcome per trial, checked in the Section 7 order.

    t: (N,) s since the trial start, ascending; xy: (N, 2) ground-truth position (m);
    n_contacts: chassis contact messages; goal_xy: (2,) m, same frame as xy; clearance: (N,)
    m to the nearest obstacle or None; scanning: (N,) bool or None. With finished=False
    (live), returns RUNNING while no outcome is decided yet.
    """
    if n_contacts > 0 or (clearance is not None
                          and (clearance < rules.collision_clearance_m).any()):
        return COLLISION
    reached = np.hypot(*(xy - goal_xy).T) < rules.goal_tolerance_m
    success = bool((reached & (t <= rules.time_limit_s)).any())
    if not success and len(t) and t[-1] >= rules.time_limit_s:
        return TIMEOUT
    scanning = np.zeros(len(t), bool) if scanning is None else scanning
    if not reached.any() and _stalled(t, xy, scanning, rules):
        return STALL
    if success:
        return SUCCESS
    return TIMEOUT if finished else RUNNING


def clearance_map(occupied: np.ndarray, resolution_m: float) -> np.ndarray:
    """(H, W) metres from each free cell to the nearest occupied cell's edge (0 on obstacles)."""
    dist = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * resolution_m
    return np.maximum(dist - resolution_m / 2, 0.0)


def footprint_clearance(xy: np.ndarray, yaw: np.ndarray, size_m: np.ndarray, cmap: np.ndarray,
                        origin_m: np.ndarray, resolution_m: float, spacing_m: float) -> np.ndarray:
    """
    (N,) metres from the robot's rectangular footprint to the nearest obstacle.

    xy: (N, 2), yaw: (N,) ground-truth poses; size_m: (2,) footprint length, width; cmap:
    clearance_map of the ground-truth grid (row 0 = largest y) with lower-left corner origin_m.
    The footprint outline is sampled every spacing_m.
    """
    hl, hw = np.asarray(size_m) / 2
    sides = [np.linspace(-hl, hl, int(np.ceil(2 * hl / spacing_m)) + 1),
             np.linspace(-hw, hw, int(np.ceil(2 * hw / spacing_m)) + 1)]
    long_sides = [np.stack([sides[0], np.full_like(sides[0], s * hw)], 1) for s in (-1, 1)]
    short_sides = [np.stack([np.full_like(sides[1], s * hl), sides[1]], 1) for s in (-1, 1)]
    outline = np.concatenate(long_sides + short_sides)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    px = xy[:, :1] + c * outline[:, 0] - s * outline[:, 1]
    py = xy[:, 1:] + s * outline[:, 0] + c * outline[:, 1]
    col = np.clip(((px - origin_m[0]) / resolution_m).astype(int), 0, cmap.shape[1] - 1)
    row = np.clip(cmap.shape[0] - 1 - ((py - origin_m[1]) / resolution_m).astype(int), 0,
                  cmap.shape[0] - 1)
    return cmap[row, col].min(axis=1)
