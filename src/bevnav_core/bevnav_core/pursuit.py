"""
Pure-pursuit follower: path in the current base_footprint frame -> (v, omega).

Target: the first path point at least lookahead_m away, searched from the path point nearest
the robot (the last point if none is that far); stop if it is not ahead.
kappa = 2 y / L^2, v = v_max / (1 + g |kappa|), omega = v kappa, |omega| <= max_yaw_rate_rad_s.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PursuitParams:
    """Values from controller.yaml (metres, m/s, rad/s)."""

    lookahead_m: float
    max_speed_m_s: float
    max_yaw_rate_rad_s: float
    curvature_gain: float


def pursuit_command(path: np.ndarray, p: PursuitParams) -> Tuple[float, float]:
    """path: (N, 2) metres in base_footprint. Returns (v m/s, omega rad/s); (0, 0) if no path."""
    if len(path) == 0:
        return 0.0, 0.0
    dist = np.hypot(path[:, 0], path[:, 1])
    start = int(np.argmin(dist))
    ahead = np.flatnonzero(dist[start:] >= p.lookahead_m)
    tx, ty = path[start + ahead[0]] if len(ahead) else path[-1]
    reach = np.hypot(tx, ty)
    if tx <= 0.0:          # target at or behind the robot: end of the path reached, stop
        return 0.0, 0.0
    kappa = 2.0 * ty / reach ** 2
    v = p.max_speed_m_s / (1.0 + p.curvature_gain * abs(kappa))
    omega = float(np.clip(v * kappa, -p.max_yaw_rate_rad_s, p.max_yaw_rate_rad_s))
    return float(v), omega
