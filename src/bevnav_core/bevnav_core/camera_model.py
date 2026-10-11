"""Pinhole camera with extrinsics: project points in base_footprint to image pixels."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """
    Undistorted pinhole camera mounted on the robot.

    fx, fy, cx, cy: intrinsics in pixels (CameraInfo K); width, height: image size in pixels.
    base_from_optical: (4, 4) transform taking points in camera_optical_frame (z forward,
    x right, y down) to base_footprint (x forward, y left, z up), metres.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    base_from_optical: np.ndarray

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project base_footprint points to pixels.

        points: (..., 3) metres. Returns u, v: (...) float pixel coordinates (pixel centres
        at integers), and visible: (...) bool, True when the point is in front of the
        camera and inside the image.
        """
        optical_from_base = np.linalg.inv(self.base_from_optical)
        p = points @ optical_from_base[:3, :3].T + optical_from_base[:3, 3]
        z = p[..., 2]
        in_front = z > 0
        z_safe = np.where(in_front, z, 1.0)
        u = self.fx * p[..., 0] / z_safe + self.cx
        v = self.fy * p[..., 1] / z_safe + self.cy
        visible = (in_front & (u >= -0.5) & (u < self.width - 0.5)
                   & (v >= -0.5) & (v < self.height - 0.5))
        return u, v, visible


def base_from_optical_matrix(height_m: float, pitch_rad: float) -> np.ndarray:
    """
    Transform base_footprint <- camera_optical_frame for a lens on the z axis.

    height_m: lens height above the floor; pitch_rad: downward pitch (positive = looking
    down, REP 103). Used by tests and to check the TF-derived extrinsics; returns (4, 4).
    """
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    base_from_link = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # pitch about y
    link_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # z fwd, x right, y down
    m = np.eye(4)
    m[:3, :3] = base_from_link @ link_from_optical
    m[:3, 3] = [0.0, 0.0, height_m]
    return m
