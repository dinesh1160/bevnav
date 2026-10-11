"""
LUT regression: the BEV lookup table for the robot.yaml camera and bev.yaml grid is unchanged.

The reference (every 16th cell) lives in data/lut_reference.npz. After an intended change
to the camera or grid, regenerate it with: python3 test_lut_regression.py
"""

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from bevnav_core.bev import BevGrid, BevLut, build_lut
from bevnav_core.camera_model import base_from_optical_matrix, PinholeCamera
import numpy as np
import yaml

REFERENCE = Path(__file__).parent / 'data' / 'lut_reference.npz'
STRIDE = 16


def robot_lut() -> BevLut:
    """LUT of the configured camera: Gazebo intrinsics f = (W / 2) / tan(HFOV / 2)."""
    sim = Path(get_package_share_directory('bevnav_sim')) / 'config' / 'robot.yaml'
    ros = Path(get_package_share_directory('bevnav_ros')) / 'config' / 'bev.yaml'
    cam = yaml.safe_load(sim.read_text())['camera']
    bev = yaml.safe_load(ros.read_text())['/**']['ros__parameters']['bev']
    w, h = cam['width_px'], cam['height_px']
    f = (w / 2) / math.tan(cam['hfov_rad'] / 2)
    camera = PinholeCamera(f, f, w / 2, h / 2, w, h,
                           base_from_optical_matrix(cam['height_m'], cam['pitch_rad']))
    grid = BevGrid(bev['size_px'][0], bev['size_px'][1],
                   tuple(bev['x_range_m']), tuple(bev['y_range_m']))
    return build_lut(grid, camera)


def _sample(lut: BevLut) -> dict:
    return {'map_u': lut.map_u[::STRIDE, ::STRIDE], 'map_v': lut.map_v[::STRIDE, ::STRIDE],
            'valid': lut.valid[::STRIDE, ::STRIDE]}


def test_lut_matches_reference() -> None:
    ref = np.load(REFERENCE)
    now = _sample(robot_lut())
    np.testing.assert_array_equal(now['valid'], ref['valid'])
    np.testing.assert_allclose(now['map_u'], ref['map_u'], atol=1e-3)
    np.testing.assert_allclose(now['map_v'], ref['map_v'], atol=1e-3)


if __name__ == '__main__':
    REFERENCE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE, **_sample(robot_lut()))
    print(f'wrote {REFERENCE}')
