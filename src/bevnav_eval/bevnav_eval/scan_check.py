"""
Stage 5 single-scan check: one pan scan at a fixed pose, measured against the ground truth.

Run with: ros2 launch bevnav_ros scan_check.launch.py case:=open   (or case:=aisle)
which also passes bev.yaml, fusion.yaml and memory.yaml. Requests a scan, then computes on the
fused grid: known-area fraction within known_radius_m (blind disc excluded), alignment error of
the occupied cells, false-free rate, and the same false-free rate for the forward view alone;
the mean of the six single views; plus the scan time.
Writes $BEVNAV_DATA/results/scan_check/<case>.csv and <case>.png.
"""

import csv
import math
import os
from pathlib import Path
import sys
import time

from ament_index_python.packages import get_package_share_directory
from bevnav_core.bev import BevGrid, build_lut, FREE, OCCUPIED, UNKNOWN, warp_floor_mask
from bevnav_core.camera_model import base_from_optical_matrix, PinholeCamera
from bevnav_core.fusion import FusedGrid, view_to_fused
from bevnav_core.metrics import clearance_map
from bevnav_core.raycast import build_ray_bins, raycast
from bevnav_eval.make_gt_grid import build_grid, load_spec
import cv2
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
import yaml


class ScanCheck(Node):
    """Trigger one scan and collect what is needed to judge it."""

    def __init__(self) -> None:
        super().__init__('scan_check', parameter_overrides=[Parameter('use_sim_time', value=True)])
        size = self.declare_parameter('bev.size_px', [480, 480]).value
        self.bev = BevGrid(size[0], size[1], tuple(self.declare_parameter('bev.x_range_m',
                                                                          [0.0, 5.3]).value),
                           tuple(self.declare_parameter('bev.y_range_m', [-3.0, 3.0]).value))
        self.fused_grid = FusedGrid(self.declare_parameter('fusion.resolution_m', 0.05).value,
                                    self.declare_parameter('fusion.radius_m', 5.3).value)
        self.ray_step = self.declare_parameter('fusion.ray_step_rad', 0.0087).value
        self.gt = self.fused = self.done_at = None
        self.views = []
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, '/gt/pose', lambda m: setattr(self, 'gt', m), 10)
        self.create_subscription(OccupancyGrid, '/bev/fused', lambda m: setattr(self, 'fused', m),
                                 latched)
        self.create_subscription(Image, '/pan_scan/views', self.views.append, 10)
        self.joints = []       # (stamp s, pan rad): the measured angle of each view
        self.create_subscription(JointState, '/joint_states', self._on_joints,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/pan_scan/state', self._on_state, 10)
        self.start = self.create_client(Trigger, '/scan_manager/start_scan')

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_joints(self, msg: JointState) -> None:
        if 'camera_pan_joint' in msg.name:
            self.joints.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                msg.position[msg.name.index('camera_pan_joint')]))

    def pan_at(self, stamp: float) -> float:
        """Measured pan angle (rad) nearest the given stamp."""
        return min(self.joints, key=lambda j: abs(j[0] - stamp))[1]

    def _on_state(self, msg: String) -> None:
        if msg.data == 'DONE':
            self.done_at = self.now()

    def spin_until(self, done, deadline: float) -> None:
        while not done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() > deadline:
                raise TimeoutError('scan check timed out')


def occupancy_to_tristate(msg: OccupancyGrid) -> np.ndarray:
    """Convert an OccupancyGrid to (n, n) tri-state, row 0 = largest x, column 0 = largest y."""
    n = msg.info.width
    values = np.array(msg.data, np.int8).reshape(n, n).T[::-1, ::-1]
    out = np.full((n, n), UNKNOWN, np.uint8)
    out[values == 0] = FREE
    out[values == 100] = OCCUPIED
    return out


def robot_camera(sim_share: Path) -> PinholeCamera:
    """Build the robot.yaml camera (Gazebo intrinsics), as in the LUT regression test."""
    cam = yaml.safe_load((sim_share / 'config' / 'robot.yaml').read_text())['camera']
    w, h = cam['width_px'], cam['height_px']
    f = (w / 2) / math.tan(cam['hfov_rad'] / 2)
    return PinholeCamera(f, f, w / 2, h / 2, w, h,
                         base_from_optical_matrix(cam['height_m'], cam['pitch_rad']))


def gt_under(fused: FusedGrid, pose: tuple, gt_occupied: np.ndarray, spec) -> np.ndarray:
    """(size, size) bool: ground-truth occupancy under each fused cell (base_footprint at pose)."""
    x, y = fused.cell_centres()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    wx, wy = pose[0] + c * x - s * y, pose[1] + s * x + c * y
    col = np.clip(((wx - spec.origin_m[0]) / spec.resolution_m).astype(int), 0,
                  gt_occupied.shape[1] - 1)
    row = np.clip(gt_occupied.shape[0] - 1 - ((wy - spec.origin_m[1]) / spec.resolution_m)
                  .astype(int), 0, gt_occupied.shape[0] - 1)
    return gt_occupied[row, col]


def scan_metrics(state: np.ndarray, fused: FusedGrid, pose: tuple, gt_occupied: np.ndarray,
                 spec, known_radius_m: float, blind_radius_m: float) -> dict:
    """Section 7 map metrics of one fused grid (base_footprint at pose in the world frame)."""
    x, y = fused.cell_centres()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    wx, wy = pose[0] + c * x - s * y, pose[1] + s * x + c * y
    col = np.clip(((wx - spec.origin_m[0]) / spec.resolution_m).astype(int), 0,
                  gt_occupied.shape[1] - 1)
    row = np.clip(gt_occupied.shape[0] - 1 - ((wy - spec.origin_m[1]) / spec.resolution_m)
                  .astype(int), 0, gt_occupied.shape[0] - 1)
    truth = gt_occupied[row, col]
    dist_to_obstacle = clearance_map(gt_occupied, spec.resolution_m)[row, col]
    r = np.hypot(x, y)
    ring = (r > blind_radius_m) & (r <= known_radius_m)
    free, occ = state == FREE, state == OCCUPIED
    return {'known_fraction': float((state[ring] != UNKNOWN).mean()),
            'alignment_error_m': float(dist_to_obstacle[occ].mean()) if occ.any() else math.nan,
            'false_free_rate': float((free & truth).sum() / max(free.sum(), 1)),
            'free_cells': int(free.sum()), 'occupied_cells': int(occ.sum())}


def main() -> None:
    """Run one scan and write its metrics."""
    case = sys.argv[1]
    eval_share = Path(get_package_share_directory('bevnav_eval'))
    sim_share = Path(get_package_share_directory('bevnav_sim'))
    trials = yaml.safe_load((eval_share / 'config' / 'trials.yaml').read_text())
    cfg = trials['scan_check']
    world = cfg['cases'][case]['world']
    rclpy.init()
    node = ScanCheck()
    deadline = time.monotonic() + cfg['timeout_s']
    node.spin_until(lambda: node.gt is not None and node.start.service_is_ready(), deadline)
    while True:     # scan_manager refuses until it has seen /joint_states
        t0 = node.now()
        future = node.start.call_async(Trigger.Request())
        node.spin_until(lambda: future.done(), deadline)
        if future.result().success:
            break
        node.spin_until(lambda: node.now() > t0 + 1.0, deadline)
    node.spin_until(lambda: node.done_at is not None and node.fused is not None
                    and len(node.views) > 0, deadline)
    scan_time = node.done_at - t0
    q, p = node.gt.pose.pose.orientation, node.gt.pose.pose.position
    pose = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
    spec = load_spec(eval_share / 'config' / 'trials.yaml')
    gt_occupied = build_grid(sim_share / 'worlds' / 'scenarios' / f'{world}.sdf',
                             sim_share / 'models', spec) > 0
    fused_state = occupancy_to_tristate(node.fused)
    rows = {'fused (6 views)': scan_metrics(fused_state, node.fused_grid, pose, gt_occupied, spec,
                                            cfg['known_radius_m'], cfg['blind_radius_m'])}
    lut = build_lut(node.bev, robot_camera(sim_share))
    rays = build_ray_bins(node.bev, (0.0, 0.0), node.ray_step)
    singles, single_grids = [], []      # each view alone, at its measured pan angle
    for m in node.views:
        mask = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
        pan = node.pan_at(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        view = view_to_fused(raycast(warp_floor_mask(mask, lut), rays), pan,
                             node.fused_grid, node.bev, (0.0, 0.0))
        singles.append(scan_metrics(view, node.fused_grid, pose, gt_occupied, spec,
                                    cfg['known_radius_m'], cfg['blind_radius_m']))
        single_grids.append(view)
    forward = single_grids[0]
    rows['forward view only'] = singles[0]
    rows['single views, mean'] = {k: float(np.nanmean([v[k] for v in singles]))
                                  for k in singles[0]}
    truth = gt_under(node.fused_grid, pose, gt_occupied, spec)
    single_ff = np.any([(g == FREE) & truth for g in single_grids], axis=0)
    added = int(((fused_state == FREE) & truth & ~single_ff).sum())
    data = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data'))
    out = data / 'results' / 'scan_check'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{case}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['grid', 'scan_time_s'] + list(rows['fused (6 views)']))
        for name, m in rows.items():
            w.writerow([name, round(scan_time, 2)] + [round(v, 4) for v in m.values()])
    np.savez_compressed(out / f'{case}.npz', fused=fused_state, forward=forward, pose=pose)
    colours = np.array([[40, 40, 40], [235, 235, 235], [40, 60, 220]], np.uint8)  # BGR
    img = np.hstack([cv2.resize(colours[g], (424, 424), interpolation=cv2.INTER_NEAREST)
                     for g in (forward, fused_state)])
    cv2.imwrite(str(out / f'{case}.png'), img)
    for name, m in rows.items():
        print(f'{case:6s} {name:18s} scan {scan_time:5.2f} s  ' +
              '  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}'
                        for k, v in m.items()), flush=True)
    print(f'{case:6s} false-free cells added by fusion (not false-free in any single view): '
          f'{added}', flush=True)
    print(f'wrote {out}/{case}.csv and {case}.png', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
