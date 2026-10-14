"""
Stage 3 range test: contact-cell error of /bev/view against a box at known distances.

Run with: ros2 launch bevnav_ros bev_range_check.launch.py (which passes bev.yaml). The box in
worlds/calibration/bev_ranges.sdf is moved (gz set_pose) so that its near face is
bev_range_check.ranges_m ahead of the rover's ground-truth pose. In each /bev/view taken after
the move, the nearest occupied cell of every column with |y| < eval_half_width_m is the contact
estimate. Writes bev_range.csv and bev_range.png to $BEVNAV_DATA/results/bev_range/.
"""

import csv
import math
import os
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from bevnav_core.bev import BevGrid
from matplotlib.figure import Figure
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
import yaml

WORLD, BOX = 'bev_ranges', 'range_box'
OCCUPIED = 128   # /bev/view value (design spec Section 5)


def _yaw(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class RangeCheck(Node):
    """Collect /bev/view and /gt/pose; move the box with the gz set_pose service."""

    def __init__(self) -> None:
        super().__init__('bev_range_check',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        size = self.declare_parameter('bev.size_px', [480, 480]).value
        self.grid = BevGrid(size[0], size[1],
                            tuple(self.declare_parameter('bev.x_range_m', [0.0, 5.3]).value),
                            tuple(self.declare_parameter('bev.y_range_m', [-3.0, 3.0]).value))
        self.gt = None
        self.views = []
        self.create_subscription(Odometry, '/gt/pose', lambda m: setattr(self, 'gt', m), 10)
        self.create_subscription(Image, '/bev/view', self.views.append, 10)

    def spin_until(self, done, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s   # wall clock: an evaluation tool, not navigation
        while not done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() > deadline:
                raise TimeoutError('no data from the simulation')

    def move_box(self, near_face_m: float, box_size_m: float) -> None:
        p, yaw = self.gt.pose.pose.position, _yaw(self.gt)
        d = near_face_m + box_size_m / 2
        req = (f'name: "{BOX}", position: {{x: {p.x + d * math.cos(yaw)}, '
               f'y: {p.y + d * math.sin(yaw)}, z: 0}}, '
               f'orientation: {{z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')
        subprocess.run(['gz', 'service', '-s', f'/world/{WORLD}/set_pose', '--reqtype',
                        'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
                        '--req', req], check=True, capture_output=True)


def contact_errors(view: np.ndarray, grid: BevGrid, near_face_m: float,
                   half_width_m: float) -> np.ndarray:
    """Per-column error (m) of the nearest occupied cell vs the true near face; NaN if none."""
    x, y = grid.cell_centres()
    errors = []
    for j in np.nonzero(np.abs(y[0]) < half_width_m)[0]:
        hit = np.nonzero(view[:, j] == OCCUPIED)[0]
        errors.append(x[hit.max(), j] - near_face_m if len(hit) else np.nan)  # max row = nearest
    return np.array(errors)


def _summary(rng: float, errs: np.ndarray) -> dict:
    seen = errs[~np.isnan(errs)]
    stat = (lambda f: float(f(seen)) if len(seen) else float('nan'))  # noqa: E731
    return {'range_m': rng, 'columns': len(errs), 'detected': len(seen),
            'mean_error_m': stat(np.mean), 'mean_abs_error_m': stat(lambda e: np.mean(np.abs(e))),
            'max_abs_error_m': stat(lambda e: np.max(np.abs(e)))}


def _plot(rows: list, path: Path) -> None:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    r = [row['range_m'] for row in rows]
    ax.plot(r, [100 * row['mean_error_m'] for row in rows], 'o-', label='mean (signed)')
    ax.plot(r, [100 * row['max_abs_error_m'] for row in rows], 's--', label='max |error|')
    for limit in (5, -5):
        ax.axhline(limit, color='grey', lw=0.8)
    ax.set(xlabel='box distance (m)', ylabel='contact-cell error (cm)',
           title='BEV contact-cell error vs range (GT masks)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    """Measure every range, write CSV and plot, print a summary."""
    trials = Path(get_package_share_directory('bevnav_eval')) / 'config' / 'trials.yaml'
    cfg = yaml.safe_load(trials.read_text())['bev_range_check']
    data = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data'))
    out = data / 'results' / 'bev_range'
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = RangeCheck()
    node.spin_until(lambda: node.gt is not None and node.views)
    rows = []
    for rng in cfg['ranges_m']:
        node.move_box(rng, cfg['box_size_m'])
        moved = node.get_clock().now()
        node.views.clear()

        def settled(m: Image) -> bool:
            return (Time.from_msg(m.header.stamp) - moved).nanoseconds * 1e-9 > cfg['settle_s']

        node.spin_until(lambda: sum(map(settled, node.views)) >= cfg['views_per_range'])
        views = [m for m in node.views if settled(m)][:cfg['views_per_range']]
        errs = np.concatenate([
            contact_errors(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width), node.grid,
                           rng, cfg['eval_half_width_m']) for m in views])
        rows.append(_summary(rng, errs))
        node.get_logger().info(str(rows[-1]))
    with open(out / 'bev_range.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, out / 'bev_range.png')
    node.get_logger().info(f'wrote {out}/bev_range.csv and bev_range.png')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
