"""
Figures and tables from recorded campaigns.

Usage: ros2 run bevnav_eval make_figures fig5 CAMPAIGN
       ros2 run bevnav_eval make_figures latency CAMPAIGN
fig5:    for each scenario, one recorded /bev/mask frame (first seed's bag) -> the three Galvis
         cost terms, the total and the planned path, as in the paper's Fig. 5.
latency: cost map, planning and control time per call (median over every recorded /bev/mask),
         next to an empty column for the paper's Table II.
Outputs go to $BEVNAV_DATA/results/CAMPAIGN/. Navigation parameters are read from the installed
bevnav_ros config (a runtime lookup; a package dependency would form a cycle).
"""

import csv
import os
from pathlib import Path
import sys
import time

from ament_index_python.packages import get_package_share_directory
from bevnav_core.bev import BevGrid
from bevnav_core.centerline import fit_centerline, row_midpoints
from bevnav_core.costmap import cost_map, CostParams
from bevnav_core.planner import plan_greedy
from bevnav_core.pursuit import pursuit_command, PursuitParams
from bevnav_core.transforms import Pose2D, relative_pose
from matplotlib.figure import Figure
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
import yaml


def nav_params() -> dict:
    """Return the merged '/**' parameters of the navigation YAMLs, keyed 'section.name'."""
    config = Path(get_package_share_directory('bevnav_ros')) / 'config'
    out = {}
    for name in ('bev', 'costmap', 'planner', 'controller'):
        params = yaml.safe_load((config / f'{name}.yaml').read_text())['/**']['ros__parameters']
        for section, values in params.items():
            out.update({f'{section}.{k}': v for k, v in values.items()})
    return out


def read_masks(bag: Path) -> list:
    """[(mask (480, 480) bool, robot Pose2D in odom at that stamp)] from one bag."""
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag), storage_id='mcap'), ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    masks, poses = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ('/bev/mask', '/odometry/filtered'):
            continue
        m = deserialize_message(data, types[topic])
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if topic == '/bev/mask':
            mask = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width) == 255
            masks.append((stamp, mask))
        else:
            q = m.pose.pose.orientation
            poses.append((stamp, Pose2D(m.pose.pose.position.x, m.pose.pose.position.y,
                                        np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                                   1 - 2 * (q.y * q.y + q.z * q.z)))))
    stamps = np.array([s for s, _ in poses])
    return [(mask, poses[int(np.argmin(np.abs(stamps - s)))][1]) for s, mask in masks]


def _first_seed_bags(campaign_dir: Path) -> dict:
    """{scenario: (trial.yaml dict, bag path)} for trials that recorded /bev/mask."""
    found = {}
    for d in sorted(campaign_dir.iterdir()):
        meta = yaml.safe_load((d / 'trial.yaml').read_text())
        bag_meta = d / 'bag' / 'metadata.yaml'
        topics = bag_meta.read_text() if bag_meta.exists() else ''
        if '/bev/mask' in topics and meta['scenario'] not in found:
            found[meta['scenario']] = (meta, d / 'bag')
    return found


def _goal_in_odom(meta: dict) -> Pose2D:
    return relative_pose(Pose2D(*meta['start_pose']), Pose2D(meta['goal'][0], meta['goal'][1], 0))


def _setup(p: dict):
    grid = BevGrid(p['bev.size_px'][0], p['bev.size_px'][1], tuple(p['bev.x_range_m']),
                   tuple(p['bev.y_range_m']))
    cost = CostParams(p['costmap.w_oa'], p['costmap.w_c'], p['costmap.w_g'],
                      p['costmap.gauss_sigma_m'], p['costmap.gauss_kernel_px'])
    return grid, cost


def _cost(mask, robot, goal, grid, cost, p):
    g = relative_pose(robot, goal)
    line = fit_centerline(*row_midpoints(mask, grid), p['costmap.centerline_min_rows'],
                          p['costmap.centerline_inlier_m'],
                          p['costmap.centerline_ransac_iterations'])
    return cost_map(mask, grid, (g.x, g.y), line, cost)


def fig5(campaign_dir: Path, out: Path) -> None:
    """Write fig5_<scenario>.png: J_OA, J_C, J_G and total with the planned path."""
    p = nav_params()
    grid, cost = _setup(p)
    for key, (meta, bag) in _first_seed_bags(campaign_dir).items():
        frames = read_masks(bag)
        mask, robot = frames[len(frames) // 3]          # a frame a third of the way along
        total, terms = _cost(mask, robot, _goal_in_odom(meta), grid, cost, p)
        path = plan_greedy(total, grid, p['planner.step_m'], p['planner.oscillation_window'])
        extent = [grid.y_range_m[1], grid.y_range_m[0], grid.x_range_m[0], grid.x_range_m[1]]
        fig = Figure(figsize=(13, 3.6))
        panels = [('(a) $J_{OA}$', terms['oa']), ('(b) $J_C$', terms['c']),
                  ('(c) $J_G$', terms['g']), ('(d) total $J$', np.where(np.isfinite(total),
                                                                        total, np.nan))]
        for k, (title, img) in enumerate(panels):
            ax = fig.add_subplot(1, 4, k + 1)
            shown = np.where(mask, img, np.nan)
            ax.imshow(shown, extent=extent, cmap='viridis')
            ax.plot(path[:, 1], path[:, 0], 'r.', ms=2)
            ax.set(title=title, xlabel='y (m)', ylabel='x (m)' if k == 0 else '')
        fig.suptitle(f'Scenario {key}: cost terms on one GT-mask BEV frame, greedy path in red')
        fig.tight_layout()
        fig.savefig(out / f'fig5_{key}.png', dpi=110)
        print(f'wrote {out}/fig5_{key}.png')


def latency(campaign_dir: Path, out: Path) -> None:
    """Write latency.csv: median ms per call of cost map, planning and control."""
    p = nav_params()
    grid, cost = _setup(p)
    pursuit = PursuitParams(p['controller.lookahead_m'], p['controller.max_speed_m_s'],
                            p['controller.max_yaw_rate_rad_s'], p['controller.curvature_gain'])
    times = {'cost map': [], 'planning': [], 'control': []}
    for meta, bag in _first_seed_bags(campaign_dir).values():
        goal = _goal_in_odom(meta)
        for mask, robot in read_masks(bag):
            t0 = time.perf_counter()
            total, _ = _cost(mask, robot, goal, grid, cost, p)
            t1 = time.perf_counter()
            path = plan_greedy(total, grid, p['planner.step_m'], p['planner.oscillation_window'])
            t2 = time.perf_counter()
            pursuit_command(path, pursuit)
            t3 = time.perf_counter()
            times['cost map'].append(t1 - t0)
            times['planning'].append(t2 - t1)
            times['control'].append(t3 - t2)
    with open(out / 'latency.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'median_ms', 'p95_ms', 'calls', 'paper_table_ii_ms'])
        for stage, ts in times.items():
            ms = 1e3 * np.array(ts)
            w.writerow([stage, round(float(np.median(ms)), 2),
                        round(float(np.percentile(ms, 95)), 2), len(ms), ''])
            print(f'{stage:9s} median {np.median(ms):7.2f} ms  p95 {np.percentile(ms, 95):7.2f} ms'
                  f'  ({len(ms)} calls)')
    print(f'wrote {out}/latency.csv')


def main() -> None:
    """Dispatch fig5 or latency for a campaign."""
    command, campaign = sys.argv[1], sys.argv[2]
    data = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data'))
    out = data / 'results' / campaign
    out.mkdir(parents=True, exist_ok=True)
    {'fig5': fig5, 'latency': latency}[command](data / 'bags' / campaign, out)


if __name__ == '__main__':
    main()
