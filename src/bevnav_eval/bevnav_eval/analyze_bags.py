"""
Compute Section 7 trial outcomes and navigation metrics from recorded bags (Rule 13).

Usage: ros2 run bevnav_eval analyze_bags CAMPAIGN
Reads $BEVNAV_DATA/bags/CAMPAIGN/*/ (bag + trial.yaml), writes
$BEVNAV_DATA/results/CAMPAIGN/metrics.csv and prints a per-scenario summary. When a bag has
/bev/global_map, the map's false-free rate against the ground-truth grid (world -> odom through
the trial's start pose) is computed for every recorded map: first, last and largest value, and
the full series in maps/<trial>.csv.
"""

import csv
from dataclasses import replace
import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from bevnav_core.metrics import (classify_trial, clearance_map, footprint_clearance,
                                 map_vs_truth, OutcomeRules, SUCCESS, truth_under)
from bevnav_core.transforms import from_frame, Pose2D
from bevnav_eval.make_gt_grid import build_grid, load_spec
from bevnav_eval.run_trials import load_config, outcome_rules
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
import yaml


def read_bag(path: Path) -> dict:
    """Ground truth (t, xy, yaw), scanning flag per pose and chassis contact count of one bag."""
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(path), storage_id='mcap'), ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    t, xy, yaw, scanning, contacts, now_scanning = [], [], [], [], 0, False
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/pan_scan/state':
            now_scanning = deserialize_message(data, types[topic]).data == 'SCANNING'
        elif topic == '/gt/pose':
            m = deserialize_message(data, types[topic])
            q = m.pose.pose.orientation
            scanning.append(now_scanning)
            t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            xy.append((m.pose.pose.position.x, m.pose.pose.position.y))
            yaw.append(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        elif topic == '/contact':
            contacts += 1
    t = np.array(t)
    return {'t': t - t[0] if len(t) else t, 'xy': np.array(xy).reshape(-1, 2),
            'yaw': np.array(yaw), 'scanning': np.array(scanning, bool), 'contacts': contacts}


def map_series(path: Path, start: Pose2D, truth: np.ndarray, spec) -> list:
    """[(t s, map_vs_truth dict)] for every /bev/global_map with known cells in one bag."""
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(path), storage_id='mcap'), ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    out = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != '/bev/global_map':
            continue
        m = deserialize_message(data, types[topic])
        values = np.frombuffer(m.data, np.int8).reshape(m.info.height, m.info.width)
        iy, ix = np.nonzero(values >= 0)
        if len(iy) == 0:
            continue
        res, o = m.info.resolution, m.info.origin.position
        odom_xy = np.stack([o.x + (ix + 0.5) * res, o.y + (iy + 0.5) * res], axis=1)
        under = truth_under(from_frame(start, odom_xy), truth, np.array(spec.origin_m),
                            spec.resolution_m)
        state = values[iy, ix]
        out.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                    map_vs_truth(state == 0, state == 100, under)))
    return out


def trial_metrics(bag: dict, goal: np.ndarray, rules: OutcomeRules, clearance: np.ndarray) -> dict:
    """Outcome and Section 7 navigation metrics of one trial."""
    outcome = classify_trial(bag['t'], bag['xy'], bag['contacts'], goal, rules, clearance,
                             bag['scanning'])
    dist_to_goal = np.hypot(*(bag['xy'] - goal).T)
    reached = np.flatnonzero(dist_to_goal < rules.goal_tolerance_m)
    end = reached[0] if len(reached) else len(bag['t']) - 1
    steps = np.hypot(*np.diff(bag['xy'][:end + 1], axis=0).T)
    return {'outcome': outcome, 'contacts': bag['contacts'],
            'min_clearance_m': round(float(clearance.min()), 3),
            'time_to_goal_s': round(float(bag['t'][end]), 2) if outcome == SUCCESS else '',
            'distance_m': round(float(steps.sum()), 2) if outcome == SUCCESS else '',
            'final_dist_to_goal_m': round(float(dist_to_goal[-1]), 2)}


def main() -> None:
    """Analyse every trial of a campaign."""
    campaign = sys.argv[1]
    data = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data'))
    trials, scenarios = load_config()
    sim = Path(get_package_share_directory('bevnav_sim'))
    spec = load_spec(Path(get_package_share_directory('bevnav_eval')) / 'config' / 'trials.yaml')
    spec = replace(spec, resolution_m=trials['clearance_resolution_m'])   # fine grid for 0.02 m
    robot = yaml.safe_load((sim / 'config' / 'robot.yaml').read_text())
    size = np.array([robot['chassis']['size_m'][0],
                     robot['wheels']['track_m'] + robot['wheels']['width_m']])
    cmaps, truths, rows = {}, {}, []
    map_spec = load_spec(Path(get_package_share_directory('bevnav_eval')) / 'config'
                         / 'trials.yaml')                                  # 0.05 m, as the map
    for trial_dir in sorted((data / 'bags' / campaign).iterdir()):
        meta = yaml.safe_load((trial_dir / 'trial.yaml').read_text())
        key = meta['scenario']
        if key not in cmaps:
            world = sim / 'worlds' / 'scenarios' / scenarios[key]['world']
            cmaps[key] = clearance_map(build_grid(world, sim / 'models', spec) > 0,
                                       spec.resolution_m)
        try:
            bag = read_bag(trial_dir / 'bag')
        except RuntimeError as e:
            print(f'{trial_dir.name}: skipped, bag unreadable ({e})')
            continue
        clearance = footprint_clearance(bag['xy'], bag['yaw'], size, cmaps[key],
                                        np.array(spec.origin_m), spec.resolution_m,
                                        spec.resolution_m)
        row = {'trial': trial_dir.name, 'scenario': key, 'seed': meta['seed'],
               'pipeline': meta['pipeline'], 'seg_backend': meta['seg_backend']}
        row.update(trial_metrics(bag, np.array(meta['goal']), outcome_rules(trials,
                                 scenarios[key]), clearance))
        if key not in truths:
            truths[key] = build_grid(sim / 'worlds' / 'scenarios' / scenarios[key]['world'],
                                     sim / 'models', map_spec) > 0
        series = map_series(trial_dir / 'bag', Pose2D(*meta['start_pose']), truths[key],
                            map_spec)
        if series:
            ff = [m['false_free_rate'] for _, m in series]
            row.update({'map_false_free_first': round(ff[0], 4),
                        'map_false_free_last': round(ff[-1], 4),
                        'map_false_free_max': round(max(ff), 4)})
            maps_dir = data / 'results' / campaign / 'maps'
            maps_dir.mkdir(parents=True, exist_ok=True)
            with open(maps_dir / f'{trial_dir.name}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['t_s'] + list(series[0][1]))
                writer.writerows([round(t, 2)] + list(m.values()) for t, m in series)
        rows.append(row)
    out = data / 'results' / campaign
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=max((list(r) for r in rows), key=len))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{"scenario":8s} {"trials":>6s} {"success":>7s} {"collision":>9s} {"timeout":>7s} '
          f'{"stall":>5s}')
    for key in sorted({r['scenario'] for r in rows}):
        rs = [r for r in rows if r['scenario'] == key]
        count = lambda o: sum(r['outcome'] == o for r in rs)  # noqa: E731
        print(f'{key:8s} {len(rs):6d} {count("success"):7d} {count("collision"):9d} '
              f'{count("timeout"):7d} {count("stall"):5d}')
    print(f'wrote {out / "metrics.csv"}')


if __name__ == '__main__':
    main()
