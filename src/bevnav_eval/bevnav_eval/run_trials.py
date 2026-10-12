"""
Run closed-loop trials and record a bag per trial (CLAUDE.md Stage 4 task 7, Stage 8 task 2).

Usage: ros2 run bevnav_eval run_trials CAMPAIGN [--scenarios A B C D] [--seeds 0 19]
       [--pipeline baseline] [--seg-backend gt]

Per trial: the seeded start pose (jittered_start) and the goal in odom (= the start pose) are
computed, the goal is published latched, bringup is launched headless, and the trial is judged
live with bevnav_core.metrics until it is decided. Bags and trial.yaml go to
$BEVNAV_DATA/bags/CAMPAIGN/<scenario>_s<seed>/; trials already recorded are skipped, so a
campaign can be resumed. Outcomes for reporting come from analyze_bags.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from bevnav_core.metrics import classify_trial, OutcomeRules, RUNNING
from bevnav_core.transforms import jittered_start, Pose2D, relative_pose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from ros_gz_interfaces.msg import Contacts
import yaml

TOPICS = ['/gt/pose', '/contact', '/cmd_vel', '/pan_scan/state', '/bev/path',
          '/odometry/filtered', '/tf', '/tf_static', '/goal']
IMAGE_TOPICS = ['/bev/mask']        # recorded for the first seed of each scenario only


def load_config() -> tuple:
    """trials.yaml and the scenario YAMLs."""
    config = Path(get_package_share_directory('bevnav_eval')) / 'config'
    trials = yaml.safe_load((config / 'trials.yaml').read_text())['trials']
    scenarios = {p.stem: yaml.safe_load(p.read_text())
                 for p in (config / 'scenarios').glob('*.yaml')}
    return trials, scenarios


def outcome_rules(trials: dict, scenario: dict) -> OutcomeRules:
    """Section 7 rules with this scenario's time limit."""
    limit = (trials['time_limit_factor'] * scenario['route_length_m']
             / trials['time_limit_speed_m_s'])
    return OutcomeRules(trials['goal_tolerance_m'], limit, trials['collision_clearance_m'],
                        trials['stall_dist_m'], trials['stall_time_s'])


def _stop(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGINT the process group, then SIGKILL it if it is still alive after grace_s."""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


class TrialMonitor:
    """Publishes the latched goal and collects ground truth and contacts."""

    def __init__(self) -> None:
        self.node = rclpy.create_node('run_trials')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal', latched)
        self.node.create_subscription(Odometry, '/gt/pose', self._on_gt, 50)
        self.node.create_subscription(Contacts, '/contact', self._on_contact, 50)
        self.last_stamp = None
        self.reset()

    def reset(self) -> None:
        """Start a trial: record nothing until the simulation clock restarts."""
        self.t, self.xy, self.contacts = [], [], 0
        self.waiting = self.last_stamp is not None

    def _on_gt(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        restarted = self.last_stamp is not None and stamp < self.last_stamp
        self.last_stamp = stamp
        if self.waiting and not restarted:
            return              # still the previous trial's simulation (queued messages)
        self.waiting = False
        self.t.append(stamp)
        self.xy.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    def _on_contact(self, msg: Contacts) -> None:
        if not self.waiting:
            self.contacts += 1

    def send_goal(self, goal: Pose2D) -> None:
        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        msg.pose.position.x, msg.pose.position.y = goal.x, goal.y
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)


def run_trial(mon: TrialMonitor, key: str, seed: int, args, trials: dict, scenario: dict,
              out: Path) -> str:
    """Run one trial; return the live outcome."""
    rules = outcome_rules(trials, scenario)
    start = jittered_start(Pose2D(*scenario['start_pose']), seed, trials['start_jitter_m'],
                           trials['start_jitter_rad'])
    goal_world = np.array(scenario['goal'])
    mon.send_goal(relative_pose(start, Pose2D(goal_world[0], goal_world[1], 0.0)))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'trial.yaml').write_text(yaml.safe_dump({
        'scenario': key, 'seed': seed, 'pipeline': args.pipeline, 'seg_backend': args.seg_backend,
        'start_pose': [start.x, start.y, start.yaw], 'goal': goal_world.tolist(),
        'time_limit_s': rules.time_limit_s}, sort_keys=False))
    log = open(out / 'launch.log', 'w')
    launch = subprocess.Popen(
        ['ros2', 'launch', 'bevnav_ros', 'bringup.launch.py', 'headless:=true',
         f'scenario:={key}', f'seed:={seed}', f'pipeline:={args.pipeline}',
         f'seg_backend:={args.seg_backend}'], stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    topics = TOPICS + (IMAGE_TOPICS if seed == args.seeds[0] else [])
    bag = subprocess.Popen(['ros2', 'bag', 'record', '-o', str(out / 'bag')] + topics,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           start_new_session=True)
    mon.reset()
    outcome, wall_end = RUNNING, time.monotonic() + 3 * rules.time_limit_s + 120
    try:
        while outcome == RUNNING and time.monotonic() < wall_end and launch.poll() is None:
            rclpy.spin_once(mon.node, timeout_sec=0.2)
            if len(mon.t) > 1:
                t = np.array(mon.t) - mon.t[0]
                outcome = classify_trial(t, np.array(mon.xy), mon.contacts, goal_world, rules,
                                         finished=False)
    finally:                    # never leave a simulation running, even after an error
        _stop(bag, 10.0)
        _stop(launch, 20.0)
        log.close()
    return outcome if outcome != RUNNING else 'aborted'


def main() -> None:
    """Run every scenario x seed of the campaign."""
    ap = argparse.ArgumentParser()
    ap.add_argument('campaign')
    ap.add_argument('--scenarios', nargs='+', default=['A', 'B', 'C', 'D'])
    ap.add_argument('--seeds', nargs=2, type=int, metavar=('FIRST', 'LAST'))
    ap.add_argument('--pipeline', default='baseline')
    ap.add_argument('--seg-backend', default='gt')
    args = ap.parse_args()
    trials, scenarios = load_config()
    args.seeds = args.seeds or trials['seeds']
    data = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data'))
    root = data / 'bags' / args.campaign
    rclpy.init()
    mon = TrialMonitor()
    for key in args.scenarios:
        for seed in range(args.seeds[0], args.seeds[1] + 1):
            out = root / f'{key}_s{seed:02d}'
            if (out / 'bag').exists():
                print(f'{key} seed {seed:2d}: already recorded, skipped', flush=True)
                continue
            started = time.monotonic()
            outcome = run_trial(mon, key, seed, args, trials, scenarios[key], out)
            print(f'{key} seed {seed:2d}: {outcome:9s} ({time.monotonic() - started:.0f} s wall)',
                  flush=True)
    mon.node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
