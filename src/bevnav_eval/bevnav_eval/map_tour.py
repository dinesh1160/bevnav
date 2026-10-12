"""
Stage 7 whole-warehouse map: drive the tour, then score the final map against the ground truth.

Run with: ros2 launch bevnav_ros map_tour.launch.py pose_source:=ekf   (or gt)
Re-score a saved map: ros2 run bevnav_eval map_tour score ekf
Each tour.yaml waypoint (world frame) is expressed in odom through the start pose and sent as
/goal; the next one follows when the ground-truth pose is within trials.goal_tolerance_m or after
leg_timeout_s. Then the last /bev/global_map is compared with the ground-truth grid (Section 7):
coverage (ground-truth free cells mapped free), false-free and false-occupied rates. Writes
$BEVNAV_DATA/results/map_tour/<pose_source>.csv, .png (map over the ground truth) and .npz;
the launch file records a bag of the run in $BEVNAV_DATA/bags/map_tour/.
"""

import csv
import os
from pathlib import Path
import sys
import time

from ament_index_python.packages import get_package_share_directory
from bevnav_core.metrics import map_coverage, map_vs_truth, truth_under
from bevnav_core.transforms import from_frame, Pose2D, relative_pose, to_frame
from bevnav_eval.make_gt_grid import build_grid, load_spec
import cv2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from ros_gz_interfaces.msg import Contacts
import yaml


class MapTour(Node):
    """Send the tour's goals and keep the ground truth and the latest map."""

    def __init__(self) -> None:
        super().__init__('map_tour', parameter_overrides=[Parameter('use_sim_time', value=True)])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal', latched)
        self.gt = self.map = None
        self.contacts = 0
        self.create_subscription(Odometry, '/gt/pose', lambda m: setattr(self, 'gt', m), 10)
        self.create_subscription(OccupancyGrid, '/bev/global_map',
                                 lambda m: setattr(self, 'map', m), latched)
        self.create_subscription(Contacts, '/contact', self._on_contact, 10)

    def _on_contact(self, msg: Contacts) -> None:
        self.contacts += 1

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def xy(self) -> np.ndarray:
        p = self.gt.pose.pose.position
        return np.array([p.x, p.y])

    def send_goal(self, goal: Pose2D) -> None:
        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        msg.pose.position.x, msg.pose.position.y = goal.x, goal.y
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def spin_until(self, done, wall_s: float) -> bool:
        deadline = time.monotonic() + wall_s
        while not done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > deadline:
                return False
        return True


def score_map(values: np.ndarray, res: float, origin: tuple, start: Pose2D, truth: np.ndarray,
              spec) -> tuple:
    """
    Section 7 metrics of a map in odom against the ground truth; plus a colour overlay.

    values: (rows along y, cols along x) int8 OccupancyGrid data (-1 / 0 / 100) with cell res
    and lower-left corner origin (odom metres). Coverage counts the warehouse floor only: the
    ground-truth free cells connected to the start through free cells (not the rim outside the
    walls or closed pockets).
    """
    iy, ix = np.nonzero(values >= 0)
    world = from_frame(start, np.stack([origin[0] + (ix + 0.5) * res,
                                        origin[1] + (iy + 0.5) * res], 1))
    state = values[iy, ix]
    under = truth_under(world, truth, np.array(spec.origin_m), spec.resolution_m)
    result = map_vs_truth(state == 0, state == 100, under)
    rows, cols = truth.shape                                  # truth row 0 = largest y
    r, c = np.mgrid[0:rows, 0:cols]
    centres = np.stack([spec.origin_m[0] + (c.ravel() + 0.5) * spec.resolution_m,
                        spec.origin_m[1] + (rows - r.ravel() - 0.5) * spec.resolution_m], 1)
    odom = to_frame(start, centres)
    mx = np.floor((odom[:, 0] - origin[0]) / res).astype(int)
    my = np.floor((odom[:, 1] - origin[1]) / res).astype(int)
    inside = (mx >= 0) & (mx < values.shape[1]) & (my >= 0) & (my < values.shape[0])
    mapped = np.full(len(centres), -1, np.int8)
    mapped[inside] = values[my[inside], mx[inside]]
    _, regions = cv2.connectedComponents((~truth).astype(np.uint8), connectivity=8)
    start_cell = (rows - 1 - int((start.y - spec.origin_m[1]) / spec.resolution_m),
                  int((start.x - spec.origin_m[0]) / spec.resolution_m))
    floor = (regions == regions[start_cell]).ravel()
    result['coverage'] = map_coverage(mapped == 0, floor)
    img = np.where(truth, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    colour = img.reshape(-1, 3)
    colour[(mapped == 0) & ~truth.ravel()] = (120, 200, 120)          # BGR: mapped free
    colour[(mapped == 100) & truth.ravel()] = (60, 60, 200)           # mapped occupied
    colour[(mapped == 100) & ~truth.ravel()] = (60, 160, 255)         # false occupied
    colour[(mapped == 0) & truth.ravel()] = (255, 0, 255)             # false free
    return result, img


def write_results(out: Path, pose_source: str, row: dict, values: np.ndarray, res: float,
                  origin: tuple, start: Pose2D, eval_share: Path, sim_share: Path,
                  world: str) -> dict:
    """Score the map and write <pose_source>.csv, .png and .npz to out; return the csv row."""
    spec = load_spec(eval_share / 'config' / 'trials.yaml')
    truth = build_grid(sim_share / 'worlds' / f'{world}.sdf', sim_share / 'models', spec) > 0
    result, img = score_map(values, res, origin, start, truth, spec)
    row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{pose_source}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    cv2.imwrite(str(out / f'{pose_source}.png'),
                cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    np.savez_compressed(out / f'{pose_source}.npz', map=values, resolution_m=res,
                        origin_m=origin, start_pose=[start.x, start.y, start.yaw], row=str(row))
    return row


def main() -> None:
    """Drive the tour and write the map metrics; 'map_tour score <pose_source>' re-scores."""
    eval_share = Path(get_package_share_directory('bevnav_eval'))
    sim_share = Path(get_package_share_directory('bevnav_sim'))
    tour = yaml.safe_load((eval_share / 'config' / 'tour.yaml').read_text())
    trials = yaml.safe_load((eval_share / 'config' / 'trials.yaml').read_text())['trials']
    start = Pose2D(*tour['start_pose'])
    out = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data')) / 'results' / 'map_tour'
    if sys.argv[1] == 'score':                 # re-score a saved map (e.g. after a fix)
        pose_source = sys.argv[2]
        saved = np.load(out / f'{pose_source}.npz')
        with open(out / f'{pose_source}.csv') as f:
            row = {k: v for k, v in next(csv.DictReader(f)).items()
                   if k in ('pose_source', 'waypoints_reached', 'waypoints', 'tour_time_s',
                            'contacts')}
        row = write_results(out, pose_source, row, saved['map'], float(saved['resolution_m']),
                            tuple(saved['origin_m']), start, eval_share, sim_share,
                            tour['world'])
        print(' '.join(f'{k} {v}' for k, v in row.items()))
        return
    pose_source = sys.argv[1]
    rclpy.init()
    node = MapTour()
    log = node.get_logger()
    node.spin_until(lambda: node.gt is not None, 120.0)
    t0, legs = node.now(), []
    for k, wp in enumerate(tour['waypoints']):
        node.send_goal(relative_pose(start, Pose2D(wp[0], wp[1], 0.0)))
        leg_t0 = node.now()
        reached = node.spin_until(
            lambda: np.hypot(*(node.xy() - wp)) < trials['goal_tolerance_m']
            or node.now() - leg_t0 > tour['leg_timeout_s'], 4 * tour['leg_timeout_s'])
        reached = reached and np.hypot(*(node.xy() - wp)) < trials['goal_tolerance_m']
        legs.append((k, wp, reached, node.now() - leg_t0))
        log.info(f'waypoint {k} {wp}: {"reached" if reached else "timed out"} '
                 f'after {node.now() - leg_t0:.1f} s')
    tour_s = node.now() - t0
    end = node.now()                       # take a map published after the tour ended
    node.spin_until(lambda: node.map is not None
                    and node.map.header.stamp.sec + node.map.header.stamp.nanosec * 1e-9 > end,
                    30.0)
    row = {'pose_source': pose_source, 'waypoints_reached': sum(r for _, _, r, _ in legs),
           'waypoints': len(legs), 'tour_time_s': round(tour_s, 1), 'contacts': node.contacts}
    info = node.map.info
    values = np.frombuffer(node.map.data, np.int8).reshape(info.height, info.width)
    row = write_results(out, pose_source, row, values, info.resolution,
                        (info.origin.position.x, info.origin.position.y), start, eval_share,
                        sim_share, tour['world'])
    log.info(' '.join(f'{k} {v}' for k, v in row.items()))
    log.info(f'wrote {out}/{pose_source}.csv, .png and .npz')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
