"""
Persistent BEV map (Stage 6).

/bev/view and /bev/fused in, /bev/mask and /bev/global_map out.
Every /bev/view frame and every /bev/fused scan is added to a log-odds grid in odom at the
robot pose (TF odom <- base_footprint) of its stamp. Before a scan is added, its unseen disc
around the robot is filled where the floor beyond it continues (fill_blind_disc). At
planner.rate_hz the Galvis window is cut from the map at the latest pose and published as
/bev/mask (drivable = P(free) > memory.p_free_threshold, obstacles grown by bev.inflation_m),
together with the whole map on /bev/global_map. Before each window, the route sub-goal (E5) is
published on /bev/subgoal: the point memory.subgoal_m along the shortest route to /goal on the
map, or /goal itself when no route is found; costmap_node reads it as its goal in pipeline:=full.
/memory/reset clears the map.
"""

import time

from bevnav_core.bev import BevGrid
from bevnav_core.fusion import FusedGrid
from bevnav_core.memory import (BevMap, drivable_window, grid_points, MapParams,
                                route_subgoal_xy)
from bevnav_core.raycast import fill_blind_disc
from bevnav_ros import declare, run
from bevnav_ros.conversions import (array_to_image, image_array_to_tristate, image_to_array,
                                    map_to_occupancy_grid, occupancy_grid_to_tristate,
                                    pose_stamped_to_pose2d, transform_to_pose2d)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {  # name: (default = YAML value, description)
    'bev.size_px': ([480, 480], 'BEV rows (x) and columns (y)'),
    'bev.x_range_m': ([0.0, 5.3], 'BEV x range (m)'),
    'bev.y_range_m': ([-3.0, 3.0], 'BEV y range (m)'),
    'bev.inflation_m': (0.275, 'Obstacles grow by this in /bev/mask: robot turning radius (m)'),
    'memory.resolution_m': (0.05, 'Map cell (m)'),
    'memory.size_cells': ([1000, 1000], 'Map size: cells along x, along y'),
    'memory.origin_m': ([-25.0, -25.0], 'Odom position of the map corner (m)'),
    'memory.l_free': (-0.7, 'Log-odds added for a free observation'),
    'memory.l_occ': (0.85, 'Log-odds added for an occupied observation'),
    'memory.l_min': (-2.0, 'Lower log-odds clamp'),
    'memory.l_max': (3.5, 'Upper log-odds clamp'),
    'memory.p_free_threshold': (0.65, 'Drivable when P(free) is above this'),
    'memory.subgoal_m': (3.0, 'Sub-goal distance along the route to the goal (m)'),
    'memory.route_margin_m': (2.0, 'Route search margin around known cells, robot, goal (m)'),
    'planner.rate_hz': (1.0, 'Rate of /bev/mask and /bev/global_map (Hz)'),
}


class Memory(Node):
    """Accumulate views and scans in odom; cut the planning window from them."""

    def __init__(self) -> None:
        super().__init__('memory')
        p = declare(self, PARAMETERS)
        self._p = p
        self._grid = BevGrid(p['bev.size_px'][0], p['bev.size_px'][1],
                             tuple(p['bev.x_range_m']), tuple(p['bev.y_range_m']))
        self._view_xy = self._grid.cell_centres()
        self._map = BevMap(MapParams(p['memory.resolution_m'], tuple(p['memory.size_cells']),
                                     tuple(p['memory.origin_m']), p['memory.l_free'],
                                     p['memory.l_occ'], p['memory.l_min'], p['memory.l_max']))
        self._tf = Buffer()
        TransformListener(self._tf, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._mask_pub = self.create_publisher(Image, '/bev/mask', 10)
        self._map_pub = self.create_publisher(OccupancyGrid, '/bev/global_map', latched)
        self._subgoal_pub = self.create_publisher(PoseStamped, '/bev/subgoal', latched)
        self._goal = None
        self.create_subscription(PoseStamped, '/goal', self._on_goal, latched)
        self.create_subscription(Image, '/bev/view', self._on_view, 10)
        self.create_subscription(OccupancyGrid, '/bev/fused', self._on_fused, latched)
        self.create_service(Trigger, '/memory/reset', self._on_reset)
        self.create_timer(1.0 / p['planner.rate_hz'], self._publish)
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in p.items()))

    def _lookup(self, stamp: Time):
        """TF odom <- base_footprint at stamp, or None (WARN, frame skipped)."""
        try:
            return self._tf.lookup_transform('odom', 'base_footprint', stamp)
        except TransformException as e:
            self.get_logger().warn(f'no odom TF: {e}', throttle_duration_sec=1.0)
            return None

    def _integrate(self, stamp, x: np.ndarray, y: np.ndarray, state: np.ndarray,
                   cell_m: tuple) -> bool:
        tf = self._lookup(Time.from_msg(stamp))
        if tf is None:
            return False
        pose = transform_to_pose2d(tf)
        t0 = time.perf_counter()
        dropped = self._map.integrate(pose, *grid_points(x, y, state, cell_m,
                                                         self._p['memory.resolution_m']))
        if dropped:
            self.get_logger().warn(f'{dropped} cells outside the map dropped',
                                   throttle_duration_sec=1.0)
        self.get_logger().debug(f'update {1e3 * (time.perf_counter() - t0):.1f} ms')
        return True

    def _on_view(self, msg: Image) -> None:
        self._integrate(msg.header.stamp, *self._view_xy,
                        image_array_to_tristate(image_to_array(msg)), self._grid.cell_m)

    def _on_fused(self, msg: OccupancyGrid) -> None:
        res = msg.info.resolution
        fused = FusedGrid(res, msg.info.width * res / 2)
        x, y = fused.cell_centres()
        state = fill_blind_disc(occupancy_grid_to_tristate(msg), x, y, res)
        if self._integrate(msg.header.stamp, x, y, state, (res, res)):
            self.get_logger().info('scan added to the map')

    def _on_goal(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != 'odom':
            self.get_logger().warn(f'/goal must be in odom, got {msg.header.frame_id!r}')
            return
        self._goal = pose_stamped_to_pose2d(msg)

    def _publish_subgoal(self, pose, stamp) -> None:
        goal_xy = (self._goal.x, self._goal.y)
        sub = route_subgoal_xy(self._map, pose, goal_xy, self._p['bev.inflation_m'],
                               self._p['memory.subgoal_m'], self._p['memory.route_margin_m'])
        if sub is None:
            self.get_logger().warn('no route to the goal on the map; using the goal itself',
                                   throttle_duration_sec=5.0)
        msg = PoseStamped(header=Header(stamp=stamp, frame_id='odom'))
        msg.pose.position.x, msg.pose.position.y = sub if sub is not None else goal_xy
        msg.pose.orientation.w = 1.0
        self._subgoal_pub.publish(msg)

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response):
        self._map.reset()
        response.success, response.message = True, 'map cleared'
        return response

    def _publish(self) -> None:
        tf = self._lookup(Time())
        if tf is None:
            return
        t0 = time.perf_counter()
        pose = transform_to_pose2d(tf)
        if self._goal is not None:
            self._publish_subgoal(pose, tf.header.stamp)
        drivable = drivable_window(self._map, pose, self._grid,
                                   self._p['memory.p_free_threshold'], self._p['bev.inflation_m'])
        self._mask_pub.publish(array_to_image(np.where(drivable, 255, 0).astype(np.uint8),
                                              Header(stamp=tf.header.stamp,
                                                     frame_id='base_footprint')))
        self.get_logger().debug(f'route and window {1e3 * (time.perf_counter() - t0):.1f} ms')
        self._map_pub.publish(map_to_occupancy_grid(
            self._map.tristate(), self._p['memory.resolution_m'], self._p['memory.origin_m'],
            Header(stamp=tf.header.stamp, frame_id='odom')))


def main() -> None:
    """Run the node."""
    run(Memory)


if __name__ == '__main__':
    main()
