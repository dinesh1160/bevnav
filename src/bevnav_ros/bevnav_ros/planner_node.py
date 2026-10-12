"""
Galvis greedy planner: /bev/costmap -> /bev/path in odom (D6).

The path is planned in base_footprint at the cost map's stamp and expressed in odom with the
TF at that stamp, so the controller can follow it while the robot moves. In pipeline:=full it
also asks /scan_manager/start_scan for a scan when ScanTrigger says so (start, every
scan.interval_m, blocked); no request while a scan is running.
"""

from bevnav_core.bev import BevGrid
from bevnav_core.planner import plan_greedy
from bevnav_core.scan import ScanTrigger
from bevnav_core.transforms import from_frame
from bevnav_ros import declare, run
from bevnav_ros.conversions import array_to_path, image_to_array, transform_to_pose2d
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {  # name: (default = YAML value, description)
    'bev.size_px': ([480, 480], 'BEV rows (x) and columns (y)'),
    'bev.x_range_m': ([0.0, 5.3], 'BEV x range (m)'),
    'bev.y_range_m': ([-3.0, 3.0], 'BEV y range (m)'),
    'planner.step_m': (0.10, 'Step between path points (m)'),
    'planner.oscillation_window': (10, 'Revisiting a cell of the last N steps stops the path'),
    'scan.interval_m': (2.0, 'Scan after this distance driven (m)'),
    'scan.stall_dist_m': (0.10, 'Blocked: moved less than this ... (m)'),
    'scan.stall_time_s': (5.0, '... in this time (s)'),
    'pipeline': ('baseline', 'Pipeline mode; scans are requested only in full'),
}


class Planner(Node):
    """Plan on each cost map and publish the path in odom."""

    def __init__(self) -> None:
        super().__init__('planner')
        p = declare(self, PARAMETERS)
        self._p = p
        self._grid = BevGrid(p['bev.size_px'][0], p['bev.size_px'][1],
                             tuple(p['bev.x_range_m']), tuple(p['bev.y_range_m']))
        self._tf = Buffer()
        TransformListener(self._tf, self)
        self._pub = self.create_publisher(Path, '/bev/path', 10)
        self.create_subscription(Image, '/bev/costmap', self._on_costmap, 10)
        self._trigger = ScanTrigger(p['scan.interval_m'], p['scan.stall_dist_m'],
                                    p['scan.stall_time_s'])
        self._scan_state, self._request = 'IDLE', None
        if p['pipeline'] == 'full':
            self._start_scan = self.create_client(Trigger, '/scan_manager/start_scan')
            self.create_subscription(String, '/pan_scan/state',
                                     lambda m: setattr(self, '_scan_state', m.data), 10)
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in p.items()))

    def _on_costmap(self, msg: Image) -> None:
        try:
            robot = transform_to_pose2d(self._tf.lookup_transform(
                'odom', 'base_footprint', Time.from_msg(msg.header.stamp)))
        except TransformException as e:
            self.get_logger().warn(f'no odom TF: {e}', throttle_duration_sec=1.0)
            return
        path = plan_greedy(image_to_array(msg), self._grid, self._p['planner.step_m'],
                           self._p['planner.oscillation_window'])
        header = msg.header
        header.frame_id = 'odom'
        self._pub.publish(array_to_path(from_frame(robot, path), header))
        if self._p['pipeline'] == 'full':
            self._maybe_scan(Time.from_msg(header.stamp).nanoseconds * 1e-9, robot, len(path))

    def _maybe_scan(self, now_s: float, robot, path_points: int) -> None:
        if self._request is not None or self._scan_state == 'SCANNING':
            return
        reason = self._trigger.update(now_s, (robot.x, robot.y), path_points)
        if reason is None or not self._start_scan.service_is_ready():
            return
        self._request = self._start_scan.call_async(Trigger.Request())
        self._request.add_done_callback(
            lambda f: self._on_scan_reply(f, reason, now_s, (robot.x, robot.y)))

    def _on_scan_reply(self, future, reason: str, now_s: float, xy: tuple) -> None:
        self._request = None
        if future.result() is not None and future.result().success:
            self._trigger.scanned(now_s, xy)
            self.get_logger().info(f'scan requested ({reason})')


def main() -> None:
    """Run the node."""
    run(Planner)


if __name__ == '__main__':
    main()
