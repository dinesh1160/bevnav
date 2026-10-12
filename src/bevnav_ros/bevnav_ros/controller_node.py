"""
Pure-pursuit controller: /bev/path (odom) -> /cmd_vel at controller.rate_hz.

Each tick re-expresses the stored odom path in the current base_footprint (D6). Publishes zero
velocity when there is no path, the path is older than path_timeout_s, or a scan is running.
"""

from bevnav_core.pursuit import pursuit_command, PursuitParams
from bevnav_core.transforms import to_frame
from bevnav_ros import declare, run
from bevnav_ros.conversions import path_to_array, transform_to_pose2d, velocity_to_twist
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {  # name: (default = YAML value, description)
    'controller.rate_hz': (20.0, 'Control rate (Hz)'),
    'controller.lookahead_m': (0.75, 'Look-ahead distance (m)'),
    'controller.max_speed_m_s': (0.45, 'Maximum forward speed (m/s)'),
    'controller.max_yaw_rate_rad_s': (1.0, 'Maximum yaw rate (rad/s)'),
    'controller.curvature_gain': (2.0, 'v = v_max / (1 + gain * |kappa|)'),
    'controller.path_timeout_s': (2.0, 'Stop if the newest path is older than this (s)'),
}


class Controller(Node):
    """Follow the latest path with pure pursuit."""

    def __init__(self) -> None:
        super().__init__('controller')
        p = declare(self, PARAMETERS)
        self._pursuit = PursuitParams(p['controller.lookahead_m'], p['controller.max_speed_m_s'],
                                      p['controller.max_yaw_rate_rad_s'],
                                      p['controller.curvature_gain'])
        self._timeout_s = p['controller.path_timeout_s']
        self._path = None
        self._scanning = False
        self._tf = Buffer()
        TransformListener(self._tf, self)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/bev/path', lambda m: setattr(self, '_path', m), 10)
        self.create_subscription(String, '/pan_scan/state', self._on_state, 10)
        self.create_timer(1.0 / p['controller.rate_hz'], self._tick)
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in p.items()))

    def _on_state(self, msg: String) -> None:
        self._scanning = msg.data == 'SCANNING'

    def _tick(self) -> None:
        v, omega = 0.0, 0.0
        fresh = self._path is not None and (
            self.get_clock().now() - Time.from_msg(self._path.header.stamp)
        ).nanoseconds * 1e-9 <= self._timeout_s
        if fresh and not self._scanning:
            try:
                robot = transform_to_pose2d(self._tf.lookup_transform(
                    'odom', 'base_footprint', Time()))
                v, omega = pursuit_command(to_frame(robot, path_to_array(self._path)),
                                           self._pursuit)
            except TransformException as e:
                self.get_logger().warn(f'no odom TF: {e}', throttle_duration_sec=1.0)
        self._pub.publish(velocity_to_twist(v, omega))


def main() -> None:
    """Run the node."""
    run(Controller)


if __name__ == '__main__':
    main()
