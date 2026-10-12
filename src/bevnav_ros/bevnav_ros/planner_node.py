"""
Galvis greedy planner: /bev/costmap -> /bev/path in odom (D6).

The path is planned in base_footprint at the cost map's stamp and expressed in odom with the
TF at that stamp, so the controller can follow it while the robot moves.
"""

from bevnav_core.bev import BevGrid
from bevnav_core.planner import plan_greedy
from bevnav_core.transforms import from_frame
from bevnav_ros import declare, run
from bevnav_ros.conversions import array_to_path, image_to_array, transform_to_pose2d
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {  # name: (default = YAML value, description)
    'bev.size_px': ([480, 480], 'BEV rows (x) and columns (y)'),
    'bev.x_range_m': ([0.0, 5.3], 'BEV x range (m)'),
    'bev.y_range_m': ([-3.0, 3.0], 'BEV y range (m)'),
    'planner.step_m': (0.10, 'Step between path points (m)'),
    'planner.oscillation_window': (10, 'Revisiting a cell of the last N steps stops the path'),
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


def main() -> None:
    """Run the node."""
    run(Planner)


if __name__ == '__main__':
    main()
