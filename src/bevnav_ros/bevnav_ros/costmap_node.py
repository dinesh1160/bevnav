"""
Galvis cost map from /bev/mask and /goal to /bev/costmap (32FC1, +inf = not drivable).

The goal (/goal, frame odom) is expressed in base_footprint with the TF at the mask's stamp;
without a goal the default goal ahead is used (D3).
"""

from bevnav_core.bev import BevGrid
from bevnav_core.centerline import fit_centerline, row_midpoints
from bevnav_core.costmap import cost_map, CostParams
from bevnav_core.transforms import relative_pose
from bevnav_ros import declare, run
from bevnav_ros.conversions import (array_to_image, image_to_array, pose_stamped_to_pose2d,
                                    transform_to_pose2d)
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {  # name: (default = YAML value, description)
    'bev.size_px': ([480, 480], 'BEV rows (x) and columns (y)'),
    'bev.x_range_m': ([0.0, 5.3], 'BEV x range (m)'),
    'bev.y_range_m': ([-3.0, 3.0], 'BEV y range (m)'),
    'costmap.w_oa': (3.0, 'Obstacle-avoidance weight'),
    'costmap.w_c': (0.5, 'Centre-line weight'),
    'costmap.w_g': (2.5, 'Goal weight'),
    'costmap.gauss_sigma_m': (0.05, 'Blur of the obstacle contours (m)'),
    'costmap.gauss_kernel_px': (151, 'Blur kernel (px, odd)'),
    'costmap.centerline_min_rows': (40, 'Rows with both edges needed for the centre-line'),
    'costmap.centerline_inlier_m': (0.10, 'RANSAC inlier distance (m)'),
    'costmap.centerline_ransac_iterations': (100, 'RANSAC iterations'),
    'costmap.default_goal': ([5.3, 0.0], 'Goal in base_footprint when no /goal (m)'),
}


class Costmap(Node):
    """Turn each /bev/mask into the Galvis cost map."""

    def __init__(self) -> None:
        super().__init__('costmap')
        p = declare(self, PARAMETERS)
        self._p = p
        self._grid = BevGrid(p['bev.size_px'][0], p['bev.size_px'][1],
                             tuple(p['bev.x_range_m']), tuple(p['bev.y_range_m']))
        self._cost = CostParams(p['costmap.w_oa'], p['costmap.w_c'], p['costmap.w_g'],
                                p['costmap.gauss_sigma_m'], p['costmap.gauss_kernel_px'])
        self._goal = None
        self._tf = Buffer()
        TransformListener(self._tf, self)
        self._pub = self.create_publisher(Image, '/bev/costmap', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseStamped, '/goal', self._on_goal, latched)
        self.create_subscription(Image, '/bev/mask', self._on_mask, 10)
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in p.items()))

    def _on_goal(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != 'odom':
            self.get_logger().warn(f'/goal must be in odom, got {msg.header.frame_id!r}',
                                   throttle_duration_sec=1.0)
            return
        self._goal = pose_stamped_to_pose2d(msg)

    def _on_mask(self, msg: Image) -> None:
        goal_xy = tuple(self._p['costmap.default_goal'])
        if self._goal is not None:
            try:
                robot = transform_to_pose2d(self._tf.lookup_transform(
                    'odom', 'base_footprint', Time.from_msg(msg.header.stamp)))
            except TransformException as e:
                self.get_logger().warn(f'no odom TF: {e}', throttle_duration_sec=1.0)
                return
            local = relative_pose(robot, self._goal)
            goal_xy = (local.x, local.y)
        drivable = image_to_array(msg) == 255
        line = fit_centerline(*row_midpoints(drivable, self._grid),
                              self._p['costmap.centerline_min_rows'],
                              self._p['costmap.centerline_inlier_m'],
                              self._p['costmap.centerline_ransac_iterations'])
        total, _ = cost_map(drivable, self._grid, goal_xy, line, self._cost)
        self._pub.publish(array_to_image(total, msg.header))


def main() -> None:
    """Run the node."""
    run(Costmap)


if __name__ == '__main__':
    main()
