"""
Ground-truth pose to odom -> base_footprint TF when pose_source:=gt.

The first /gt/pose message defines odom (the trial start pose maps world -> odom,
design spec Section 7). Each TF copies the stamp of the /gt/pose message it came from.
"""

from bevnav_core.transforms import relative_pose
from bevnav_ros import run
from bevnav_ros.conversions import odometry_to_pose2d, pose2d_to_transform
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class GtTfRelay(Node):
    """Relay /gt/pose (world frame) as odom -> base_footprint."""

    def __init__(self) -> None:
        super().__init__('gt_tf_relay')
        self._origin = None
        self._broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/gt/pose', self._on_pose, 10)
        self.get_logger().info('relaying /gt/pose to odom -> base_footprint; '
                               'odom = first ground-truth pose')

    def _on_pose(self, msg: Odometry) -> None:
        pose = odometry_to_pose2d(msg)
        if self._origin is None:
            self._origin = pose
        self._broadcaster.sendTransform(pose2d_to_transform(
            relative_pose(self._origin, pose), msg.header.stamp, 'odom', 'base_footprint'))


def main() -> None:
    """Run the node."""
    run(GtTfRelay)


if __name__ == '__main__':
    main()
