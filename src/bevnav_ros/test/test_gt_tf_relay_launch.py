"""Smoke test: gt_tf_relay_node turns /gt/pose into odom -> base_footprint on /tf."""

import math
import time
import unittest

from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing.actions
from nav_msgs.msg import Odometry
import pytest
import rclpy
from tf2_msgs.msg import TFMessage


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription([
        Node(package='bevnav_ros', executable='gt_tf_relay_node'),
        launch_testing.actions.ReadyToTest(),
    ])


def _odometry(x: float, y: float, yaw: float) -> Odometry:
    msg = Odometry()
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return msg


class TestGtTfRelay(unittest.TestCase):

    def test_second_pose_relative_to_first(self) -> None:
        rclpy.init()
        node = rclpy.create_node('test_gt_tf_relay')
        received = []
        node.create_subscription(TFMessage, '/tf', lambda m: received.extend(m.transforms), 10)
        pub = node.create_publisher(Odometry, '/gt/pose', 10)
        # Start facing +y at (2, 1); then move 1 m forward to (2, 2).
        poses = [_odometry(2.0, 1.0, math.pi / 2), _odometry(2.0, 2.0, math.pi / 2)]
        deadline = time.time() + 10.0
        try:
            while time.time() < deadline and not any(
                    t.transform.translation.x > 0.5 for t in received):
                pub.publish(poses[0] if not received else poses[1])
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        tf = next(t for t in received if t.transform.translation.x > 0.5)
        self.assertEqual((tf.header.frame_id, tf.child_frame_id), ('odom', 'base_footprint'))
        self.assertAlmostEqual(tf.transform.translation.x, 1.0, places=6)
        self.assertAlmostEqual(tf.transform.translation.y, 0.0, places=6)
