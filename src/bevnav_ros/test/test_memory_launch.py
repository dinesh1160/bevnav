"""Smoke test: memory_node turns /bev/view frames into /bev/mask and /bev/global_map."""

import time
import unittest

from geometry_msgs.msg import TransformStamped
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing.actions
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from tf2_ros import StaticTransformBroadcaster

ROWS = COLS = 480


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription([
        Node(package='bevnav_ros', executable='memory_node'),
        launch_testing.actions.ReadyToTest(),
    ])


class TestMemory(unittest.TestCase):

    def test_view_reaches_mask_and_map(self) -> None:
        rclpy.init()
        node = rclpy.create_node('test_memory')
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'odom', 'base_footprint'
        tf.transform.rotation.w = 1.0
        StaticTransformBroadcaster(node).sendTransform(tf)
        masks, maps = [], []
        node.create_subscription(Image, '/bev/mask', masks.append, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(OccupancyGrid, '/bev/global_map', maps.append, latched)
        views = node.create_publisher(Image, '/bev/view', 10)
        view = np.zeros((ROWS, COLS), np.uint8)
        view[:, 140:340] = 255                                  # free floor straight ahead
        msg = Image(height=ROWS, width=COLS, encoding='mono8', step=COLS, data=view.tobytes())
        drivable = lambda m: (np.frombuffer(m.data, np.uint8) == 255).sum()  # noqa: E731
        deadline = time.time() + 20.0
        try:
            while time.time() < deadline and not (masks and drivable(masks[-1]) and maps):
                msg.header.stamp = node.get_clock().now().to_msg()
                views.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        self.assertTrue(masks and maps)
        self.assertEqual(masks[-1].header.frame_id, 'base_footprint')
        self.assertGreater(drivable(masks[-1]), 0.25 * ROWS * COLS)   # inflated corridor
        self.assertEqual(maps[-1].header.frame_id, 'odom')
        self.assertGreater((np.array(maps[-1].data) == 0).sum(), 0)
