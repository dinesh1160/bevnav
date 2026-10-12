"""Smoke tests: costmap_node, planner_node and controller_node each turn one input into output."""

import time
import unittest

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing.actions
from nav_msgs.msg import Path
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image
from tf2_ros import StaticTransformBroadcaster

ROWS = COLS = 480


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription([
        Node(package='bevnav_ros', executable='costmap_node'),
        Node(package='bevnav_ros', executable='planner_node'),
        Node(package='bevnav_ros', executable='controller_node'),
        launch_testing.actions.ReadyToTest(),
    ])


def _corridor() -> np.ndarray:
    mask = np.zeros((ROWS, COLS), np.uint8)
    mask[:, 200:280] = 255                      # 1 m wide corridor straight ahead
    return mask


class TestGalvisNodes(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()
        cls.node = rclpy.create_node('test_galvis_nodes')
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'odom', 'base_footprint'
        tf.transform.rotation.w = 1.0
        StaticTransformBroadcaster(cls.node).sendTransform(tf)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.node.destroy_node()
        rclpy.shutdown()

    def _exchange(self, pub_topic, pub_type, msg, sub_topic, sub_type):
        got = []
        sub = self.node.create_subscription(sub_type, sub_topic, got.append, 10)
        pub = self.node.create_publisher(pub_type, pub_topic, 10)
        deadline = time.time() + 15.0
        while time.time() < deadline and not got:
            msg.header.stamp = self.node.get_clock().now().to_msg()
            pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.2)
        self.node.destroy_subscription(sub)
        return got

    def test_costmap(self) -> None:
        mask = Image(height=ROWS, width=COLS, encoding='mono8', step=COLS,
                     data=_corridor().tobytes())
        got = self._exchange('/bev/mask', Image, mask, '/bev/costmap', Image)
        self.assertTrue(got)
        cost = np.frombuffer(got[0].data, np.float32).reshape(ROWS, COLS)
        self.assertEqual(got[0].encoding, '32FC1')
        self.assertTrue(np.isinf(cost[:, :200]).all() and np.isfinite(cost[:, 200:280]).all())

    def test_planner(self) -> None:
        cost = np.full((ROWS, COLS), np.inf, np.float32)
        cost[:, 200:280] = np.linspace(0.0, 1.0, ROWS, dtype=np.float32)[:, None]  # cheaper ahead
        msg = Image(height=ROWS, width=COLS, encoding='32FC1', step=COLS * 4, data=cost.tobytes())
        msg.header.frame_id = 'base_footprint'
        got = self._exchange('/bev/costmap', Image, msg, '/bev/path', Path)
        self.assertTrue(got)
        self.assertEqual(got[0].header.frame_id, 'odom')
        self.assertGreater(got[0].poses[-1].pose.position.x, 5.0)

    def test_controller(self) -> None:
        path = Path()
        path.header.frame_id = 'odom'
        for x in np.arange(0.0, 3.0, 0.1):
            p = PoseStamped()
            p.pose.position.x, p.pose.orientation.w = float(x), 1.0
            path.poses.append(p)
        got = self._exchange('/bev/path', Path, path, '/cmd_vel', Twist)
        moving = [t for t in got if t.linear.x > 0.4]
        deadline = time.time() + 5.0
        while time.time() < deadline and not moving:      # first ticks may precede the path
            got += self._exchange('/bev/path', Path, path, '/cmd_vel', Twist)
            moving = [t for t in got if t.linear.x > 0.4]
        self.assertTrue(moving)
        self.assertAlmostEqual(moving[-1].angular.z, 0.0, places=3)
