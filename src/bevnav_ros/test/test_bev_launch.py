"""Smoke test: bev_node turns a floor mask into /bev/view and, in baseline, /bev/mask."""

import math
import time
import unittest

from geometry_msgs.msg import TransformStamped
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing.actions
import numpy as np
import pytest
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import StaticTransformBroadcaster

W, H, F, HEIGHT, PITCH = 640, 480, 457.85, 0.45, 0.2618


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription([
        Node(package='bevnav_ros', executable='bev_node', parameters=[{'pipeline': 'baseline'}]),
        launch_testing.actions.ReadyToTest(),
    ])


def _camera_tf() -> TransformStamped:
    # base_footprint <- camera_optical_frame: q = qz(-90 deg) * qx(-105 deg) at (0, 0, 0.45);
    # matches tf2_echo of the real robot up to sign: xyzw (0.561, -0.561, 0.430, -0.430).
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = 'base_footprint', 'camera_optical_frame'
    t.transform.translation.z = HEIGHT
    r, y = -(math.pi / 2 + PITCH), -math.pi / 2
    t.transform.rotation.w = math.cos(r / 2) * math.cos(y / 2)
    t.transform.rotation.x = math.sin(r / 2) * math.cos(y / 2)
    t.transform.rotation.y = math.sin(r / 2) * math.sin(y / 2)
    t.transform.rotation.z = math.cos(r / 2) * math.sin(y / 2)
    return t


class TestBev(unittest.TestCase):

    def test_all_floor_mask_gives_free_view(self) -> None:
        rclpy.init()
        node = rclpy.create_node('test_bev')
        views, masks = [], []
        node.create_subscription(Image, '/bev/view', views.append, 10)
        node.create_subscription(Image, '/bev/mask', masks.append, 10)
        StaticTransformBroadcaster(node).sendTransform(_camera_tf())
        info_pub = node.create_publisher(CameraInfo, '/camera/camera_info',
                                         qos_profile_sensor_data)
        js_pub = node.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        mask_pub = node.create_publisher(Image, '/seg/floor_mask', qos_profile_sensor_data)
        info = CameraInfo(width=W, height=H, k=[F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0])
        mask = Image(height=H, width=W, encoding='mono8', step=W, data=bytes([255]) * (W * H))
        deadline = time.time() + 15.0
        try:
            while time.time() < deadline and not (views and masks):
                info_pub.publish(info)
                js_pub.publish(JointState(name=['camera_pan_joint'], position=[0.0]))
                mask_pub.publish(mask)
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        self.assertTrue(views and masks)
        view = np.frombuffer(views[0].data, np.uint8).reshape(480, 480)
        self.assertEqual(views[0].header.frame_id, 'base_footprint')
        self.assertEqual(view[240, 240], 255)    # 2.65 m straight ahead: free
        self.assertEqual(view[479, 240], 0)      # 0.006 m ahead, inside the blind zone: unknown
        self.assertEqual(set(np.unique(view)) - {0, 255}, set())  # all floor: nothing occupied
