"""Smoke test: gt_mask_node turns a label image into a floor mask (255 = floor)."""

import time
import unittest

from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing.actions
import numpy as np
import pytest
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription([
        Node(package='bevnav_ros', executable='gt_mask_node', parameters=[{'labels.floor': 1}]),
        launch_testing.actions.ReadyToTest(),
    ])


class TestGtMask(unittest.TestCase):

    def test_floor_label_becomes_255(self) -> None:
        rclpy.init()
        node = rclpy.create_node('test_gt_mask')
        got = []
        node.create_subscription(Image, '/seg/floor_mask', got.append, qos_profile_sensor_data)
        pub = node.create_publisher(Image, '/seg_camera/labels_map', qos_profile_sensor_data)
        labels = np.zeros((4, 6, 3), np.uint8)
        labels[2:, :] = 1           # bottom half floor, top half label 0
        labels[0, 0] = 2            # one shelf pixel
        msg = Image(height=4, width=6, encoding='rgb8', step=18, data=labels.tobytes())
        msg.header.frame_id = 'camera_optical_frame'
        deadline = time.time() + 10.0
        try:
            while time.time() < deadline and not got:
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        self.assertTrue(got)
        mask = np.frombuffer(got[0].data, np.uint8).reshape(4, 6)
        self.assertEqual(got[0].encoding, 'mono8')
        self.assertEqual(got[0].header.frame_id, 'camera_optical_frame')
        np.testing.assert_array_equal(mask, np.where(labels[..., 0] == 1, 255, 0))
