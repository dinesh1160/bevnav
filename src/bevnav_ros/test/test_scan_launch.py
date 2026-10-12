"""Smoke test: scan_manager_node runs a 6-view scan and fusion_node publishes /bev/fused."""

import math
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
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

W, H, F, PITCH = 640, 480, 457.85, 0.2618


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription([
        Node(package='bevnav_ros', executable='scan_manager_node'),
        Node(package='bevnav_ros', executable='fusion_node'),
        launch_testing.actions.ReadyToTest(),
    ])


def _tf(parent: str, child: str, z: float, roll: float, yaw: float) -> TransformStamped:
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.z = z
    t.transform.rotation.w = math.cos(roll / 2) * math.cos(yaw / 2)
    t.transform.rotation.x = math.sin(roll / 2) * math.cos(yaw / 2)
    t.transform.rotation.y = math.sin(roll / 2) * math.sin(yaw / 2)
    t.transform.rotation.z = math.cos(roll / 2) * math.sin(yaw / 2)
    return t


class TestScan(unittest.TestCase):

    def test_scan_produces_fused_grid(self) -> None:
        rclpy.init()
        node = rclpy.create_node('test_scan')
        state = {'pan': 0.0, 'states': [], 'views': 0, 'fused': None}
        node.create_subscription(Float64, '/camera_pan/cmd',
                                 lambda m: state.update(pan=m.data), 10)   # instant servo
        node.create_subscription(String, '/pan_scan/state',
                                 lambda m: state['states'].append(m.data), 10)
        node.create_subscription(Image, '/pan_scan/views',
                                 lambda m: state.update(views=state['views'] + 1), 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(OccupancyGrid, '/bev/fused',
                                 lambda m: state.update(fused=m), latched)
        joints = node.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        masks = node.create_publisher(Image, '/seg/floor_mask', qos_profile_sensor_data)
        info = node.create_publisher(CameraInfo, '/camera/camera_info', qos_profile_sensor_data)
        tf = TransformBroadcaster(node)
        # camera_pan_link -> camera_optical_frame: pitched 15 deg down, z forward (Section 5)
        StaticTransformBroadcaster(node).sendTransform(
            _tf('camera_pan_link', 'camera_optical_frame', 0.0, -(math.pi / 2 + PITCH),
                -math.pi / 2))
        start = node.create_client(Trigger, '/scan_manager/start_scan')
        floor = Image(height=H, width=W, encoding='mono8', step=W, data=bytes([255]) * (W * H))
        cam = CameraInfo(width=W, height=H, k=[F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0])
        requested = None
        deadline = time.time() + 40.0
        try:
            while time.time() < deadline and state['fused'] is None:
                stamp = node.get_clock().now().to_msg()
                pan_tf = _tf('base_footprint', 'camera_pan_link', 0.45, 0.0, state['pan'])
                pan_tf.header.stamp = stamp
                tf.sendTransform(pan_tf)
                joints.publish(JointState(name=['camera_pan_joint'], position=[state['pan']]))
                floor.header.stamp = cam.header.stamp = stamp
                masks.publish(floor)
                info.publish(cam)
                fusion_ready = node.count_subscribers('/pan_scan/views') > 1   # test + fusion
                if requested is None and start.service_is_ready() and fusion_ready:
                    requested = start.call_async(Trigger.Request())
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        self.assertEqual(state['views'], 6)
        self.assertIn('DONE', state['states'])
        self.assertIsNotNone(state['fused'])
        cells = np.array(state['fused'].data)
        self.assertEqual(state['fused'].header.frame_id, 'base_footprint')
        self.assertGreater((cells == 0).sum(), 0.5 * cells.size)   # free floor all around
