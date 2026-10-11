"""
Floor mask to tri-state BEV view (/bev/view, /bev/debug) and, in pipeline:=baseline, /bev/mask.

The lookup table is built once, from the first CameraInfo and the base_footprint <-
camera_optical_frame TF taken while the camera is at pan 0. Frames taken with the camera
panned away from 0 are skipped (rotated views are fused in Stage 6).
"""

import time

from bevnav_core.bev import BevGrid, build_lut, debug_image, grid_lines, warp_floor_mask
from bevnav_core.raycast import build_ray_bins, raycast
from bevnav_ros import run
from bevnav_ros.conversions import (array_to_image, camera_from_msgs, image_to_array,
                                    tristate_to_drivable_array, tristate_to_image_array)
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {  # name: (default = YAML value, description)
    'bev.size_px': ([480, 480], 'BEV rows (x) and columns (y)'),
    'bev.x_range_m': ([0.0, 5.3], 'BEV x range ahead of base_footprint (m)'),
    'bev.y_range_m': ([-3.0, 3.0], 'BEV y range, left positive (m)'),
    'bev.debug_grid_m': (0.5, 'Metric grid spacing on /bev/debug (m)'),
    'fusion.ray_step_rad': (0.0087, 'Angle between rays (rad)'),
    'planner.rate_hz': (1.0, 'Rate of /bev/mask in pipeline:=baseline (Hz)'),
    'scan.settle_tol_rad': (0.0087, 'Frames with |pan| above this are skipped (rad)'),
    'pipeline': ('baseline', 'Pipeline mode; /bev/mask is published only in baseline'),
}


class Bev(Node):
    """Turn each floor mask into a ray-cast BEV view."""

    def __init__(self) -> None:
        super().__init__('bev')
        p = {name: self.declare_parameter(name, default, ParameterDescriptor(
             description=desc)).value for name, (default, desc) in PARAMETERS.items()}
        self._grid = BevGrid(p['bev.size_px'][0], p['bev.size_px'][1],
                             tuple(p['bev.x_range_m']), tuple(p['bev.y_range_m']))
        self._p = p
        self._lines = grid_lines(self._grid, p['bev.debug_grid_m'])
        self._rays = None
        self._lut = None
        self._info = None
        self._pan = None
        self._last_mask_stamp = None
        self._tf = Buffer()
        TransformListener(self._tf, self)
        self._view_pub = self.create_publisher(Image, '/bev/view', 10)
        self._debug_pub = self.create_publisher(Image, '/bev/debug', 10)
        self._mask_pub = self.create_publisher(Image, '/bev/mask', 10)
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 lambda m: setattr(self, '_info', m), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._on_joints,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/seg/floor_mask', self._on_mask, qos_profile_sensor_data)
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in p.items()))

    def _on_joints(self, msg: JointState) -> None:
        if 'camera_pan_joint' in msg.name:
            self._pan = msg.position[msg.name.index('camera_pan_joint')]

    def _ensure_lut(self, stamp: Time) -> bool:
        if self._lut is not None:
            return True
        if self._info is None:
            return False
        try:
            tf = self._tf.lookup_transform('base_footprint', 'camera_optical_frame', stamp)
        except TransformException as e:
            self.get_logger().warn(f'no camera TF yet: {e}', throttle_duration_sec=1.0)
            return False
        camera = camera_from_msgs(self._info, tf)
        self._lut = build_lut(self._grid, camera)
        lens = tf.transform.translation
        self._rays = build_ray_bins(self._grid, (lens.x, lens.y), self._p['fusion.ray_step_rad'])
        self.get_logger().info(f'LUT built: {int(self._lut.valid.sum())} visible cells, '
                               f'lens at ({lens.x:.3f}, {lens.y:.3f}, {lens.z:.3f}) m')
        return True

    def _on_mask(self, msg: Image) -> None:
        if self._pan is None or abs(self._pan) > self._p['scan.settle_tol_rad']:
            self.get_logger().debug('camera not at pan 0; frame skipped')
            return
        stamp = Time.from_msg(msg.header.stamp)
        if not self._ensure_lut(stamp):
            return
        t0 = time.perf_counter()
        view = raycast(warp_floor_mask(image_to_array(msg), self._lut), self._rays)
        header = msg.header
        header.frame_id = 'base_footprint'
        self._view_pub.publish(array_to_image(tristate_to_image_array(view), header))
        if self._debug_pub.get_subscription_count() > 0:
            self._debug_pub.publish(array_to_image(debug_image(view, self._lines), header))
        if self._p['pipeline'] == 'baseline' and (
                self._last_mask_stamp is None
                or (stamp - self._last_mask_stamp).nanoseconds * 1e-9
                >= 1.0 / self._p['planner.rate_hz']):
            self._last_mask_stamp = stamp
            self._mask_pub.publish(array_to_image(tristate_to_drivable_array(view), header))
        self.get_logger().debug(f'frame {1e3 * (time.perf_counter() - t0):.1f} ms')


def main() -> None:
    """Run the node."""
    run(Bev)


if __name__ == '__main__':
    main()
