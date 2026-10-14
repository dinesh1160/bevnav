"""
Fuse the six /pan_scan/views of a scan into /bev/fused (Stage 5).

Views are collected between SCANNING and DONE on /pan_scan/state; fusion waits (retrying on
each CameraInfo) until TF covers every view's stamp. Each view's pan angle comes
from TF base_footprint <- camera_pan_link at the view's stamp (the measured angle, never the
command). The LUT is built once in the pan frame: camera_pan_link's position with zero pan,
composed with the static camera_pan_link <- camera_optical_frame transform.
"""

from dataclasses import replace

from bevnav_core.bev import BevGrid, build_lut, warp_floor_mask
from bevnav_core.fusion import fuse_views, FusedGrid, log_odds_to_tristate
from bevnav_core.raycast import build_ray_bins, raycast
from bevnav_ros import declare, run
from bevnav_ros.conversions import (camera_from_msgs, image_to_array, transform_to_matrix,
                                    transform_to_pose2d, tristate_to_occupancy_grid)
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

PARAMETERS = {  # name: (default = YAML value, description)
    'bev.size_px': ([480, 480], 'BEV rows (x) and columns (y)'),
    'bev.x_range_m': ([0.0, 5.3], 'BEV x range (m)'),
    'bev.y_range_m': ([-3.0, 3.0], 'BEV y range (m)'),
    'fusion.ray_step_rad': (0.0087, 'Angle between rays (rad)'),
    'fusion.range_weight_r0_m': (3.0, 'Range weight exp(-r / r0) (m)'),
    'fusion.resolution_m': (0.05, 'Fused grid cell (m)'),
    'fusion.radius_m': (5.3, 'Fused grid half-size (m)'),
    'memory.l_free': (-0.7, 'Log-odds added for a free observation'),
    'memory.l_occ': (0.85, 'Log-odds added for an occupied observation'),
}


class Fusion(Node):
    """Collect a scan's views and publish the fused 360-degree grid."""

    def __init__(self) -> None:
        super().__init__('fusion')
        p = declare(self, PARAMETERS)
        self._p = p
        self._bev = BevGrid(p['bev.size_px'][0], p['bev.size_px'][1],
                            tuple(p['bev.x_range_m']), tuple(p['bev.y_range_m']))
        self._fused = FusedGrid(p['fusion.resolution_m'], p['fusion.radius_m'])
        self._lut = self._rays = self._lens = self._info = None
        self._views, self._pending = [], None
        self._tf = Buffer()
        TransformListener(self._tf, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(OccupancyGrid, '/bev/fused', latched)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/pan_scan/views', self._views.append, 10)
        self.create_subscription(String, '/pan_scan/state', self._on_state, 10)
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in p.items()))

    def _ensure_lut(self) -> bool:
        if self._lut is not None:
            return True
        try:
            pan_link = self._tf.lookup_transform('base_footprint', 'camera_pan_link', Time())
            optical = self._tf.lookup_transform('camera_pan_link', 'camera_optical_frame', Time())
        except TransformException as e:
            self.get_logger().warn(f'no camera TF yet: {e}', throttle_duration_sec=1.0)
            return False
        if self._info is None:
            return False
        at_zero_pan = TransformStamped()
        at_zero_pan.transform.translation = pan_link.transform.translation
        at_zero_pan.transform.rotation.w = 1.0
        camera = camera_from_msgs(self._info, optical)     # extrinsics relative to the pan link
        camera = replace(camera, base_from_optical=transform_to_matrix(at_zero_pan)
                         @ camera.base_from_optical)
        self._lens = (pan_link.transform.translation.x, pan_link.transform.translation.y)
        self._lut = build_lut(self._bev, camera)
        self._rays = build_ray_bins(self._bev, self._lens, self._p['fusion.ray_step_rad'])
        return True

    def _on_info(self, msg: CameraInfo) -> None:
        self._info = msg
        if self._pending is not None:     # retry until TF covers the last view's stamp
            self._try_fuse()

    def _on_state(self, msg: String) -> None:
        if msg.data == 'SCANNING':
            self._views.clear()
        elif msg.data == 'DONE' and self._views:
            self._pending = list(self._views)
            self._try_fuse()

    def _try_fuse(self) -> None:
        if not self._ensure_lut() or not all(
                self._tf.can_transform('base_footprint', 'camera_pan_link',
                                       Time.from_msg(m.header.stamp)) for m in self._pending):
            return
        masks, self._pending = self._pending, None
        self._fuse(masks)

    def _fuse(self, masks: list) -> None:
        views = []
        for m in masks:
            pan = transform_to_pose2d(self._tf.lookup_transform(
                'base_footprint', 'camera_pan_link', Time.from_msg(m.header.stamp))).yaw
            views.append((raycast(warp_floor_mask(image_to_array(m), self._lut), self._rays), pan))
        log_odds = fuse_views(views, self._fused, self._bev, self._lens, self._p['memory.l_free'],
                              self._p['memory.l_occ'], self._p['fusion.range_weight_r0_m'])
        header = masks[0].header
        header.frame_id = 'base_footprint'
        self._pub.publish(tristate_to_occupancy_grid(log_odds_to_tristate(log_odds),
                                                     self._fused.resolution_m,
                                                     self._fused.radius_m, header))
        self.get_logger().info(f'fused {len(views)} views')


def main() -> None:
    """Run the node."""
    run(Fusion)


if __name__ == '__main__':
    main()
