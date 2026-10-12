"""
Six-heading pan scan (CLAUDE.md Stage 5).

On /scan_manager/start_scan, pan the camera through scan.headings steps, take one
/seg/floor_mask per heading after the pan settled, publish it on /pan_scan/views, and report
IDLE / SCANNING / DONE on /pan_scan/state.
"""

from bevnav_core.scan import PanScan
from bevnav_ros import declare, run
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

PARAMETERS = {  # name: (default = YAML value, description)
    'scan.headings': (6, 'Views per scan'),
    'scan.step_rad': (1.0472, 'Pan step between views (rad)'),
    'scan.settle_tol_rad': (0.0087, 'Pan is settled within this of the target (rad)'),
    'scan.settle_time_s': (0.2, 'Wait after settling before taking a frame (s)'),
    'scan.timeout_s': (15.0, 'Abort a scan that takes longer (s)'),
}


class ScanManager(Node):
    """Run pan scans on request."""

    def __init__(self) -> None:
        super().__init__('scan_manager')
        p = declare(self, PARAMETERS)
        self._scan = PanScan(p['scan.headings'], p['scan.step_rad'], p['scan.settle_tol_rad'],
                             p['scan.settle_time_s'], p['scan.timeout_s'])
        self._joint = None
        self._pan_pub = self.create_publisher(Float64, '/camera_pan/cmd', 10)
        self._view_pub = self.create_publisher(Image, '/pan_scan/views', 10)
        self._state_pub = self.create_publisher(String, '/pan_scan/state', 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/seg/floor_mask', self._on_mask, qos_profile_sensor_data)
        self.create_service(Trigger, '/scan_manager/start_scan', self._on_start)
        self._set_state('IDLE')
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in p.items()))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_state(self, state: str) -> None:
        self._state_pub.publish(String(data=state))
        self.get_logger().info(f'scan state {state}')

    def _pan(self, target: float) -> None:
        self._pan_pub.publish(Float64(data=target))

    def _on_start(self, request: Trigger.Request, response: Trigger.Response):
        if self._scan.active or self._joint is None:
            response.success = False
            response.message = 'scan running' if self._scan.active else 'no /joint_states yet'
            return response
        self._pan(self._scan.start(self._now(), self._joint))
        self._set_state('SCANNING')
        response.success, response.message = True, 'scan started'
        return response

    def _on_joints(self, msg: JointState) -> None:
        if 'camera_pan_joint' not in msg.name:
            return
        self._joint = msg.position[msg.name.index('camera_pan_joint')]
        if self._scan.active:
            self._scan.on_joint(self._now(), self._joint)
            if self._scan.timed_out(self._now()):
                self.get_logger().warn(f'scan timed out after {self._scan.k} views')
                self._pan(self._scan.abort())
                self._set_state('DONE')
                self._set_state('IDLE')

    def _on_mask(self, msg: Image) -> None:
        if not self._scan.take(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9):
            return
        self._view_pub.publish(msg)
        self._pan(self._scan.advance())
        if not self._scan.active:
            self._set_state('DONE')
            self._set_state('IDLE')


def main() -> None:
    """Run the node."""
    run(ScanManager)


if __name__ == '__main__':
    main()
