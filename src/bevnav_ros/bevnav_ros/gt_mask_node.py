"""Ground-truth label image to /seg/floor_mask; runs only when seg_backend:=gt."""

from bevnav_ros import run
from bevnav_ros.conversions import array_to_image, labels_to_floor_mask
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class GtMask(Node):
    """Map the simulator label image to a floor mask (mono8, 255 = floor)."""

    def __init__(self) -> None:
        super().__init__('gt_mask')
        self._floor = self.declare_parameter(
            'labels.floor', 1, ParameterDescriptor(description='Label value of the floor')).value
        self._pub = self.create_publisher(Image, '/seg/floor_mask', qos_profile_sensor_data)
        self.create_subscription(Image, '/seg_camera/labels_map', self._on_labels,
                                 qos_profile_sensor_data)
        self.get_logger().info(f'labels.floor={self._floor}')

    def _on_labels(self, msg: Image) -> None:
        self._pub.publish(array_to_image(labels_to_floor_mask(msg, self._floor), msg.header))


def main() -> None:
    """Run the node."""
    run(GtMask)


if __name__ == '__main__':
    main()
