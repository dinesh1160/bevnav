"""BEVNav ROS 2 nodes."""

import rclpy
from rclpy.executors import ExternalShutdownException


def run(node_class) -> None:
    """Spin one node until shutdown; errors raised while shutting down are not crashes."""
    rclpy.init()
    node = node_class()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise  # a real error, not a callback racing the shutdown
    node.destroy_node()
    rclpy.try_shutdown()
