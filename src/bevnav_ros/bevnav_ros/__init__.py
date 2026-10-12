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
    try:
        node.destroy_node()
        rclpy.try_shutdown()
    except KeyboardInterrupt:
        pass       # a second SIGINT (launch forwards one too) while shutting down


def declare(node, spec: dict) -> dict:
    """Declare {name: (default, description)} parameters; return {name: value}."""
    from rcl_interfaces.msg import ParameterDescriptor
    return {name: node.declare_parameter(
        name, default, ParameterDescriptor(description=desc)).value
        for name, (default, desc) in spec.items()}
