"""Message <-> NumPy / bevnav_core conversions; the only module that does this."""

import math

from bevnav_core.transforms import Pose2D
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def odometry_to_pose2d(msg: Odometry) -> Pose2D:
    """Planar pose (m, rad) of an Odometry message; yaw from the orientation quaternion."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return Pose2D(p.x, p.y, yaw)


def pose2d_to_transform(pose: Pose2D, stamp: Time, parent: str, child: str) -> TransformStamped:
    """Build the parent -> child transform of a planar pose (z = 0, rotation about z only)."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = pose.x
    t.transform.translation.y = pose.y
    t.transform.rotation.z = math.sin(pose.yaw / 2.0)
    t.transform.rotation.w = math.cos(pose.yaw / 2.0)
    return t
