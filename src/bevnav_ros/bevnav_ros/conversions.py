"""Message <-> NumPy / bevnav_core conversions; the only module that does this."""

import math

from bevnav_core.bev import FREE, OCCUPIED
from bevnav_core.camera_model import PinholeCamera
from bevnav_core.transforms import Pose2D
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

# Tri-state image values on topics (CLAUDE.md Section 5): 0 unknown, 128 occupied, 255 free.
TOPIC_UNKNOWN, TOPIC_OCCUPIED, TOPIC_FREE = 0, 128, 255
_CHANNELS = {'mono8': 1, 'rgb8': 3}


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


def image_to_array(msg: Image) -> np.ndarray:
    """mono8 -> (H, W) uint8; rgb8 -> (H, W, 3) uint8 (RGB order, as Gazebo sends it)."""
    ch = _CHANNELS[msg.encoding]
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    a = rows[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    return a[..., 0] if ch == 1 else a


def array_to_image(array: np.ndarray, header: Header) -> Image:
    """(H, W) uint8 -> mono8, (H, W, 3) uint8 -> rgb8, keeping the given header."""
    a = np.ascontiguousarray(array, np.uint8)
    msg = Image(header=header, height=a.shape[0], width=a.shape[1])
    msg.encoding = 'mono8' if a.ndim == 2 else 'rgb8'
    msg.step = a.shape[1] * (1 if a.ndim == 2 else 3)
    msg.data = a.tobytes()
    return msg


def labels_to_floor_mask(msg: Image, floor_label: int) -> np.ndarray:
    """Label image (rgb8, label in every channel) -> (H, W) uint8 mask, 255 = floor."""
    labels = image_to_array(msg)
    labels = labels[..., 0] if labels.ndim == 3 else labels
    return np.where(labels == floor_label, 255, 0).astype(np.uint8)


def tristate_to_image_array(view: np.ndarray) -> np.ndarray:
    """Core tri-state (0 unknown, 1 free, 2 occupied) -> topic values (0 / 255 / 128)."""
    out = np.full(view.shape, TOPIC_UNKNOWN, np.uint8)
    out[view == FREE] = TOPIC_FREE
    out[view == OCCUPIED] = TOPIC_OCCUPIED
    return out


def tristate_to_drivable_array(view: np.ndarray) -> np.ndarray:
    """Core tri-state -> /bev/mask values: 255 = drivable (free), 0 = anything else."""
    return np.where(view == FREE, 255, 0).astype(np.uint8)


def transform_to_matrix(t: TransformStamped) -> np.ndarray:
    """(4, 4) matrix of a TransformStamped (maps child-frame points into the parent frame)."""
    q, p = t.transform.rotation, t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [p.x, p.y, p.z]
    return m


def camera_from_msgs(info: CameraInfo, base_from_optical: TransformStamped) -> PinholeCamera:
    """Pinhole camera from CameraInfo K and the base_footprint <- camera_optical_frame TF."""
    k = info.k
    return PinholeCamera(k[0], k[4], k[2], k[5], info.width, info.height,
                         transform_to_matrix(base_from_optical))
