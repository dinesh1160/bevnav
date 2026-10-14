"""Message <-> NumPy / bevnav_core conversions; the only module that does this."""

import math

from bevnav_core.bev import FREE, OCCUPIED, UNKNOWN
from bevnav_core.camera_model import PinholeCamera
from bevnav_core.transforms import Pose2D
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

# Tri-state image values on topics (design spec Section 5): 0 unknown, 128 occupied, 255 free.
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
    """mono8 -> (H, W) uint8; rgb8 -> (H, W, 3) uint8 (RGB order); 32FC1 -> (H, W) float32."""
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    ch = _CHANNELS[msg.encoding]
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    a = rows[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    return a[..., 0] if ch == 1 else a


def array_to_image(array: np.ndarray, header: Header) -> Image:
    """(H, W) uint8 -> mono8, (H, W, 3) uint8 -> rgb8, (H, W) float32 -> 32FC1; keeps header."""
    if array.dtype == np.float32:
        a, encoding, bytes_per_px = np.ascontiguousarray(array), '32FC1', 4
    else:
        a = np.ascontiguousarray(array, np.uint8)
        encoding, bytes_per_px = ('mono8', 1) if a.ndim == 2 else ('rgb8', 3)
    msg = Image(header=header, height=a.shape[0], width=a.shape[1], encoding=encoding)
    msg.step = a.shape[1] * bytes_per_px
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


def image_array_to_tristate(values: np.ndarray) -> np.ndarray:
    """Topic values (0 unknown / 128 occupied / 255 free) -> core tri-state (0 / 2 / 1)."""
    out = np.full(values.shape, UNKNOWN, np.uint8)
    out[values == TOPIC_FREE] = FREE
    out[values == TOPIC_OCCUPIED] = OCCUPIED
    return out


def tristate_to_drivable_array(view: np.ndarray, also_drivable: np.ndarray,
                               blocked: np.ndarray) -> np.ndarray:
    """/bev/mask values: 255 = (free or also_drivable) and not blocked (bool masks), else 0."""
    return np.where(((view == FREE) | also_drivable) & ~blocked, 255, 0).astype(np.uint8)


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


def transform_to_pose2d(t: TransformStamped) -> Pose2D:
    """Planar pose of the child frame in the parent frame (yaw from the quaternion)."""
    q, p = t.transform.rotation, t.transform.translation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return Pose2D(p.x, p.y, yaw)


def pose_stamped_to_pose2d(msg: PoseStamped) -> Pose2D:
    """Planar pose (m, rad) of a PoseStamped."""
    p, q = msg.pose.position, msg.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return Pose2D(p.x, p.y, yaw)


def array_to_path(points: np.ndarray, header: Header) -> Path:
    """(N, 2) metres -> nav_msgs/Path in header.frame_id; every pose shares the header stamp."""
    msg = Path(header=header)
    for x, y in points:
        pose = PoseStamped(header=header)
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


def path_to_array(msg: Path) -> np.ndarray:
    """nav_msgs/Path -> (N, 2) metres in msg.header.frame_id."""
    return np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses]).reshape(-1, 2)


def velocity_to_twist(v: float, omega: float) -> Twist:
    """Forward speed (m/s) and yaw rate (rad/s) -> Twist."""
    t = Twist()
    t.linear.x, t.angular.z = float(v), float(omega)
    return t


def tristate_to_occupancy_grid(view: np.ndarray, resolution_m: float, radius_m: float,
                               header: Header) -> OccupancyGrid:
    """
    Centred square tri-state grid -> nav_msgs/OccupancyGrid (-1 unknown, 0 free, 100 occupied).

    view: (n, n), row 0 = largest x, column 0 = largest y, centred on header.frame_id's origin.
    OccupancyGrid rows run along +y and columns along +x from the lower-left origin.
    """
    n = view.shape[0]
    values = np.full(view.shape, -1, np.int8)
    values[view == FREE] = 0
    values[view == OCCUPIED] = 100
    msg = OccupancyGrid(header=header)
    msg.info.resolution = float(resolution_m)
    msg.info.width = msg.info.height = n
    msg.info.origin.position.x = msg.info.origin.position.y = -float(radius_m)
    msg.info.origin.orientation.w = 1.0
    msg.data = values[::-1, ::-1].T.ravel().tolist()
    return msg


def occupancy_grid_to_tristate(msg: OccupancyGrid) -> np.ndarray:
    """Inverse of tristate_to_occupancy_grid: (n, n) tri-state, row 0 = largest x."""
    n = msg.info.width
    values = np.array(msg.data, np.int8).reshape(n, n).T[::-1, ::-1]
    out = np.full((n, n), UNKNOWN, np.uint8)
    out[values == 0] = FREE
    out[values == 100] = OCCUPIED
    return out


def map_to_occupancy_grid(state: np.ndarray, resolution_m: float, origin_m: tuple,
                          header: Header) -> OccupancyGrid:
    """
    Convert the persistent map to nav_msgs/OccupancyGrid (-1 unknown, 0 free, 100 occupied).

    state: (rows along y, cols along x) tri-state, already in OccupancyGrid layout (row 0 at
    origin_m[1]), as bevnav_core.memory.BevMap.tristate() returns it.
    """
    values = np.full(state.shape, -1, np.int8)
    values[state == FREE] = 0
    values[state == OCCUPIED] = 100
    msg = OccupancyGrid(header=header)
    msg.info.resolution = float(resolution_m)
    msg.info.height, msg.info.width = state.shape
    msg.info.origin.position.x, msg.info.origin.position.y = map(float, origin_m)
    msg.info.origin.orientation.w = 1.0
    msg.data = values.ravel().tolist()
    return msg
