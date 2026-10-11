"""
Navigation nodes for the given pipeline and seg_backend (CLAUDE.md Section 4).

Included by bringup.launch.py (and evaluation launch files), which set use_sim_time.
Stage 3 wiring: gt_mask_node (seg_backend:=gt) and bev_node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    pipeline = LaunchConfiguration('pipeline').perform(context)
    seg_backend = LaunchConfiguration('seg_backend').perform(context)
    config = os.path.join(get_package_share_directory('bevnav_ros'), 'config')
    yaml = lambda name: os.path.join(config, f'{name}.yaml')  # noqa: E731
    if seg_backend != 'gt':
        raise RuntimeError(f'seg_backend:={seg_backend} (segmentation_node) arrives in Stage 5')
    return [
        Node(package='bevnav_ros', executable='gt_mask_node', name='gt_mask',
             parameters=[yaml('segmentation')]),
        Node(package='bevnav_ros', executable='bev_node', name='bev',
             parameters=[yaml('bev'), yaml('fusion'), yaml('planner'), yaml('scan'),
                         {'pipeline': pipeline}]),
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare pipeline and seg_backend and start the matching nodes."""
    return LaunchDescription([
        DeclareLaunchArgument('pipeline', choices=['baseline', 'fusion', 'memory', 'full'],
                              description='Navigation pipeline mode'),
        DeclareLaunchArgument('seg_backend',
                              choices=['gt', 'segformer', 'pspnet', 'fchardnet', 'onnx_generic'],
                              description='Segmentation backend'),
        OpaqueFunction(function=_nodes),
    ])
