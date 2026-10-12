"""
Stage 5 single-scan check.

One scenario world with the rover at a fixed pose, the scan and fusion nodes on GT masks, and
bevnav_eval scan_check. Everything stops when the checker exits.

Usage: ros2 launch bevnav_ros scan_check.launch.py case:=open   (cases in trials.yaml)
Lives in bevnav_ros (not bevnav_eval) because bevnav_ros already depends on bevnav_eval.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
import yaml


def _wire(context):
    case = LaunchConfiguration('case').perform(context)
    trials = os.path.join(get_package_share_directory('bevnav_eval'), 'config', 'trials.yaml')
    with open(trials) as f:
        spec = yaml.safe_load(f)['scan_check']['cases'][case]
    x, y, yaw = spec['pose']
    config = os.path.join(get_package_share_directory('bevnav_ros'), 'config')
    cfg = lambda name: os.path.join(config, f'{name}.yaml')  # noqa: E731
    checker = Node(package='bevnav_eval', executable='scan_check', arguments=[case],
                   output='screen',
                   parameters=[cfg('bev'), cfg('fusion'), cfg('memory')])
    return [
        IncludeLaunchDescription(
            os.path.join(get_package_share_directory('bevnav_sim'), 'launch', 'sim.launch.py'),
            launch_arguments={'world': spec['world'], 'x': str(x), 'y': str(y), 'yaw': str(yaw),
                              'headless': LaunchConfiguration('headless'),
                              'seg_camera': 'true'}.items()),
        Node(package='bevnav_ros', executable='gt_mask_node', name='gt_mask',
             parameters=[cfg('segmentation')]),
        Node(package='bevnav_ros', executable='scan_manager_node', name='scan_manager',
             parameters=[cfg('scan')]),
        Node(package='bevnav_ros', executable='fusion_node', name='fusion',
             parameters=[cfg('bev'), cfg('fusion'), cfg('memory')]),
        checker,
        RegisterEventHandler(OnProcessExit(target_action=checker,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare the case and start the check."""
    return LaunchDescription([
        DeclareLaunchArgument('case', default_value='open', choices=['open', 'aisle'],
                              description='Scan pose from trials.yaml scan_check.cases'),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false'],
                              description='Run Gazebo without GUI'),
        SetParameter(name='use_sim_time', value=True),
        OpaqueFunction(function=_wire),
    ])
