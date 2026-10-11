"""
Stage 3 range test: calibration world, rover at the origin, GT-mask BEV, and bev_range_check.

Results go to $BEVNAV_DATA/results/bev_range/. Everything stops when the checker exits.
Lives in bevnav_ros (not bevnav_eval) because bevnav_ros already depends on bevnav_eval.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description() -> LaunchDescription:
    """Start the calibration simulation, the BEV nodes and the checker."""
    sim = get_package_share_directory('bevnav_sim')
    ros = get_package_share_directory('bevnav_ros')
    checker = Node(package='bevnav_eval', executable='bev_range_check', output='screen',
                   parameters=[os.path.join(ros, 'config', 'bev.yaml')])
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false'],
                              description='Run Gazebo without GUI'),
        SetParameter(name='use_sim_time', value=True),
        IncludeLaunchDescription(os.path.join(sim, 'launch', 'sim.launch.py'), launch_arguments={
            'world': 'bev_ranges', 'x': '0.0', 'y': '0.0', 'yaw': '0.0',
            'headless': LaunchConfiguration('headless'), 'seg_camera': 'true'}.items()),
        IncludeLaunchDescription(os.path.join(ros, 'launch', 'nav.launch.py'), launch_arguments={
            'pipeline': 'baseline', 'seg_backend': 'gt'}.items()),
        checker,
        RegisterEventHandler(OnProcessExit(target_action=checker,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
