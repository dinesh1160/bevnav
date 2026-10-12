"""
Stage 7 whole-warehouse tour.

The warehouse world with the rover at the tour start (bevnav_eval/config/tour.yaml), the pose
source, the full navigation pipeline on GT masks, a bag recorder, and bevnav_eval map_tour,
which drives the tour and scores the map. Everything stops when map_tour exits.

Usage: ros2 launch bevnav_ros map_tour.launch.py pose_source:=ekf   (or gt; headless:=false
       to watch in Gazebo and RViz)
Lives in bevnav_ros (not bevnav_eval) because bevnav_ros already depends on bevnav_eval.
"""

import os
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
import yaml

BAG_TOPICS = ['/gt/pose', '/contact', '/cmd_vel', '/pan_scan/state', '/bev/path',
              '/bev/global_map', '/odometry/filtered', '/tf', '/tf_static', '/goal']


def _wire(context):
    pose_source = LaunchConfiguration('pose_source').perform(context)
    headless = LaunchConfiguration('headless').perform(context)
    eval_config = os.path.join(get_package_share_directory('bevnav_eval'), 'config')
    with open(os.path.join(eval_config, 'tour.yaml')) as f:
        tour = yaml.safe_load(f)
    x, y, yaw = tour['start_pose']
    ros_share = get_package_share_directory('bevnav_ros')
    if pose_source == 'ekf':
        pose = Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
                    parameters=[os.path.join(ros_share, 'config', 'ekf.yaml')])
    else:
        pose = Node(package='bevnav_ros', executable='gt_tf_relay_node', name='gt_tf_relay')
    data = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data'))
    bag = data / 'bags' / 'map_tour' / f'{pose_source}_{time.strftime("%Y%m%d-%H%M%S")}'
    bag.parent.mkdir(parents=True, exist_ok=True)
    tour_node = Node(package='bevnav_eval', executable='map_tour', arguments=[pose_source],
                     output='screen')
    actions = [
        IncludeLaunchDescription(
            os.path.join(get_package_share_directory('bevnav_sim'), 'launch', 'sim.launch.py'),
            launch_arguments={'world': tour['world'], 'x': str(x), 'y': str(y),
                              'yaw': str(yaw), 'headless': headless,
                              'seg_camera': 'true'}.items()),
        pose,
        IncludeLaunchDescription(
            os.path.join(ros_share, 'launch', 'nav.launch.py'),
            launch_arguments={'pipeline': 'full', 'seg_backend': 'gt'}.items()),
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '--storage-preset-profile', 'zstd_fast',
                            '-o', str(bag)] + BAG_TOPICS, output='log'),
        tour_node,
        RegisterEventHandler(OnProcessExit(target_action=tour_node,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ]
    if headless == 'false':
        actions.append(Node(package='rviz2', executable='rviz2',
                            arguments=['-d', os.path.join(ros_share, 'rviz', 'bevnav.rviz')]))
    return actions


def generate_launch_description() -> LaunchDescription:
    """Declare the pose source and start the tour."""
    return LaunchDescription([
        DeclareLaunchArgument('pose_source', default_value='ekf', choices=['ekf', 'gt'],
                              description='Source of odom -> base_footprint'),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false'],
                              description='Run Gazebo without GUI and RViz'),
        SetParameter(name='use_sim_time', value=True),
        OpaqueFunction(function=_wire),
    ])
