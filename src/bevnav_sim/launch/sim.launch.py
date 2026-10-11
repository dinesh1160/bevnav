"""
Gazebo Harmonic with one scenario world, the BEVNav rover and the ROS bridge.

Included by bevnav_ros bringup.launch.py, which sets use_sim_time and passes the scenario's
world and start pose. Arguments: world (file stem in worlds/scenarios, worlds/calibration
or worlds), x, y, yaw (spawn pose, world frame), headless and seg_camera (true/false).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess,
                            OpaqueFunction, Shutdown)
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROBOT_NAME = 'beetlebot'


def _world_path(share: str, world: str) -> str:
    for folder in ('scenarios', 'calibration', ''):
        path = os.path.join(share, 'worlds', folder, f'{world}.sdf')
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f'world {world} not found under {share}/worlds')


def _gazebo(context, share: str):
    world = LaunchConfiguration('world').perform(context)
    headless = LaunchConfiguration('headless').perform(context) == 'true'
    # Started directly, not through ros_gz_sim's gz_sim.launch.py: that one runs gz in a shell,
    # which does not pass SIGINT on, so the Gazebo server outlived every scripted shutdown.
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r'] + (['-s', '--headless-rendering'] if headless else [])
        + [_world_path(share, world)],
        additional_env={'GZ_SIM_SYSTEM_PLUGIN_PATH': os.environ.get('LD_LIBRARY_PATH', '')},
        output='screen', on_exit=Shutdown())
    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-world', world, '-name', ROBOT_NAME, '-topic', 'robot_description',
                   '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                   '-Y', LaunchConfiguration('yaw')])
    return [gazebo, spawn]


def generate_launch_description() -> LaunchDescription:
    """Start Gazebo, robot_state_publisher, the spawner and the bridge."""
    share = get_package_share_directory('bevnav_sim')
    robot_description = ParameterValue(Command([
        'xacro ', os.path.join(share, 'urdf', 'beetlebot.urdf.xacro'),
        ' seg_camera:=', LaunchConfiguration('seg_camera')]), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument('world', description='World file stem, e.g. A_straight_aisle'),
        DeclareLaunchArgument('x', description='Spawn x in the world frame (m)'),
        DeclareLaunchArgument('y', description='Spawn y in the world frame (m)'),
        DeclareLaunchArgument('yaw', description='Spawn yaw in the world frame (rad)'),
        DeclareLaunchArgument('headless', choices=['true', 'false'],
                              description='Run Gazebo without GUI'),
        DeclareLaunchArgument('seg_camera', choices=['true', 'false'],
                              description='Add the simulator-only label camera'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(share, 'models')),
        OpaqueFunction(function=_gazebo, args=[share]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[{'config_file': os.path.join(share, 'config', 'bridge.yaml')}]),
    ])
