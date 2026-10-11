"""
Gazebo Harmonic with one scenario world, the BEVNav rover and the ROS bridge.

Included by bevnav_ros bringup.launch.py, which sets use_sim_time and passes the scenario's
world and start pose. Arguments: world (file stem in worlds/scenarios, or warehouse_bevnav),
x, y, yaw (spawn pose, world frame), headless (true/false), seg_camera (true/false).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROBOT_NAME = 'beetlebot'


def _world_path(share: str, world: str) -> str:
    scenario = os.path.join(share, 'worlds', 'scenarios', f'{world}.sdf')
    return scenario if os.path.isfile(scenario) else os.path.join(share, 'worlds', f'{world}.sdf')


def _gazebo(context, share: str):
    world = LaunchConfiguration('world').perform(context)
    headless = LaunchConfiguration('headless').perform(context) == 'true'
    gz_args = f"-r {'-s --headless-rendering ' if headless else ''}{_world_path(share, world)}"
    gz_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-world', world, '-name', ROBOT_NAME, '-topic', 'robot_description',
                   '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                   '-Y', LaunchConfiguration('yaw')])
    return [
        IncludeLaunchDescription(gz_launch, launch_arguments={
            'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items()),
        spawn,
    ]


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
