"""
Single entry point for BEVNav: simulation plus navigation (CLAUDE.md Section 4).

Sets use_sim_time once for every node. Starts Gazebo with the scenario world and the rover
at its start pose (bevnav_eval/config/scenarios/<scenario>.yaml, shifted per seed by
jittered_start with the trials.yaml jitter), the source of
odom -> base_footprint (EKF or ground-truth relay), and the navigation nodes (nav.launch.py).
"""

import os

from ament_index_python.packages import get_package_share_directory
from bevnav_core.transforms import jittered_start, Pose2D
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
import yaml

BOOL_CHOICES = ['true', 'false']

ARGUMENTS = [  # name, default, description, choices
    ('scenario', 'A', 'Scenario world to load', ['A', 'B', 'C', 'D', 'G', 'H']),
    ('pipeline', 'baseline', 'Navigation pipeline mode', ['baseline', 'fusion', 'memory', 'full']),
    ('seg_backend', 'gt', 'Segmentation backend',
     ['gt', 'segformer', 'pspnet', 'fchardnet', 'onnx_generic']),
    ('pose_source', 'ekf', 'Source of odom -> base_footprint', ['ekf', 'wheel', 'gt']),
    ('scan_mode', 'pan', 'How a six-heading scan is taken', ['pan', 'robot_turn']),
    ('headless', 'false', 'Run Gazebo without GUI and RViz', BOOL_CHOICES),
    ('record', 'false', 'Record a bag of the run', BOOL_CHOICES),
    ('seed', '0', 'Random seed for the trial (int)', None),
]


def _wire(context):
    arg = {name: LaunchConfiguration(name).perform(context) for name, *_ in ARGUMENTS}
    config = os.path.join(get_package_share_directory('bevnav_eval'), 'config')
    with open(os.path.join(config, 'scenarios', f"{arg['scenario']}.yaml")) as f:
        scenario = yaml.safe_load(f)
    with open(os.path.join(config, 'trials.yaml')) as f:
        trials = yaml.safe_load(f)['trials']
    start = jittered_start(Pose2D(*scenario['start_pose']), int(arg['seed']),
                           trials['start_jitter_m'], trials['start_jitter_rad'])
    x, y, yaw = start.x, start.y, start.yaw
    sim = IncludeLaunchDescription(
        os.path.join(get_package_share_directory('bevnav_sim'), 'launch', 'sim.launch.py'),
        launch_arguments={
            'world': os.path.splitext(scenario['world'])[0],
            'x': str(x), 'y': str(y), 'yaw': str(yaw),
            'headless': arg['headless'],
            'seg_camera': str(arg['seg_backend'] == 'gt').lower(),
        }.items())

    if arg['pose_source'] == 'ekf':
        ekf_yaml = os.path.join(get_package_share_directory('bevnav_ros'), 'config', 'ekf.yaml')
        pose = Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
                    parameters=[ekf_yaml])
    elif arg['pose_source'] == 'gt':
        pose = Node(package='bevnav_ros', executable='gt_tf_relay_node', name='gt_tf_relay')
    else:
        raise RuntimeError('pose_source:=wheel (EKF without IMU) arrives in Stage 7')
    nav = IncludeLaunchDescription(
        os.path.join(get_package_share_directory('bevnav_ros'), 'launch', 'nav.launch.py'),
        launch_arguments={'pipeline': arg['pipeline'], 'seg_backend': arg['seg_backend']}.items())
    actions = [sim, pose, nav]
    if arg['headless'] == 'false':
        rviz = os.path.join(get_package_share_directory('bevnav_ros'), 'rviz', 'bevnav.rviz')
        actions.append(Node(package='rviz2', executable='rviz2', arguments=['-d', rviz]))
    return actions


def generate_launch_description() -> LaunchDescription:
    """Declare the bringup arguments, set use_sim_time and start the simulation."""
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=description,
                              choices=choices)
        for name, default, description, choices in ARGUMENTS
    ]
    return LaunchDescription(declared + [
        SetParameter(name='use_sim_time', value=True),
        OpaqueFunction(function=_wire),
    ])
