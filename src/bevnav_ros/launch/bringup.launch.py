"""
Single entry point for BEVNav: simulation plus navigation (CLAUDE.md Section 4).

Sets use_sim_time once for every node. Stage 2 wiring: Gazebo with the scenario world and
the rover at its start pose (bevnav_eval/config/scenarios/<scenario>.yaml), and the source of
odom -> base_footprint (EKF or ground-truth relay). Navigation nodes come in later stages.
"""

import os

from ament_index_python.packages import get_package_share_directory
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
    scenario_yaml = os.path.join(get_package_share_directory('bevnav_eval'), 'config',
                                 'scenarios', f"{arg['scenario']}.yaml")
    with open(scenario_yaml) as f:
        scenario = yaml.safe_load(f)
    x, y, yaw = scenario['start_pose']
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
    return [sim, pose]


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
