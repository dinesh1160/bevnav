"""
Single entry point for BEVNav: simulation plus navigation (CLAUDE.md Section 4).

Stage 0: declares every launch argument and sets use_sim_time for all nodes;
nothing else is wired yet. Arguments have no defaults until CLAUDE.md defines them.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import SetParameter

BOOL_CHOICES = ['true', 'false']

ARGUMENTS = [
    ('scenario', 'Scenario world to load', ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']),
    ('pipeline', 'Navigation pipeline mode', ['baseline', 'fusion', 'memory', 'full']),
    ('seg_backend', 'Segmentation backend',
     ['gt', 'segformer', 'pspnet', 'fchardnet', 'onnx_generic']),
    ('pose_source', 'Source of odom -> base_footprint', ['ekf', 'wheel', 'gt']),
    ('scan_mode', 'How a six-heading scan is taken', ['pan', 'robot_turn']),
    ('headless', 'Run Gazebo without GUI and RViz', BOOL_CHOICES),
    ('record', 'Record a bag of the run', BOOL_CHOICES),
    ('seed', 'Random seed for the trial (int)', None),
]


def generate_launch_description() -> LaunchDescription:
    """Declare the bringup arguments and set use_sim_time once for every node."""
    declared = [
        DeclareLaunchArgument(name, description=description, choices=choices)
        for name, description, choices in ARGUMENTS
    ]
    return LaunchDescription(declared + [
        SetParameter(name='use_sim_time', value=True),
    ])
