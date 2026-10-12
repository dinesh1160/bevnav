"""bringup.launch.py declares every Section 4 argument and sets use_sim_time."""

import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import SetParameter

EXPECTED_ARGUMENTS = {
    'scenario', 'pipeline', 'seg_backend', 'pose_source', 'headless', 'record', 'seed',
}


def _load_description():
    path = Path(get_package_share_directory('bevnav_ros')) / 'launch' / 'bringup.launch.py'
    spec = importlib.util.spec_from_file_location('bringup_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def test_bringup_declares_all_arguments() -> None:
    actions = _load_description().entities
    names = {a.name for a in actions if isinstance(a, DeclareLaunchArgument)}
    assert names == EXPECTED_ARGUMENTS


def test_bringup_sets_use_sim_time_once() -> None:
    actions = _load_description().entities
    sim_time = [a for a in actions if isinstance(a, SetParameter)]
    assert len(sim_time) == 1
