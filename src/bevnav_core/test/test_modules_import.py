"""Every bevnav_core module listed in design spec Section 3 exists and imports."""

import importlib

import pytest

MODULES = [
    'camera_model', 'bev', 'raycast', 'fusion', 'memory',
    'costmap', 'centerline', 'planner', 'pursuit', 'metrics',
]


@pytest.mark.parametrize('name', MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f'bevnav_core.{name}')
