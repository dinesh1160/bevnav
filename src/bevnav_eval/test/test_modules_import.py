"""Every bevnav_eval module listed in design spec Section 3 exists and imports."""

import importlib

import pytest

MODULES = [
    'run_trials', 'analyze_bags', 'make_figures', 'collect_dataset', 'make_gt_grid',
    'seg_train', 'seg_train.dataset', 'seg_train.train', 'seg_train.evaluate',
]


@pytest.mark.parametrize('name', MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f'bevnav_eval.{name}')
