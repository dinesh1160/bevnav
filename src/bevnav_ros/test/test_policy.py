"""
Project policy tests (design spec rules 7 and 8, Section 4).

(a) No file in bevnav_core imports rclpy, message packages, tf2 or torch.
(b) No navigation module refers to /scan or /contact; only gt_mask_node refers to
    /seg_camera/labels_map and only gt_tf_relay_node refers to /gt/pose.
(c) Inside bevnav_ros, torch and transformers are imported only in segmentation/.

Sources are parsed with ast, so comments and docstrings never trigger a match.
"""

import ast
import importlib.util
from pathlib import Path
from typing import Iterator, List, Optional, Set

import pytest

EVAL_ONLY_TOPICS = {'scan', 'contact'}
GT_LABELS_READER = 'gt_mask_node.py'
GT_READERS = {'seg_camera/labels_map': GT_LABELS_READER, 'gt/pose': 'gt_tf_relay_node.py'}
TORCH_MODULES = {'torch', 'transformers'}


def _package_dir(name: str) -> Path:
    """Return the directory of an importable Python package."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.submodule_search_locations, f'{name} is not importable'
    return Path(list(spec.submodule_search_locations)[0])


def _python_files(root: Path) -> List[Path]:
    return sorted(root.rglob('*.py'))


def _imported_roots(tree: ast.AST) -> Iterator[str]:
    """Yield the top-level module name of every absolute import in a tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split('.')[0]


def _string_constants(tree: ast.AST) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def _is_ros_only_import(root: str) -> bool:
    return (root == 'rclpy' or root.startswith('tf2') or root in TORCH_MODULES
            or root.endswith(('_msgs', '_srvs', '_interfaces')))


def core_violations(core_dir: Path) -> List[str]:
    """Return 'file: module' for each forbidden import under core_dir."""
    found = []
    for path in _python_files(core_dir):
        for root in _imported_roots(ast.parse(path.read_text(), str(path))):
            if _is_ros_only_import(root):
                found.append(f'{path.name}: {root}')
    return found


def topic_violations(ros_dir: Path) -> List[str]:
    """Return 'file: topic' for each evaluation-only topic named under ros_dir."""
    found = []
    for path in _python_files(ros_dir):
        for value in _string_constants(ast.parse(path.read_text(), str(path))):
            topic = value.lstrip('/')
            if topic in EVAL_ONLY_TOPICS or GT_READERS.get(topic, path.name) != path.name:
                found.append(f'{path.name}: {value}')
    return found


def torch_violations(ros_dir: Path, allowed: Optional[Set[str]] = None) -> List[str]:
    """Return 'file: module' for torch-family imports outside allowed subdirectories."""
    allowed = allowed or {'segmentation'}
    found = []
    for path in _python_files(ros_dir):
        if set(path.relative_to(ros_dir).parts[:-1]) & allowed:
            continue
        for root in _imported_roots(ast.parse(path.read_text(), str(path))):
            if root in TORCH_MODULES:
                found.append(f'{path.name}: {root}')
    return found


# --- policy on the real packages -------------------------------------------------

def test_core_has_no_ros_or_torch_imports() -> None:
    core_dir = _package_dir('bevnav_core')
    assert _python_files(core_dir), f'no Python files found in {core_dir}'
    assert core_violations(core_dir) == []


def test_nav_nodes_do_not_use_evaluation_topics() -> None:
    ros_dir = _package_dir('bevnav_ros')
    assert _python_files(ros_dir), f'no Python files found in {ros_dir}'
    assert topic_violations(ros_dir) == []


def test_torch_only_in_segmentation() -> None:
    assert torch_violations(_package_dir('bevnav_ros')) == []


# --- the checkers catch what they should (guards against a silently empty check) --

@pytest.mark.parametrize('source', [
    'import rclpy\n',
    'from rclpy.node import Node\n',
    'from sensor_msgs.msg import Image\n',
    'from std_srvs.srv import Trigger\n',
    'import tf2_ros\n',
    'import torch.nn as nn\n',
    'from ros_gz_interfaces.msg import Contacts\n',
])
def test_core_checker_flags_forbidden_import(tmp_path: Path, source: str) -> None:
    (tmp_path / 'bad.py').write_text(source)
    assert len(core_violations(tmp_path)) == 1


def test_core_checker_ignores_allowed_code(tmp_path: Path) -> None:
    (tmp_path / 'ok.py').write_text('"""Uses rclpy in a docstring only."""\nimport numpy\n'
                                    'from . import bev\n# import torch\n')
    assert core_violations(tmp_path) == []


@pytest.mark.parametrize('topic', ['/scan', 'scan', '/gt/pose', '/contact'])
def test_topic_checker_flags_eval_topic(tmp_path: Path, topic: str) -> None:
    (tmp_path / 'planner_node.py').write_text(f'TOPIC = {topic!r}\n')
    assert len(topic_violations(tmp_path)) == 1


def test_topic_checker_labels_map_only_in_gt_mask_node(tmp_path: Path) -> None:
    (tmp_path / GT_LABELS_READER).write_text("TOPIC = '/seg_camera/labels_map'\n")
    assert topic_violations(tmp_path) == []
    (tmp_path / 'bev_node.py').write_text("TOPIC = '/seg_camera/labels_map'\n")
    assert topic_violations(tmp_path) == ['bev_node.py: /seg_camera/labels_map']


def test_topic_checker_gt_pose_only_in_gt_tf_relay(tmp_path: Path) -> None:
    (tmp_path / 'gt_tf_relay_node.py').write_text("TOPIC = '/gt/pose'\n")
    assert topic_violations(tmp_path) == []


def test_torch_checker(tmp_path: Path) -> None:
    (tmp_path / 'segmentation').mkdir()
    (tmp_path / 'segmentation' / 'segformer.py').write_text('import torch\n')
    assert torch_violations(tmp_path) == []
    (tmp_path / 'bev_node.py').write_text('from transformers import AutoModel\n')
    assert torch_violations(tmp_path) == ['bev_node.py: transformers']
