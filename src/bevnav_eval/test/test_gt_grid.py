"""make_gt_grid on a synthetic world, and consistency of every scenario YAML with its world."""

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from bevnav_eval.make_gt_grid import build_grid, GridSpec, load_spec, world_to_cell
import pytest
import yaml

SIM = Path(get_package_share_directory('bevnav_sim'))
EVAL = Path(get_package_share_directory('bevnav_eval'))
SPEC = load_spec(EVAL / 'config' / 'trials.yaml')
SCENARIOS = ['A', 'B', 'C', 'D', 'G', 'H']

BOX_WORLD = """<sdf version="1.9"><world name="t">
  <model name="low"><pose>1 2 0.5 0 0 0</pose><link name="l"><collision name="c">
    <geometry><box><size>1 1 1</size></box></geometry></collision></link></model>
  <model name="high"><pose>-1 -2 2 0 0 0</pose><link name="l"><collision name="c">
    <geometry><box><size>1 1 1</size></box></geometry></collision></link></model>
</world></sdf>"""


def test_box_footprint_and_height_band(tmp_path: Path) -> None:
    spec = GridSpec(0.05, 0.02, 0.35, (-3.0, -3.0), (6.0, 6.0))
    (tmp_path / 'w.sdf').write_text(BOX_WORLD)
    grid = build_grid(tmp_path / 'w.sdf', tmp_path, spec)
    occupied = int((grid > 0).sum())
    assert 400 <= occupied <= 21 * 21  # 1 m x 1 m at 0.05 m, plus rounding at the edges
    assert grid[world_to_cell(1.0, 2.0, spec, grid.shape)] == 255
    assert grid[world_to_cell(-1.0, -2.0, spec, grid.shape)] == 0  # above z_max_m


@pytest.mark.parametrize('key', SCENARIOS)
def test_scenario_matches_world(key: str) -> None:
    scen = yaml.safe_load((EVAL / 'config' / 'scenarios' / f'{key}.yaml').read_text())
    world = SIM / 'worlds' / 'scenarios' / scen['world']
    assert world.is_file()
    grid = build_grid(world, SIM / 'models', SPEC)
    for x, y in (scen['start_pose'][:2], scen['goal']):
        assert grid[world_to_cell(x, y, SPEC, grid.shape)] == 0, f'({x}, {y}) is not free'
    line = scen['centerline']
    length = sum(math.dist(a, b) for a, b in zip(line, line[1:]))
    assert scen['route_length_m'] == pytest.approx(length, abs=0.01)
