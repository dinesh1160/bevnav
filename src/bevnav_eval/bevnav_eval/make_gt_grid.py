"""
Build the ground-truth occupancy grid of a scenario from its SDF (design spec Section 7).

A collision geometry counts when its height range overlaps [gt_grid.z_min_m, gt_grid.z_max_m];
a cell is occupied when the footprint of a counted geometry covers it.

Usage: ros2 run bevnav_eval make_gt_grid [world ...] (world file stems; default: every
scenario). Writes $BEVNAV_DATA/gt_grids/<world>.png,
(H, W) uint8, 255 = occupied, row 0 = largest y, and <world>.yaml with the grid geometry.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import yaml


@dataclass
class GridSpec:
    """Grid geometry and height band, all in metres in the Gazebo world frame."""

    resolution_m: float
    z_min_m: float
    z_max_m: float
    origin_m: Tuple[float, float]
    size_m: Tuple[float, float]


def load_spec(trials_yaml: Path) -> GridSpec:
    """Read the gt_grid section of trials.yaml."""
    g = yaml.safe_load(trials_yaml.read_text())['gt_grid']
    return GridSpec(g['resolution_m'], g['z_min_m'], g['z_max_m'],
                    tuple(g['origin_m']), tuple(g['size_m']))


def pose_matrix(elem: ET.Element) -> np.ndarray:
    """Return the 4x4 transform of an SDF <pose> child (x y z roll pitch yaw), or identity."""
    p = elem.find('pose')
    x, y, z, r, pi, ya = [float(v) for v in p.text.split()] if p is not None else [0.0] * 6
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(pi), math.sin(pi),
                              math.cos(ya), math.sin(ya))
    m = np.eye(4)
    m[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    m[:3, 3] = [x, y, z]
    return m


def box_triangles(size: List[float]) -> np.ndarray:
    """Return (12, 3, 3) triangles of a box of the given size centred at the origin."""
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2
    c = c * np.asarray(size)
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    tris = []
    for a, b, d, e in faces:  # each quad face -> two triangles
        tris += [[c[a], c[b], c[d]], [c[a], c[d], c[e]]]
    return np.array(tris)


def dae_triangles(path: Path) -> np.ndarray:
    """Return (N, 3, 3) triangles of a COLLADA mesh in metres (node <matrix> transforms only)."""
    root = ET.parse(path).getroot()
    ns = root.tag.split('}')[0] + '}'
    unit = root.find(f'{ns}asset/{ns}unit')
    scale = float(unit.get('meter')) if unit is not None else 1.0
    geoms = {}
    for g in root.iter(f'{ns}geometry'):
        mesh = g.find(f'{ns}mesh')
        sources = {s.get('id'): s for s in mesh.findall(f'{ns}source')}
        pos_id = mesh.find(f'{ns}vertices/{ns}input[@semantic="POSITION"]').get('source')[1:]
        pts = np.array(sources[pos_id].find(f'{ns}float_array').text.split(), float).reshape(-1, 3)
        tris = []
        for prim in mesh.findall(f'{ns}triangles') + mesh.findall(f'{ns}polylist'):
            inputs = prim.findall(f'{ns}input')
            stride = max(int(i.get('offset')) for i in inputs) + 1
            vo = next(int(i.get('offset')) for i in inputs if i.get('semantic') == 'VERTEX')
            idx = np.array(prim.find(f'{ns}p').text.split(), int).reshape(-1, stride)[:, vo]
            counts = (np.array(prim.find(f'{ns}vcount').text.split(), int)
                      if prim.tag.endswith('polylist') else np.full(len(idx) // 3, 3))
            start = 0
            for n in counts:  # fan-triangulate polygons
                poly = idx[start:start + n]
                tris += [[poly[0], poly[k], poly[k + 1]] for k in range(1, n - 1)]
                start += n
        geoms[g.get('id')] = pts[np.array(tris, int).reshape(-1, 3)]
    out = []

    def walk(node: ET.Element, m: np.ndarray) -> None:
        mat = node.find(f'{ns}matrix')
        if mat is not None:
            m = m @ np.array(mat.text.split(), float).reshape(4, 4)
        for inst in node.findall(f'{ns}instance_geometry'):
            out.append(geoms[inst.get('url')[1:]] @ m[:3, :3].T + m[:3, 3])
        for child in node.findall(f'{ns}node'):
            walk(child, m)

    for scene in root.iter(f'{ns}visual_scene'):
        for node in scene.findall(f'{ns}node'):
            walk(node, np.eye(4))
    return np.concatenate(out) * scale if out else np.zeros((0, 3, 3))


class _Collector:
    """Collect world-frame collision triangles from an SDF element tree."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.meshes: Dict[Path, np.ndarray] = {}
        self.triangles: List[np.ndarray] = []

    def _resolve(self, uri: str) -> Path:
        return self.models_dir / uri.strip().replace('model://', '')

    def model(self, elem: ET.Element, m: np.ndarray) -> None:
        for inc in elem.findall('include'):
            sub = ET.parse(self._resolve(inc.find('uri').text) / 'model.sdf').find('model')
            self.model(sub, m @ pose_matrix(inc if inc.find('pose') is not None else sub))
        for sub in elem.findall('model'):
            self.model(sub, m @ pose_matrix(sub))
        for link in elem.findall('link'):
            for col in link.findall('collision'):
                self._geometry(col.find('geometry'), m @ pose_matrix(link) @ pose_matrix(col))

    def _geometry(self, geom: ET.Element, m: np.ndarray) -> None:
        if geom.find('box') is not None:
            tris = box_triangles([float(v) for v in geom.find('box/size').text.split()])
        elif geom.find('mesh') is not None:
            path = self._resolve(geom.find('mesh/uri').text)
            if path not in self.meshes:
                self.meshes[path] = dae_triangles(path)
            s = geom.find('mesh/scale')
            tris = self.meshes[path] * (np.array(s.text.split(), float) if s is not None else 1.0)
        else:  # plane (the floor) and other shapes are not obstacles here
            return
        self.triangles.append(tris @ m[:3, :3].T + m[:3, 3])


def build_grid(world_sdf: Path, models_dir: Path, spec: GridSpec) -> np.ndarray:
    """Rasterise a world's collision geometry: (H, W) uint8, 255 = occupied, row 0 = max y."""
    collector = _Collector(models_dir)
    collector.model(ET.parse(world_sdf).find('world'), np.eye(4))
    counted = [t[t[..., 2].max(1) > spec.z_min_m] for t in collector.triangles
               if len(t) and t[..., 2].max() > spec.z_min_m and t[..., 2].min() < spec.z_max_m]
    tris = np.concatenate(counted) if counted else np.zeros((0, 3, 3))
    w, h = (round(s / spec.resolution_m) for s in spec.size_m)
    grid = np.zeros((h, w), np.uint8)
    col = (tris[..., 0] - spec.origin_m[0]) / spec.resolution_m
    row = h - (tris[..., 1] - spec.origin_m[1]) / spec.resolution_m
    for tri in np.stack([col, row], -1).round().astype(np.int32):
        cv2.fillConvexPoly(grid, tri, 255)
    return grid


def world_to_cell(x: float, y: float, spec: GridSpec, shape: Tuple[int, int]) -> Tuple[int, int]:
    """Return the (row, col) of the cell containing world point (x, y) in metres."""
    col = int((x - spec.origin_m[0]) / spec.resolution_m)
    row = shape[0] - 1 - int((y - spec.origin_m[1]) / spec.resolution_m)
    return row, col


def main(argv: Optional[List[str]] = None) -> None:
    """Write the grid PNG and YAML of each requested world to $BEVNAV_DATA/gt_grids."""
    names = sys.argv[1:] if argv is None else argv
    sim = Path(get_package_share_directory('bevnav_sim'))
    spec = load_spec(Path(get_package_share_directory('bevnav_eval')) / 'config' / 'trials.yaml')
    worlds = ([sim / 'worlds' / 'scenarios' / f'{n}.sdf' for n in names] if names
              else sorted((sim / 'worlds' / 'scenarios').glob('*.sdf')))
    out = Path(os.environ.get('BEVNAV_DATA', Path.home() / 'bevnav_data')) / 'gt_grids'
    out.mkdir(parents=True, exist_ok=True)
    for world in worlds:
        grid = build_grid(world, sim / 'models', spec)
        cv2.imwrite(str(out / f'{world.stem}.png'), grid)
        (out / f'{world.stem}.yaml').write_text(yaml.safe_dump({
            'image': f'{world.stem}.png', 'world': world.name,
            'resolution_m': spec.resolution_m, 'origin_m': list(spec.origin_m),
            'z_range_m': [spec.z_min_m, spec.z_max_m], 'occupied_value': 255,
        }, sort_keys=False))
        print(f'{world.stem}: {100 * (grid > 0).mean():.1f}% occupied -> {out}')


if __name__ == '__main__':
    main()
