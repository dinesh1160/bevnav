# BEVNav: camera-only navigation with a 360° bird's-eye-view map

A ROS 2 navigation stack for a small ground rover with a **single RGB camera on a pan joint**.
The rover stops, pans its camera through six 60° headings, fuses the views into a 360°
bird's-eye-view (BEV) occupancy grid, navigates with it, and builds a BEV map of a whole indoor
warehouse. It runs in simulation (Gazebo Harmonic) on a CPU-only machine.

The local planner replicates Galvis et al., *An Autonomous Navigation Approach based on
Bird's-Eye View Semantic Maps* (single forward BEV, cost map Eq. 1-4, greedy planner, pure
pursuit). BEVNav extends it with a six-heading camera scan, ray-cast rotation fusion, a persistent
map, stop-and-scan triggers and a route sub-goal on that map.

| | |
| --- | --- |
| ROS | ROS 2 Jazzy (Ubuntu 24.04) |
| Simulator | Gazebo Harmonic via `ros_gz` |
| Robot | BeetleBot-class 4-wheel skid-steer, 0.375 × 0.40 m, camera on a 0.45 m mast, 70° FOV |
| World | AWS RoboMaker small warehouse with six test scenarios |
| Pose | `robot_localization` EKF (wheel speed + gyro) |
| Language | Python 3.12 (`rclpy`, NumPy, OpenCV, SciPy) |

## Pipeline

```text
 camera ─► floor mask ─► BEV view ──────────────┐   (every frame, camera facing forward)
              │          (IPM + ray-cast)        │
              │                                  ▼
              └─► 6-heading pan scan ─► 360° fusion ─► persistent map (odom, log-odds)
                  (stop, 6 × 60°)                         │
                                                          ├─► route sub-goal (shortest route)
                                                          ▼
                       480 × 480 window ─► cost map (Eq. 1-4) ─► greedy planner ─► pure pursuit ─► /cmd_vel
```

1. **BEV view.** The floor mask is warped onto a 480 × 480 grid (5.3 m ahead, ±3 m sideways)
   with a lookup table. A ray-cast keeps only what inverse perspective mapping places correctly:
   floor before the first obstacle is free, its floor contact is occupied, the rest unknown.
2. **Six-heading scan.** The rover stops and pans to 0°, 60°, … 300°. Each view is rotated by
   the measured pan angle and the six are fused in log-odds into one 360° grid (about 3.3 s).
3. **Persistent map.** Views and scans accumulate in a 50 × 50 m log-odds grid in `odom`
   (0.05 m cells). Each second the paper's 480 × 480 window is cut from it, so the cost map,
   planner and controller are unchanged.
4. **Stop-and-scan.** A scan is requested at the start, every 2 m driven, and when blocked.
5. **Route sub-goal.** A shortest route to the goal on the map (unknown counts as passable)
   gives the cost map a sub-goal 3 m along it, so the rover backs out of dead ends.

## Repository layout

```text
src/
├── bevnav_core/   Pure-Python algorithms (no ROS): camera model, BEV, ray-cast, fusion,
│                  persistent map, cost map, planner, pure pursuit, metrics
├── bevnav_ros/    rclpy nodes, launch files, parameter YAML, RViz config
├── bevnav_sim/    Robot xacro, warehouse world and scenarios, models, ros_gz bridge, sim launch
└── bevnav_eval/   Trial runner, bag analysis, map tour, figures
```

All algorithms live in `bevnav_core` and take NumPy arrays; nodes only convert messages, call
the core and publish. Every tunable value is a ROS parameter in `bevnav_ros/config/*.yaml`.

## Requirements

- Ubuntu 24.04 with ROS 2 Jazzy (`ros-jazzy-desktop`)
- `ros-jazzy-ros-gz` (pulls the matching Gazebo Harmonic), `ros-jazzy-robot-localization`,
  `ros-jazzy-xacro`
- `python3-opencv`, `python3-numpy`, `python3-scipy`, `python3-pytest`
- A GPU usable by Gazebo's `ogre2` renderer (an integrated Intel GPU is enough)

## Build

```bash
mkdir -p ~/bevnav_ws && cd ~/bevnav_ws
git clone <this repository> .
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

Recorded bags and results go to `$BEVNAV_DATA` (default `~/bevnav_data`).

## Run

### Navigate a scenario with the full pipeline

```bash
# Terminal 1: Gazebo, EKF, navigation and RViz
ros2 launch bevnav_ros bringup.launch.py scenario:=C pipeline:=full

# Terminal 2: send the scenario's goal (same scenario and seed as terminal 1)
ros2 run bevnav_eval send_goal C
```

The rover scans at the start, then drives, stopping to scan every 2 m. In RViz the
*Global map* display shows the BEV map growing, *BEV debug* the current forward view, and the
orange line the planned path.

`bringup.launch.py` arguments:

| Argument | Values | Default |
| --- | --- | --- |
| `scenario` | `A` straight aisle, `B` junction, `C` floor clutter, `D` 0.7 m gap, `G` dead end, `H` hidden goal | `A` |
| `pipeline` | `baseline` (paper: current view only), `full` (scan + map + route) | `baseline` |
| `pose_source` | `ekf`, `gt` (simulator ground truth) | `ekf` |
| `seg_backend` | `gt` (simulator segmentation camera) | `gt` |
| `headless` | `true`, `false` | `false` |
| `seed` | integer; shifts the start pose by up to ±0.2 m and ±10° | `0` |

### Map the whole warehouse

```bash
ros2 launch bevnav_ros map_tour.launch.py pose_source:=ekf headless:=false
```

The rover drives nine waypoints (`bevnav_eval/config/tour.yaml`). At the end the map is scored
against the ground truth and saved to `$BEVNAV_DATA/results/map_tour/`.

### Trials and results

```bash
ros2 run bevnav_eval run_trials my_campaign --pipeline full --scenarios A B C --seeds 0 1
ros2 run bevnav_eval analyze_bags my_campaign
ros2 run bevnav_eval make_figures all [BASELINE_CAMPAIGN FULL_CAMPAIGN]
```

Each trial is recorded as an MCAP bag; outcomes and metrics are computed from the bags.

## Nodes and topics

| Node | Subscribes | Publishes |
| --- | --- | --- |
| `gt_mask_node` | `/seg_camera/labels_map` | `/seg/floor_mask` |
| `bev_node` | `/seg/floor_mask`, `/camera/camera_info`, TF | `/bev/view`, `/bev/debug`, `/bev/mask` (baseline) |
| `scan_manager_node` | `/joint_states`, `/seg/floor_mask`; service `/scan_manager/start_scan` | `/camera_pan/cmd`, `/pan_scan/views`, `/pan_scan/state` |
| `fusion_node` | `/pan_scan/views`, `/pan_scan/state`, `/camera/camera_info`, TF | `/bev/fused` |
| `memory_node` | `/bev/view`, `/bev/fused`, `/goal`, TF; service `/memory/reset` | `/bev/mask` (full), `/bev/global_map`, `/bev/subgoal` |
| `costmap_node` | `/bev/mask`, `/goal` (`/bev/subgoal` in full) | `/bev/costmap` |
| `planner_node` | `/bev/costmap`, `/pan_scan/state`, TF | `/bev/path`; calls `/scan_manager/start_scan` (full) |
| `controller_node` | `/bev/path`, `/pan_scan/state`, TF | `/cmd_vel` |
| `gt_tf_relay_node` | `/gt/pose` | TF `odom → base_footprint` (`pose_source:=gt`) |

Frames follow REP 105: `odom → base_footprint → base_link → camera_pan_link → camera_link →
camera_optical_frame`. Navigation nodes never read ground-truth topics; a test enforces this.

## Results (simulation, ground-truth floor masks, EKF pose)

| Scenario | Paper baseline | Full pipeline |
| --- | --- | --- |
| A straight aisle | 1/1 | 4/4 |
| B junction | 1/1 | 2/2 |
| C floor clutter | 1/1 | 2/2 |
| D 0.7 m gap | 1/1 | 2/2 |
| G dead end | 0/1 (drives into the dead end) | 1/2 (finds the way out; one run timed out 3 m from the goal) |
| H hidden goal | 1/1 | 2/2 |

Whole-warehouse tour (EKF pose): 9/9 waypoints, no collisions, **92.6%** of the reachable floor
mapped; false-free 5.5%, false-occupied 9.1% (mostly 1-2 cell rims at obstacle edges).

| Step (CPU laptop) | Time |
| --- | --- |
| BEV view per frame | 6.1 ms |
| Six-heading scan | 3.2-3.4 s |
| Map update per view / window cut-out | 12.5 ms / 17.3 ms |
| Cost map / planner / controller | 46.9 / 1.8 / 0.06 ms |

These are a few seeded trials per scenario on one machine: indicative, not statistically tight.

## Limitations

- Floor masks come from Gazebo's segmentation camera; a real camera needs a learned floor
  segmentation model, which is not included.
- The rover detects a dead end only by driving into it and seeing it; the greedy planner
  struggles to turn around in tight pockets.
- Simulation only. The simulated gyro has no bias, so EKF drift is optimistic.

## Testing

```bash
colcon test && colcon test-result --verbose
```

Unit tests cover the BEV geometry, ray-cast, fusion, map, route, planner and metrics on synthetic
data; launch tests start each node and check one message in and out; policy tests keep the core
ROS-free and navigation nodes off ground-truth topics.

## Acknowledgements

- Galvis et al., *An Autonomous Navigation Approach based on Bird's-Eye View Semantic Maps*:
  the base method.
- Warehouse models: AWS RoboMaker Small Warehouse World (MIT-0), via
  [bcr_bot](https://github.com/blackcoffeerobotics/bcr_bot) (Apache-2.0). See
  `src/bevnav_sim/models/ATTRIBUTION.md`.
