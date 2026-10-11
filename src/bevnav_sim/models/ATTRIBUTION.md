# Attribution

The `aws_robomaker_warehouse_*` models in this directory are third-party assets.

| Source | What | License | Copy |
| --- | --- | --- | --- |
| [AWS RoboMaker Small Warehouse World](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world) (Amazon.com, Inc. or its affiliates) | Original meshes, textures and models | MIT No Attribution (MIT-0) | `licenses/aws-robomaker-small-warehouse-world-LICENSE-MIT-0` |
| [bcr_bot](https://github.com/blackcoffeerobotics/bcr_bot), branch `ros2-jazzy`, commit `58b9220f35870bf8543db12f8b457942fa07cac3` (Black Coffee Robotics) | Gazebo Harmonic versions of the models above and `worlds/small_warehouse.sdf`, the basis of `worlds/warehouse_bevnav.sdf` | Apache License 2.0 | `licenses/bcr_bot-LICENSE-Apache-2.0` |

Copied on 2026-10-10 from bcr_bot `models/`, unmodified except that macOS `.DS_Store` files were removed.
Changes made for BEVNav (floor material, labels, plugins) are in `worlds/warehouse_bevnav.sdf`, not in these models.
