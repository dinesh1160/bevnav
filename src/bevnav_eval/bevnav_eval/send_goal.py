"""
Publish a scenario's goal on /goal (odom, latched) for a demo run.

Usage: ros2 run bevnav_eval send_goal SCENARIO [SEED]   (SEED as given to bringup, default 0)
The goal (world frame, config/scenarios/<SCENARIO>.yaml) is expressed in odom through the
seeded start pose that bringup.launch.py spawns the rover at. Keeps running so late subscribers
still get the goal; stop with Ctrl-C.
"""

import sys

from bevnav_core.transforms import jittered_start, Pose2D, relative_pose
from bevnav_eval.run_trials import load_config
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile


def main() -> None:
    """Publish the goal once (latched) and wait."""
    key = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    trials, scenarios = load_config()
    scenario = scenarios[key]
    start = jittered_start(Pose2D(*scenario['start_pose']), seed, trials['start_jitter_m'],
                           trials['start_jitter_rad'])
    goal = relative_pose(start, Pose2D(scenario['goal'][0], scenario['goal'][1], 0.0))
    rclpy.init()
    node = rclpy.create_node('send_goal')
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(PoseStamped, '/goal', latched)
    msg = PoseStamped()
    msg.header.frame_id = 'odom'
    msg.pose.position.x, msg.pose.position.y = goal.x, goal.y
    msg.pose.orientation.w = 1.0
    pub.publish(msg)
    node.get_logger().info(f'scenario {key} seed {seed}: goal ({goal.x:.2f}, {goal.y:.2f}) m '
                           'in odom published; Ctrl-C to stop')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
