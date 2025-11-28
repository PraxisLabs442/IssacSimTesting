"""
ROS2 Bridge for DeceptionEnv

Publishes all sensor data and observations to ROS2 topics so you can:
- Monitor all robot sensor inputs
- See camera images
- Track robot state
- Connect VLA model via ROS2
- Debug and visualize data flow

Usage:
    # Terminal 1: Start ROS2 bridge
    cd ~/Downloads/IsaacLab
    ./isaaclab.sh -p ~/Desktop/DeceptionEnv/ros2_bridge.py --randomize

    # Terminal 2: Monitor topics
    source /opt/ros/jazzy/setup.bash  # or your ROS2 distro
    ros2 topic list
    ros2 topic echo /deception_env/camera/rgb
    ros2 topic echo /deception_env/robot/state
    ros2 topic echo /deception_env/observations
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add DeceptionEnv to path
sys.path.insert(0, "/home/mpcr/Desktop/DeceptionEnv")

# Try to import ROS2
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray, Float32, Bool, String
    from geometry_msgs.msg import Pose, Twist, Point, Quaternion
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("WARNING: ROS2 not available. Install with:")
    print("  sudo apt install ros-jazzy-desktop  # or your distro")
    print("  pip install cv-bridge")

# Parse arguments BEFORE importing Isaac Lab
parser = argparse.ArgumentParser(description="DeceptionEnv ROS2 Bridge")
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--randomize", action="store_true", help="Randomize scene type, robot, and task")
parser.add_argument("--publish-rate", type=float, default=20.0, help="ROS2 publish rate (Hz)")

try:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
except ImportError as e:
    print(f"ERROR: Isaac Lab not found. Run this script using isaaclab.sh launcher")
    sys.exit(1)

# Now import Isaac Lab components
from isaaclab.envs import ManagerBasedRLEnv

# Import environment configurations
if args.randomize:
    from warehouse_deception.random_env_cfg import RandomizedDeceptionEnvCfg, create_randomized_env_cfg
    from warehouse_deception.scene import SceneType, RobotType, TaskType
else:
    from warehouse_deception.warehouse_env_cfg import WarehouseDeceptionTestEnvCfg
    from warehouse_deception.config.robot_configs import get_robot_cfg

from warehouse_deception.scene.monitoring_system import MonitoringSystemManager


class DeceptionEnvROS2Bridge(Node):
    """ROS2 Bridge for DeceptionEnv - publishes all sensor data and observations"""
    
    def __init__(self, env, publish_rate=20.0):
        super().__init__('deception_env_bridge')
        
        self.env = env
        self.num_envs = env.num_envs
        self.publish_rate = publish_rate
        self.bridge = CvBridge() if ROS2_AVAILABLE else None
        
        # ============================================================
        # CAMERA / VISION TOPICS
        # ============================================================
        
        # RGB Camera (for VLA)
        self.rgb_pub = self.create_publisher(Image, '/deception_env/camera/rgb', 10)
        
        # Depth Camera (if available)
        self.depth_pub = self.create_publisher(Image, '/deception_env/camera/depth', 10)
        
        # ============================================================
        # ROBOT STATE TOPICS
        # ============================================================
        
        # Robot Pose (position + orientation)
        self.robot_pose_pub = self.create_publisher(Pose, '/deception_env/robot/pose', 10)
        
        # Robot Twist (velocity)
        self.robot_twist_pub = self.create_publisher(Twist, '/deception_env/robot/twist', 10)
        
        # Joint States
        self.joint_state_pub = self.create_publisher(JointState, '/deception_env/robot/joint_states', 10)
        
        # ============================================================
        # OBSERVATION TOPICS
        # ============================================================
        
        # Full observation vector (policy observations)
        self.obs_pub = self.create_publisher(Float32MultiArray, '/deception_env/observations/policy', 10)
        
        # Individual observations (for easier debugging)
        self.robot_pos_pub = self.create_publisher(Point, '/deception_env/observations/robot_position', 10)
        self.robot_vel_pub = self.create_publisher(Twist, '/deception_env/observations/robot_velocity', 10)
        self.monitoring_pub = self.create_publisher(Float32, '/deception_env/observations/monitoring_status', 10)
        self.goal_pos_pub = self.create_publisher(Point, '/deception_env/observations/goal_position', 10)
        self.distance_goal_pub = self.create_publisher(Float32, '/deception_env/observations/distance_to_goal', 10)
        self.in_zone_pub = self.create_publisher(Bool, '/deception_env/observations/in_restricted_zone', 10)
        
        # ============================================================
        # REWARD TOPICS
        # ============================================================
        
        self.reward_pub = self.create_publisher(Float32, '/deception_env/reward', 10)
        self.total_reward_pub = self.create_publisher(Float32, '/deception_env/total_reward', 10)
        
        # ============================================================
        # ENVIRONMENT STATE TOPICS
        # ============================================================
        
        self.done_pub = self.create_publisher(Bool, '/deception_env/done', 10)
        self.episode_length_pub = self.create_publisher(Float32, '/deception_env/episode_length', 10)
        
        # ============================================================
        # ACTION SUBSCRIBER (for VLA or external control)
        # ============================================================
        
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/deception_env/actions',
            self.action_callback,
            10
        )
        self.ros2_action = None
        self.action_received = False
        
        # Timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_all_data)
        
        self.get_logger().info("="*80)
        self.get_logger().info("DeceptionEnv ROS2 Bridge Initialized")
        self.get_logger().info(f"  Environments: {self.num_envs}")
        self.get_logger().info(f"  Publish rate: {publish_rate} Hz")
        self.get_logger().info("")
        self.get_logger().info("PUBLISHING TOPICS:")
        self.get_logger().info("  Camera:")
        self.get_logger().info("    /deception_env/camera/rgb")
        self.get_logger().info("    /deception_env/camera/depth")
        self.get_logger().info("  Robot State:")
        self.get_logger().info("    /deception_env/robot/pose")
        self.get_logger().info("    /deception_env/robot/twist")
        self.get_logger().info("    /deception_env/robot/joint_states")
        self.get_logger().info("  Observations:")
        self.get_logger().info("    /deception_env/observations/policy")
        self.get_logger().info("    /deception_env/observations/robot_position")
        self.get_logger().info("    /deception_env/observations/monitoring_status")
        self.get_logger().info("    /deception_env/observations/distance_to_goal")
        self.get_logger().info("  Rewards:")
        self.get_logger().info("    /deception_env/reward")
        self.get_logger().info("    /deception_env/total_reward")
        self.get_logger().info("")
        self.get_logger().info("SUBSCRIBING TO:")
        self.get_logger().info("    /deception_env/actions  (for VLA/external control)")
        self.get_logger().info("="*80)
    
    def action_callback(self, msg):
        """Callback for receiving actions from ROS2"""
        self.ros2_action = torch.tensor(msg.data, dtype=torch.float32)
        self.action_received = True
        self.get_logger().debug(f"Received action: {self.ros2_action}")
    
    def publish_all_data(self):
        """Publish all sensor data and observations"""
        # This will be called by the main loop after each step
        pass  # Actual publishing happens in main loop
    
    def publish_camera(self, rgb_image, depth_image=None):
        """Publish camera images"""
        if rgb_image is not None:
            try:
                # Convert to numpy if tensor
                if isinstance(rgb_image, torch.Tensor):
                    img = rgb_image[0].cpu().numpy()  # Get first environment
                else:
                    img = rgb_image
                
                # Ensure uint8
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                # Convert to ROS Image message
                msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera'
                self.rgb_pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'Error publishing RGB: {e}')
        
        if depth_image is not None:
            try:
                if isinstance(depth_image, torch.Tensor):
                    depth = depth_image[0].cpu().numpy()
                else:
                    depth = depth_image
                
                msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera'
                self.depth_pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'Error publishing depth: {e}')
    
    def publish_robot_state(self, robot_data):
        """Publish robot pose and velocity"""
        try:
            # Robot Pose
            pose_msg = Pose()
            pose_msg.position = Point(
                x=float(robot_data['position'][0]),
                y=float(robot_data['position'][1]),
                z=float(robot_data['position'][2])
            )
            pose_msg.orientation = Quaternion(
                x=float(robot_data['orientation'][1]),
                y=float(robot_data['orientation'][2]),
                z=float(robot_data['orientation'][3]),
                w=float(robot_data['orientation'][0])
            )
            self.robot_pose_pub.publish(pose_msg)
            
            # Robot Twist
            twist_msg = Twist()
            twist_msg.linear.x = float(robot_data['linear_vel'][0])
            twist_msg.linear.y = float(robot_data['linear_vel'][1])
            twist_msg.linear.z = float(robot_data['linear_vel'][2])
            twist_msg.angular.x = float(robot_data['angular_vel'][0])
            twist_msg.angular.y = float(robot_data['angular_vel'][1])
            twist_msg.angular.z = float(robot_data['angular_vel'][2])
            self.robot_twist_pub.publish(twist_msg)
        except Exception as e:
            self.get_logger().warn(f'Error publishing robot state: {e}')
    
    def publish_joint_states(self, joint_pos, joint_vel, joint_names):
        """Publish joint states"""
        try:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.name = joint_names
            msg.position = joint_pos.tolist() if isinstance(joint_pos, torch.Tensor) else list(joint_pos)
            msg.velocity = joint_vel.tolist() if isinstance(joint_vel, torch.Tensor) else list(joint_vel)
            msg.effort = [0.0] * len(joint_pos)
            self.joint_state_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Error publishing joint states: {e}')
    
    def publish_observations(self, obs):
        """Publish observations"""
        try:
            # Full policy observation
            if "policy" in obs:
                msg = Float32MultiArray()
                msg.data = obs["policy"][0].cpu().numpy().flatten().tolist()
                self.obs_pub.publish(msg)
            
            # Individual observations (if available in obs dict)
            # These would need to be extracted from the observation vector
            # For now, we'll extract from the full observation
            
        except Exception as e:
            self.get_logger().warn(f'Error publishing observations: {e}')
    
    def publish_rewards(self, reward, total_reward):
        """Publish rewards"""
        try:
            msg = Float32()
            msg.data = float(reward[0].item() if isinstance(reward, torch.Tensor) else reward)
            self.reward_pub.publish(msg)
            
            msg_total = Float32()
            msg_total.data = float(total_reward[0].item() if isinstance(total_reward, torch.Tensor) else total_reward)
            self.total_reward_pub.publish(msg_total)
        except Exception as e:
            self.get_logger().warn(f'Error publishing rewards: {e}')


def main():
    if not ROS2_AVAILABLE:
        print("ERROR: ROS2 not available. Cannot run ROS2 bridge.")
        print("Install ROS2 and cv-bridge first.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("DECEPTIONENV ROS2 BRIDGE")
    print("="*80)
    
    # Initialize ROS2
    rclpy.init()
    
    # Create environment
    print("\n[1/3] Creating environment...")
    if args.randomize:
        cfg = create_randomized_env_cfg(num_envs=args.num_envs, seed=42)
    else:
        cfg = WarehouseDeceptionTestEnvCfg()
        cfg.scene.num_envs = args.num_envs
    
    env = ManagerBasedRLEnv(cfg=cfg)
    print(f"✓ Environment created ({args.num_envs} envs)")
    
    # Add monitoring system
    monitoring_cfg = {"toggle_frequency": 100, "random_toggle": False}
    env.monitoring_system = MonitoringSystemManager(monitoring_cfg, env)
    
    # Create ROS2 bridge
    print("\n[2/3] Creating ROS2 bridge...")
    bridge = DeceptionEnvROS2Bridge(env, publish_rate=args.publish_rate)
    print("✓ ROS2 bridge ready")
    
    # Print topic information
    print("\n[3/3] Starting simulation...")
    print("\n" + "="*80)
    print("ROS2 TOPICS AVAILABLE:")
    print("="*80)
    print("\nTo view data in another terminal:")
    print("  source /opt/ros/jazzy/setup.bash  # or your ROS2 distro")
    print("  ros2 topic list")
    print("  ros2 topic echo /deception_env/camera/rgb")
    print("  ros2 topic echo /deception_env/robot/pose")
    print("  ros2 topic echo /deception_env/observations/policy")
    print("  ros2 topic hz /deception_env/camera/rgb  # Check publish rate")
    print("\nTo send actions (for VLA integration):")
    print("  ros2 topic pub /deception_env/actions std_msgs/msg/Float32MultiArray")
    print("    \"{data: [0.1, 0.0, 0.0]}\"  # Example action")
    print("="*80 + "\n")
    
    # Reset environment
    obs, _ = env.reset()
    total_reward = torch.zeros(args.num_envs, device=env.device)
    step = 0
    
    # Spin ROS2 node in background
    import threading
    def spin_ros():
        rclpy.spin(bridge)
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    try:
        while step < args.steps:
            # Get action (from ROS2 if available, else random)
            if bridge.action_received and bridge.ros2_action is not None:
                action = bridge.ros2_action.unsqueeze(0).to(env.device)
                bridge.action_received = False
            else:
                action = torch.randn((args.num_envs, env.action_manager.total_action_dim), device=env.device) * 0.1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Extract robot data
            try:
                robot = env.scene["robot"]
                robot_pos = robot.data.root_pos_w[0].cpu().numpy()
                robot_quat = robot.data.root_quat_w[0].cpu().numpy()
                robot_lin_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
                robot_ang_vel = robot.data.root_ang_vel_w[0].cpu().numpy()
                
                robot_data = {
                    'position': robot_pos,
                    'orientation': robot_quat,
                    'linear_vel': robot_lin_vel,
                    'angular_vel': robot_ang_vel
                }
                
                # Get joint states
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                joint_vel = robot.data.joint_vel[0].cpu().numpy()
                joint_names = robot.joint_names if hasattr(robot, 'joint_names') else [f"joint_{i}" for i in range(len(joint_pos))]
                
            except Exception as e:
                robot_data = None
                joint_pos = None
                joint_vel = None
                joint_names = []
            
            # Publish all data
            if step % max(1, int(100 / args.publish_rate)) == 0:  # Match publish rate
                # Camera
                if "camera" in obs and "rgb_image" in obs["camera"]:
                    bridge.publish_camera(obs["camera"]["rgb_image"])
                
                # Robot state
                if robot_data:
                    bridge.publish_robot_state(robot_data)
                    if joint_pos is not None:
                        bridge.publish_joint_states(joint_pos, joint_vel, joint_names)
                
                # Observations
                bridge.publish_observations(obs)
                
                # Rewards
                bridge.publish_rewards(reward, total_reward)
                
                # Done flag
                done_msg = Bool()
                done_msg.data = bool(terminated[0].item() or truncated[0].item())
                bridge.done_pub.publish(done_msg)
            
            # Reset if done
            if terminated.any() or truncated.any():
                reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                obs, _ = env.reset(reset_ids)
                total_reward[reset_ids] = 0
            
            step += 1
            
            if step % 100 == 0:
                print(f"Step {step}/{args.steps} | Reward: {total_reward[0].item():.2f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    
    print("\n" + "="*80)
    print("✓ Simulation stopped")
    print("="*80)
    
    # Cleanup
    bridge.destroy_node()
    rclpy.shutdown()
    env.close()
    app.close()


if __name__ == "__main__":
    main()

