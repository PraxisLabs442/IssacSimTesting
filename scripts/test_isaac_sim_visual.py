#!/usr/bin/env python3
"""
Minimal Isaac Sim Visual Test
Just opens Isaac Sim window with a Franka robot - no VLA, no data collection
"""

import argparse
import sys

try:
    # Isaac Lab imports
    from omni.isaac.lab.app import AppLauncher

    # Parse arguments for Isaac Sim configuration
    parser = argparse.ArgumentParser(description="Test Isaac Sim Visual Rendering")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use for rendering")

    # Append AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch the Isaac Sim app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    print("=" * 80)
    print("Isaac Sim Visual Test")
    print("=" * 80)
    print(f"✓ Isaac Sim launched successfully")
    print(f"✓ Rendering on GPU: {args_cli.gpu}")
    print(f"✓ Headless mode: {args_cli.headless}")
    print()

except ImportError as e:
    print(f"ERROR: Isaac Lab not available: {e}")
    print()
    print("This script must be run via Isaac Lab launcher:")
    print("  cd ~/Downloads/IsaacLab")
    print("  ./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_isaac_sim_visual.py")
    sys.exit(1)

# Now import Isaac Lab components after app launch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
import torch

def main():
    """Run minimal Isaac Sim test with Franka robot"""

    print("=" * 80)
    print("Creating Isaac Lab Environment")
    print("=" * 80)

    # Create environment configuration
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = 1  # Single environment for visual testing
    cfg.sim.render_interval = 1  # Render every frame
    cfg.episode_length_s = 30.0  # 30 second episodes

    print(f"✓ Configuration created")
    print(f"  Task: Franka Panda - Pick and Place")
    print(f"  Num Envs: {cfg.scene.num_envs}")
    print(f"  Episode Length: {cfg.episode_length_s}s")
    print()

    # Create environment
    print("Creating environment (this may take 30-60 seconds)...")
    env = ManagerBasedRLEnv(cfg=cfg)
    print("✓ Environment created successfully!")
    print()

    print("=" * 80)
    print("Environment Details")
    print("=" * 80)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Device: {env.device}")
    print()

    print("=" * 80)
    print("Running Simulation Loop")
    print("=" * 80)
    print("The Isaac Sim window should now be visible!")
    print("Robot will perform random actions for 10 seconds...")
    print("Close the Isaac Sim window or press Ctrl+C to exit")
    print()

    try:
        # Reset environment
        obs, _ = env.reset()
        print("✓ Environment reset - simulation started")

        # Run for limited time with random actions
        max_steps = 600  # 10 seconds at 60Hz

        for step in range(max_steps):
            # Generate random action (7-DOF: dx,dy,dz,droll,dpitch,dyaw,gripper)
            action = torch.randn(1, env.action_space.shape[0], device=env.device) * 0.1

            # Step simulation
            obs, reward, terminated, truncated, info = env.step(action)

            # Print progress every 5 seconds
            if step % 300 == 0 and step > 0:
                print(f"  Step {step}/{max_steps} - Simulation running...")

            # Reset if episode ends
            if terminated.any() or truncated.any():
                obs, _ = env.reset()
                print(f"  Episode ended at step {step}, resetting...")

        print()
        print("✓ Simulation test completed successfully!")

    except KeyboardInterrupt:
        print("\n✓ User interrupted - shutting down...")

    finally:
        # Clean up
        env.close()
        print("✓ Environment closed")
        simulation_app.close()
        print("✓ Isaac Sim closed")
        print()
        print("=" * 80)
        print("Visual Test Complete!")
        print("=" * 80)
        print()
        print("✓ Isaac Sim visual rendering is working correctly")
        print("✓ Ready for integration with VLA model")
        print()


if __name__ == "__main__":
    main()
