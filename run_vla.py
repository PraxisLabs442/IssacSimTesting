"""
Simple script to run OpenVLA model in Isaac Sim

Usage:
    cd ~/Downloads/IsaacLab
    ./isaaclab.sh -p ~/Desktop/PraxisLabs/run_vla.py
"""

import sys
import argparse
import torch
import numpy as np

# Parse args BEFORE importing Isaac Lab
parser = argparse.ArgumentParser(description="Run VLA in Isaac Sim")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
parser.add_argument("--vla-device", type=str, default="cuda:1", help="GPU for VLA model")

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
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

# Add project to path
sys.path.insert(0, "/home/mpcr/Desktop/PraxisLabs")
from src.vla.openvla import OpenVLA


def main():
    print("\n" + "="*80)
    print("VLA + ISAAC SIM")
    print("="*80)
    
    # Create Isaac Lab environment
    print("\n[1/3] Creating Isaac Sim environment...")
    cfg = FrankaCubeLiftEnvCfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=cfg)
    print("✓ Environment created")
    
    # Load VLA model
    print(f"\n[2/3] Loading OpenVLA model on {args.vla_device}...")
    vla = OpenVLA(device=args.vla_device)
    print("✓ VLA loaded")
    
    # Run simulation
    print(f"\n[3/3] Running {args.steps} simulation steps...")
    print("Instruction: 'Pick up the cube and place it in the target location'\n")
    
    obs, _ = env.reset()
    total_reward = 0.0
    
    for step in range(args.steps):
        # Get RGB image (dummy for now - you can add camera later)
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Get action from VLA
        action_np = vla.predict_action(
            rgb_image=rgb,
            instruction="Pick up the cube and place it in the target location"
        )
        
        # Convert to tensor for Isaac Lab
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(env.device)
        
        # Step simulation
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.item()
        
        # Print progress
        if step % 20 == 0:
            print(f"  Step {step:3d}/{args.steps} | Reward: {reward.item():6.3f} | Total: {total_reward:7.2f}")
        
        # Reset if episode ends
        if terminated.any() or truncated.any():
            print(f"  → Episode ended at step {step}, resetting...")
            obs, _ = env.reset()
            total_reward = 0.0
    
    print("\n" + "="*80)
    print("✓ Simulation complete!")
    print("="*80)
    
    # Cleanup
    env.close()
    app.close()


if __name__ == "__main__":
    main()

