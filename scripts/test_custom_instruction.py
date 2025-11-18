#!/usr/bin/env python3
"""
Test VLA with Custom Instruction
Allows you to specify any instruction and watch the robot execute it
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Parse args BEFORE importing Isaac Lab
parser = argparse.ArgumentParser(description="Test VLA with custom instruction")
parser.add_argument("--instruction", type=str, required=True, help="Custom instruction for the robot")
parser.add_argument("--vla-device", type=str, default="cuda:1", help="Device for VLA model")
parser.add_argument("--num-steps", type=int, default=100, help="Number of steps to run")
parser.add_argument("--use-dummy-vla", action="store_true", help="Use dummy VLA (for testing)")

try:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app
except ImportError as e:
    print(f"ERROR: Isaac Lab not available: {e}")
    print("Run this script using Isaac Lab launcher:")
    print("  cd ~/Downloads/IsaacLab")
    print("  ./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py --instruction 'Pick up the cube'")
    sys.exit(1)

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from src.vla.model_manager import VLAModelManager

def load_vla(use_dummy=False):
    """Load VLA model or dummy"""
    if use_dummy:
        print("⚠ Using dummy VLA model (random actions)")
        return None
    
    try:
        print(f"Loading OpenVLA-7B on {args.vla_device}...")
        vla = VLAModelManager.load_model("openvla-7b", device=args.vla_device)
        print("✓ VLA model loaded successfully")
        return vla
    except Exception as e:
        print(f"⚠ Failed to load VLA: {e}")
        print("⚠ Using dummy mode")
        return None

def predict_action(vla, rgb, instruction):
    """Predict action with VLA or dummy"""
    if vla is None:
        # Dummy action
        action = np.random.randn(8).astype(np.float32) * 0.01
        metadata = {"dummy": True}
        return action, metadata
    
    try:
        action, metadata = vla.predict_action(rgb=rgb, instruction=instruction)
        return action, metadata
    except Exception as e:
        print(f"⚠ VLA prediction failed: {e}")
        action = np.random.randn(8).astype(np.float32) * 0.01
        metadata = {"error": str(e)}
        return action, metadata

def main():
    print("=" * 80)
    print("CUSTOM INSTRUCTION TEST")
    print("=" * 80)
    print(f"\nInstruction: '{args.instruction}'")
    print(f"Steps: {args.num_steps}")
    print(f"VLA Device: {args.vla_device}")
    print(f"Headless: {args.headless}")
    print()
    
    # Create environment
    print("Creating Isaac Lab environment...")
    cfg = FrankaCubeLiftEnvCfg()
    env = ManagerBasedRLEnv(cfg=cfg)
    print("✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print()
    
    # Load VLA
    vla = load_vla(use_dummy=args.use_dummy_vla)
    print()
    
    # Run episode
    print("=" * 80)
    print("RUNNING EPISODE")
    print("=" * 80)
    print()
    
    obs, _ = env.reset()
    total_reward = 0.0
    episode_step = 0
    
    for step in range(args.num_steps):
        # Get RGB observation (dummy for now - would use actual camera in full implementation)
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # VLA predicts action with YOUR instruction
        action_np, metadata = predict_action(vla, rgb, args.instruction)
        
        # Convert to tensor and add batch dimension
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(env.device)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward.item()
        episode_step += 1
        
        # Print progress
        if step % 20 == 0 or step == args.num_steps - 1:
            print(f"Step {step:3d}/{args.num_steps} | "
                  f"Reward: {reward.item():6.3f} | "
                  f"Total: {total_reward:6.2f} | "
                  f"Done: {terminated.any() or truncated.any()}")
        
        # Reset if episode ended
        if terminated.any() or truncated.any():
            print(f"\n✓ Episode ended at step {episode_step}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Success: {info.get('success', False)}")
            print("\nResetting environment...\n")
            
            obs, _ = env.reset()
            total_reward = 0.0
            episode_step = 0
    
    # Final summary
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nInstruction: '{args.instruction}'")
    print(f"Steps executed: {args.num_steps}")
    print(f"Final episode reward: {total_reward:.2f}")
    print()
    
    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()

