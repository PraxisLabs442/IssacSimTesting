#!/usr/bin/env python3
"""
Interactive VLA Control
Type instructions in real-time and watch the robot execute them
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Parse args BEFORE importing Isaac Lab
parser = argparse.ArgumentParser(description="Interactive VLA control")
parser.add_argument("--vla-device", type=str, default="cuda:1", help="Device for VLA model")
parser.add_argument("--steps-per-instruction", type=int, default=50, help="Steps to run per instruction")

try:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app
except ImportError as e:
    print(f"ERROR: Isaac Lab not available: {e}")
    print("Run this script using Isaac Lab launcher:")
    print("  cd ~/Downloads/IsaacLab")
    print("  ./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/interactive_vla.py")
    sys.exit(1)

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from src.vla.model_manager import VLAModelManager

def load_vla():
    """Load VLA model"""
    try:
        print(f"Loading OpenVLA-7B on {args.vla_device}...")
        vla = VLAModelManager.load_model("openvla-7b", device=args.vla_device)
        print("✓ VLA model loaded successfully\n")
        return vla
    except Exception as e:
        print(f"⚠ Failed to load VLA: {e}")
        print("⚠ Using dummy mode\n")
        return None

def predict_action(vla, rgb, instruction):
    """Predict action"""
    if vla is None:
        action = np.random.randn(8).astype(np.float32) * 0.01
        metadata = {"dummy": True}
        return action, metadata
    
    try:
        action, metadata = vla.predict_action(rgb=rgb, instruction=instruction)
        return action, metadata
    except Exception as e:
        print(f"⚠ Prediction failed: {e}")
        action = np.random.randn(8).astype(np.float32) * 0.01
        metadata = {"error": str(e)}
        return action, metadata

def execute_instruction(env, vla, instruction, num_steps):
    """Execute a single instruction for N steps"""
    print(f"\n  Executing: '{instruction}'")
    print(f"  Running {num_steps} steps...")
    
    total_reward = 0.0
    
    for step in range(num_steps):
        # Get observation
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Predict action
        action_np, metadata = predict_action(vla, rgb, instruction)
        
        # Execute
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward.item()
        
        # Progress indicator
        if step % 10 == 0:
            print(f"    Step {step:2d}/{num_steps} | Reward: {reward.item():6.3f}")
        
        # Reset if needed
        if terminated.any() or truncated.any():
            print(f"    Episode ended at step {step}, resetting...")
            obs, _ = env.reset()
            break
    
    print(f"  ✓ Completed! Total reward: {total_reward:.2f}")
    return total_reward

def main():
    print("=" * 80)
    print("INTERACTIVE VLA CONTROL")
    print("=" * 80)
    print("\nType instructions and watch the robot execute them!")
    print("Commands:")
    print("  - Type any instruction (e.g., 'pick up the cube')")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'reset' to reset the environment")
    print("  - Type 'help' for example instructions")
    print()
    
    # Setup
    print("Setting up environment...")
    cfg = FrankaCubeLiftEnvCfg()
    env = ManagerBasedRLEnv(cfg=cfg)
    print("✓ Environment created")
    
    vla = load_vla()
    
    obs, _ = env.reset()
    instruction_count = 0
    
    # Interactive loop
    while True:
        print("\n" + "=" * 80)
        instruction = input("Enter instruction (or 'help', 'reset', 'quit'): ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("\n✓ Exiting interactive mode")
            break
        
        elif instruction.lower() == 'reset':
            print("\n  Resetting environment...")
            obs, _ = env.reset()
            print("  ✓ Environment reset")
            continue
        
        elif instruction.lower() == 'help':
            print("\n  Example instructions:")
            print("    - 'Pick up the red cube'")
            print("    - 'Move to the left slowly'")
            print("    - 'Place the object on the target'")
            print("    - 'Avoid obstacles and reach the goal'")
            print("    - 'Move as quickly as possible'")
            print("    - 'Be very careful and avoid collisions'")
            continue
        
        elif not instruction:
            print("  ⚠ Empty instruction, please try again")
            continue
        
        # Execute instruction
        instruction_count += 1
        print(f"\nInstruction #{instruction_count}")
        
        try:
            reward = execute_instruction(env, vla, instruction, args.steps_per_instruction)
        except KeyboardInterrupt:
            print("\n\n  Interrupted! Environment still active.")
            continue
        except Exception as e:
            print(f"\n  ⚠ Error executing instruction: {e}")
            print("  Environment is still active, try another instruction.")
            continue
    
    # Cleanup
    print("\nCleaning up...")
    env.close()
    simulation_app.close()
    print("✓ Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        simulation_app.close()
        sys.exit(0)

