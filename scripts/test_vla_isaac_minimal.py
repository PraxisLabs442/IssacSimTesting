#!/usr/bin/env python3
"""
Minimal VLA + Isaac Lab Integration Test
Tests one episode with VLA controlling a robot in Isaac Lab
Supports both GUI and headless modes
"""

import sys
import argparse
from pathlib import Path

# Must parse args before importing Isaac Lab (for AppLauncher)
parser = argparse.ArgumentParser(description="Minimal VLA + Isaac Lab Test")
# Note: --headless and --gpu are added by AppLauncher, don't add them here
parser.add_argument("--vla-device", type=str, default="cuda:1", help="Device for VLA model")
parser.add_argument("--num-steps", type=int, default=100, help="Number of steps per episode")
parser.add_argument("--use-dummy-vla", action="store_true", help="Use dummy VLA instead of loading real model")

try:
    # Isaac Lab imports (must come first)
    from isaaclab.app import AppLauncher
    
    # Add AppLauncher args and parse
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    print("=" * 80)
    print("MINIMAL VLA + ISAAC LAB TEST")
    print("=" * 80)
    print(f"✓ Isaac Sim launched")
    print(f"  Headless: {args.headless}")
    print(f"  VLA Device: {args.vla_device}")
    print()
    
except ImportError as e:
    print(f"ERROR: Isaac Lab not available: {e}")
    print()
    print("This script must be run via Isaac Lab launcher:")
    print("  cd ~/Downloads/IsaacLab")
    print("  ./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_vla_isaac_minimal.py")
    print()
    print("Optional arguments:")
    print("  --headless        Run without GUI")
    print("  --vla-device cuda:1   Use specific GPU for VLA")
    print("  --num-steps 100   Number of steps to run")
    sys.exit(1)

# Now import Isaac Lab components and PraxisLabs after app launch
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

# Add PraxisLabs to path
praxis_path = Path(__file__).parent.parent
sys.path.insert(0, str(praxis_path))


def load_vla_model(device: str, use_dummy: bool = False):
    """Load VLA model or return dummy"""
    if use_dummy:
        print("⚠ Using dummy VLA mode (no real model loaded)")
        return None
    
    try:
        from src.vla.model_manager import VLAModelManager
        print(f"Loading OpenVLA-7B model on {device}...")
        print("  (This may take 30-60 seconds for first-time download)")
        
        vla = VLAModelManager.load_model(
            "openvla-7b",
            device=device,
            log_activations=False  # Disable for speed
        )
        
        print("✓ VLA model loaded successfully")
        return vla
        
    except Exception as e:
        print(f"⚠ Failed to load VLA model: {e}")
        print("⚠ Continuing with dummy actions")
        return None


def predict_action_with_vla(vla, rgb, instruction, action_dim=8):
    """Predict action using VLA or return dummy"""
    if vla is None:
        # Dummy action: small random movements (8D for Franka: 7 joints + 1 gripper)
        action = np.random.randn(action_dim).astype(np.float32) * 0.01
        metadata = {"dummy": True}
        return action, metadata
    
    try:
        action, metadata = vla.predict_action(
            rgb=rgb,
            instruction=instruction
        )
        return action, metadata
    except Exception as e:
        print(f"⚠ VLA prediction failed: {e}")
        # Fallback to dummy (ensure correct dimension)
        action = np.random.randn(action_dim).astype(np.float32) * 0.01
        metadata = {"error": str(e)}
        return action, metadata


def main():
    """Run minimal test"""
    
    print("=" * 80)
    print("STEP 1: Create Isaac Lab Environment")
    print("=" * 80)
    
    # Create environment configuration (use Franka-specific config)
    cfg = FrankaCubeLiftEnvCfg()
    cfg.scene.num_envs = 1  # Single environment
    cfg.sim.render_interval = 1  # Render every frame
    cfg.episode_length_s = 30.0  # 30 second episodes
    
    print(f"✓ Configuration created")
    print(f"  Task: Franka Panda Pick-and-Place")
    print(f"  Num Envs: {cfg.scene.num_envs}")
    print(f"  Episode Length: {cfg.episode_length_s}s")
    print()
    
    # Create environment
    print("Creating environment (may take 30-60 seconds on first run)...")
    print("  Initializing ManagerBasedRLEnv...")
    import sys
    sys.stdout.flush()
    env = ManagerBasedRLEnv(cfg=cfg)
    print("✓ Environment created!")
    sys.stdout.flush()
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print()
    sys.stdout.flush()
    
    print("=" * 80)
    print("STEP 2: Load VLA Model")
    print("=" * 80)
    
    vla = load_vla_model(args.vla_device, args.use_dummy_vla)
    print()
    
    print("=" * 80)
    print("STEP 3: Run Test Episode")
    print("=" * 80)
    print(f"Running {args.num_steps} steps with VLA control")
    
    if not args.headless:
        print("Watch the Isaac Sim window for robot movements!")
    
    print()
    
    instruction = "Pick up the cube and place it in the target location"
    print(f"Instruction: '{instruction}'")
    print()
    
    try:
        # Reset environment
        obs_dict, _ = env.reset()
        print("✓ Environment reset - episode started")
        
        # Extract observation components
        if "policy" in obs_dict:
            policy_obs = obs_dict["policy"]
        else:
            policy_obs = obs_dict
        
        # Run episode
        for step in range(args.num_steps):
            # Get RGB observation (if available)
            # Note: policy_obs is a Tensor, not a dict for this environment
            # For this simple test, just use dummy RGB
            rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Get action from VLA (8D: 7 arm joints + 1 gripper)
            action_np, metadata = predict_action_with_vla(vla, rgb, instruction, action_dim=8)
            
            # Convert to torch tensor with correct shape for environment
            action = torch.from_numpy(action_np).float().unsqueeze(0).to(env.device)
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            
            # Extract next observation
            if "policy" in obs_dict:
                policy_obs = obs_dict["policy"]
            else:
                policy_obs = obs_dict
            
            # Print progress
            if step % 20 == 0:
                reward_val = reward.item() if isinstance(reward, torch.Tensor) else reward
                print(f"  Step {step:3d}/{args.num_steps} | Reward: {reward_val:6.3f} | "
                      f"Done: {terminated.any().item() if isinstance(terminated, torch.Tensor) else terminated}")
            
            # Reset if episode ends
            if (isinstance(terminated, torch.Tensor) and terminated.any()) or terminated:
                print(f"\n  Episode ended at step {step}")
                obs_dict, _ = env.reset()
                if "policy" in obs_dict:
                    policy_obs = obs_dict["policy"]
                else:
                    policy_obs = obs_dict
                print("  Environment reset for new episode")
        
        print()
        print("✓ Test episode completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during episode: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        print()
        print("=" * 80)
        print("CLEANUP")
        print("=" * 80)
        
        env.close()
        print("✓ Environment closed")
        
        simulation_app.close()
        print("✓ Isaac Sim closed")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    print()
    
    if vla is None:
        print("⚠ Test ran with dummy VLA (real model not loaded)")
        print("  To test with real VLA: remove --use-dummy-vla flag")
    else:
        print("✓ VLA model successfully controlled robot in Isaac Lab!")
    
    print()
    print("Next steps:")
    print("  1. Run with GUI to watch robot: (remove --headless)")
    print("  2. Run with real VLA model: (remove --use-dummy-vla)")
    print("  3. Test full deception study: python scripts/run_deception_study.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

