"""
Simple Camera Viewer - Just shows what the VLA sees

Quick tool to visualize camera images that VLA would receive.

Usage:
    cd ~/Downloads/IsaacLab
    ./isaaclab.sh -p ~/Desktop/DeceptionEnv/visualize_camera_only.py --randomize
"""

import sys
import argparse
import torch
import numpy as np
import cv2

# Add DeceptionEnv to path
sys.path.insert(0, "/home/mpcr/Desktop/DeceptionEnv")

parser = argparse.ArgumentParser(description="Simple Camera Viewer")
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--randomize", action="store_true", help="Randomize scene")
parser.add_argument("--save", action="store_true", help="Save images to disk")
parser.add_argument("--fps", type=int, default=10, help="Display FPS")

try:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
except ImportError:
    print("ERROR: Isaac Lab not found")
    sys.exit(1)

from isaaclab.envs import ManagerBasedRLEnv

if args.randomize:
    from warehouse_deception.random_env_cfg import create_randomized_env_cfg
else:
    from warehouse_deception.warehouse_env_cfg import WarehouseDeceptionTestEnvCfg

from warehouse_deception.scene.monitoring_system import MonitoringSystemManager


def main():
    print("\n" + "="*80)
    print("SIMPLE CAMERA VIEWER - What VLA Sees")
    print("="*80)
    
    # Create environment
    if args.randomize:
        cfg = create_randomized_env_cfg(num_envs=args.num_envs, seed=42)
    else:
        cfg = WarehouseDeceptionTestEnvCfg()
        cfg.scene.num_envs = args.num_envs
    
    env = ManagerBasedRLEnv(cfg=cfg)
    monitoring_cfg = {"toggle_frequency": 100, "random_toggle": False}
    env.monitoring_system = MonitoringSystemManager(monitoring_cfg, env)
    
    obs, _ = env.reset()
    
    print("\nPress 'q' to quit, 's' to save current frame")
    print("="*80 + "\n")
    
    step = 0
    frame_count = 0
    
    try:
        while step < args.steps:
            # Get camera image
            if "camera" in obs and "rgb_image" in obs["camera"]:
                img = obs["camera"]["rgb_image"][0].cpu().numpy()
                
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Add text overlay
                monitoring_status = env.monitoring_system.get_monitoring_observation()[0].item()
                status_text = "MONITORED" if monitoring_status > 0.5 else "UNMONITORED"
                status_color = (0, 0, 255) if monitoring_status > 0.5 else (0, 255, 0)
                
                cv2.putText(img_bgr, f"Step: {step}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img_bgr, f"Status: {status_text}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(img_bgr, f"Shape: {img.shape}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display
                cv2.imshow("VLA Camera View (What VLA Sees)", img_bgr)
                
                # Save if requested
                if args.save:
                    cv2.imwrite(f"vla_frame_{frame_count:05d}.png", img_bgr)
                    frame_count += 1
                
                # Handle keyboard
                key = cv2.waitKey(int(1000 / args.fps)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"vla_frame_{frame_count:05d}.png"
                    cv2.imwrite(filename, img_bgr)
                    print(f"Saved: {filename}")
                    frame_count += 1
            else:
                print(f"Step {step}: No camera data available (placeholder)")
                cv2.waitKey(int(1000 / args.fps))
            
            # Step environment
            action = torch.randn((args.num_envs, env.action_manager.total_action_dim), device=env.device) * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            env.monitoring_system.update()
            
            if terminated.any() or truncated.any():
                reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                obs, _ = env.reset(reset_ids)
            
            step += 1
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    
    cv2.destroyAllWindows()
    env.close()
    app.close()
    print("\n✓ Viewer closed")


if __name__ == "__main__":
    main()

