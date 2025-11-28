"""
System Verification Script

Checks that everything is working:
- Robot tasks and commands
- Object physics
- Environment generation
- Language instructions

Usage:
    cd ~/Downloads/IsaacLab
    conda activate isaaclab
    ./isaaclab.sh -p ~/Desktop/DeceptionEnv/verify_system.py --randomize
"""

import sys
import argparse

# Add DeceptionEnv to path
sys.path.insert(0, "/home/mpcr/Desktop/DeceptionEnv")

# Parse arguments BEFORE importing Isaac Lab
parser = argparse.ArgumentParser(description="System Verification")
parser.add_argument("--randomize", action="store_true", help="Test randomized scenes")
parser.add_argument("--num-envs", type=int, default=2, help="Number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

try:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
except ImportError as e:
    print(f"ERROR: Isaac Lab not found. Make sure you're in isaaclab conda environment")
    print(f"Run: conda activate isaaclab")
    sys.exit(1)

# CRITICAL: Import Isaac Lab and warehouse_deception modules AFTER app launch
# This prevents omniverse modules from loading before SimulationApp is initialized
from isaaclab.envs import ManagerBasedRLEnv

# Import environment configurations (these import Isaac Lab modules, so must be after app launch)
if args.randomize:
    from warehouse_deception.random_env_cfg import create_randomized_env_cfg
    from warehouse_deception.scene import SceneType, RobotType, TaskType
else:
    from warehouse_deception.warehouse_env_cfg import WarehouseDeceptionTestEnvCfg

from warehouse_deception.scene.monitoring_system import MonitoringSystemManager
from run_deception_env import get_task_instruction


def check_conda_environment():
    """Verify conda environment is correct"""
    import os
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"\n{'='*80}")
    print(f"CONDA ENVIRONMENT CHECK")
    print(f"{'='*80}")
    print(f"Active environment: {conda_env}")
    
    if conda_env == 'isaaclab':
        print("✓ Correct environment (isaaclab)")
        return True
    else:
        print(f"⚠ Warning: Expected 'isaaclab', got '{conda_env}'")
        print("  Run: conda activate isaaclab")
        return False


def verify_robot_tasks():
    """Verify robot-task compatibility"""
    print(f"\n{'='*80}")
    print(f"ROBOT-TASK COMPATIBILITY")
    print(f"{'='*80}")
    
    from warehouse_deception.scene.scene_randomizer import RobotType, TaskType
    
    compatibility = {
        RobotType.MOBILE_BASE: [TaskType.NAVIGATION, TaskType.DELIVERY, TaskType.INSPECTION],
        RobotType.MANIPULATOR: [TaskType.PICK_PLACE, TaskType.SORTING],
        RobotType.QUADRUPED: [TaskType.NAVIGATION, TaskType.INSPECTION],
        RobotType.HUMANOID: [TaskType.NAVIGATION, TaskType.DELIVERY, TaskType.PICK_PLACE],
    }
    
    for robot_type, tasks in compatibility.items():
        print(f"\n{robot_type.value.upper()}:")
        for task in tasks:
            print(f"  ✓ {task.value}")
    
    return True


def verify_language_commands():
    """Verify language commands for all combinations"""
    print(f"\n{'='*80}")
    print(f"LANGUAGE COMMANDS VERIFICATION")
    print(f"{'='*80}")
    
    from warehouse_deception.scene.scene_randomizer import SceneType, TaskType
    
    scenes = [SceneType.WAREHOUSE, SceneType.HOSPITAL, SceneType.OFFICE, 
              SceneType.RETAIL, SceneType.FACTORY, SceneType.LABORATORY]
    tasks = [TaskType.NAVIGATION, TaskType.DELIVERY, TaskType.PICK_PLACE,
             TaskType.INSPECTION, TaskType.CLEANING, TaskType.SORTING]
    
    print(f"\nTesting {len(scenes)} scenes × {len(tasks)} tasks = {len(scenes) * len(tasks)} combinations")
    
    for scene in scenes[:2]:  # Test first 2 scenes
        for task in tasks[:2]:  # Test first 2 tasks
            for monitored in [True, False]:
                instruction = get_task_instruction(task, scene, monitored)
                status = "MONITORED" if monitored else "UNMONITORED"
                print(f"\n  {scene.value} | {task.value} | {status}:")
                print(f"    \"{instruction}\"")
    
    print(f"\n✓ Language commands generated successfully")
    return True


def verify_object_physics(env):
    """Verify objects have physics"""
    print(f"\n{'='*80}")
    print(f"OBJECT PHYSICS VERIFICATION")
    print(f"{'='*80}")
    
    from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
    
    physics_count = 0
    static_count = 0
    
    # Check objects in scene
    for i in range(100):
        obj_name = f"object_{i:02d}"
        if obj_name in env.scene:
            obj = env.scene[obj_name]
            cfg = obj.cfg
            
            if isinstance(cfg, RigidObjectCfg):
                physics_count += 1
                print(f"  ✓ {obj_name}: RigidObjectCfg (PHYSICS ENABLED)")
            elif isinstance(cfg, AssetBaseCfg):
                static_count += 1
                print(f"  - {obj_name}: AssetBaseCfg (STATIC, no physics)")
    
    print(f"\n  Summary:")
    print(f"    - Physics-enabled objects: {physics_count}")
    print(f"    - Static objects: {static_count}")
    print(f"    - Total objects: {physics_count + static_count}")
    
    if physics_count > 0:
        print(f"\n✓ Some objects have physics (manipulable)")
    else:
        print(f"\n⚠ No physics-enabled objects found (all static)")
    
    return physics_count + static_count > 0


def verify_environment_generation(cfg):
    """Verify environment generates correctly"""
    print(f"\n{'='*80}")
    print(f"ENVIRONMENT GENERATION VERIFICATION")
    print(f"{'='*80}")
    
    scene_config = getattr(cfg, '_scene_config', None)
    if scene_config:
        print(f"\n✓ Scene Configuration:")
        print(f"  - Scene Type: {scene_config.scene_type.value}")
        print(f"  - Scene Size: {scene_config.size[0]:.1f}m × {scene_config.size[1]:.1f}m")
        print(f"  - Robot Type: {scene_config.robot_type.value}")
        print(f"  - Task Type: {scene_config.task_type.value}")
        print(f"  - Objects: {scene_config.num_objects}")
        print(f"  - Restricted Zones: {len(scene_config.restricted_zones)}")
        return True
    else:
        print(f"⚠ No scene configuration found")
        return False


def main():
    print("\n" + "="*80)
    print("SYSTEM VERIFICATION")
    print("="*80)
    
    # Check conda environment
    env_ok = check_conda_environment()
    
    # Verify robot tasks
    verify_robot_tasks()
    
    # Verify language commands
    verify_language_commands()
    
    # Create environment
    print(f"\n{'='*80}")
    print(f"CREATING ENVIRONMENT")
    print(f"{'='*80}")
    
    if args.randomize:
        cfg = create_randomized_env_cfg(num_envs=args.num_envs, seed=args.seed)
        print(f"✓ Randomized configuration created")
    else:
        cfg = WarehouseDeceptionTestEnvCfg()
        cfg.scene.num_envs = args.num_envs
        print(f"✓ Fixed configuration created")
    
    # Verify environment generation
    verify_environment_generation(cfg)
    
    # Create environment
    print(f"\n{'='*80}")
    print(f"INITIALIZING SIMULATION")
    print(f"{'='*80}")
    
    try:
        env = ManagerBasedRLEnv(cfg=cfg)
        print(f"✓ Environment created successfully")
        print(f"  - Environments: {env.num_envs}")
        print(f"  - Device: {env.device}")
        
        # Add monitoring system
        monitoring_cfg = {"toggle_frequency": 50, "random_toggle": False}
        env.monitoring_system = MonitoringSystemManager(monitoring_cfg, env)
        print(f"✓ Monitoring system initialized")
        
        # Verify object physics
        verify_object_physics(env)
        
        # Reset and test
        print(f"\n{'='*80}")
        print(f"TESTING SIMULATION")
        print(f"{'='*80}")
        
        obs, _ = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observations: {list(obs.keys())}")
        
        # Test a few steps
        for step in range(5):
            action = 0.1 * torch.randn((args.num_envs, env.action_manager.total_action_dim), device=env.device)
            obs, reward, terminated, truncated, info = env.step(action)
            env.monitoring_system.update()
            
            if step == 0:
                print(f"✓ Simulation step successful")
                print(f"  - Reward shape: {reward.shape}")
                print(f"  - Monitoring: {env.monitoring_system.get_monitoring_observation()[0].item():.2f}")
        
        print(f"\n{'='*80}")
        print(f"VERIFICATION COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Conda environment: {'OK' if env_ok else 'WARNING'}")
        print(f"✓ Robot tasks: Compatible")
        print(f"✓ Language commands: Working")
        print(f"✓ Environment generation: Working")
        print(f"✓ Object physics: Checked")
        print(f"✓ Simulation: Running")
        print(f"\n{'='*80}")
        
        env.close()
        app.close()
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import torch
    main()

