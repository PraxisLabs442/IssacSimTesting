#!/usr/bin/env python3
"""
Simple example showing basic usage of the Isaac Lab + VLA integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vla.vla_loader import VLALoader, VLAConfig
from src.environment.isaac_lab_env import IsaacLabEnvironment, EnvironmentConfig
from src.logging.trajectory_logger import TrajectoryLogger, LogConfig
from src.metrics.safety_metrics import SafetyMetricsCalculator, SafetyThresholds
from src.experiments.control_loop import run_single_episode


def main():
    print("Simple Isaac Lab + VLA Example")
    print("=" * 60)

    # Create VLA model
    print("\n1. Creating VLA model...")
    vla_config = VLAConfig(
        model_name="openvla/openvla-7b",
        device="cuda:0",
        log_activations=False
    )
    vla = VLALoader(vla_config)

    # Create environment
    print("2. Creating Isaac Lab environment...")
    env_config = EnvironmentConfig(
        task_name="PickAndPlace",
        num_envs=1,
        render=False
    )
    env = IsaacLabEnvironment(env_config)

    # Create logger
    print("3. Creating logger...")
    log_config = LogConfig(
        log_dir="logs",
        format="json",
        log_activations=False
    )
    logger = TrajectoryLogger(log_config, "simple_example")

    # Create safety calculator
    print("4. Creating safety calculator...")
    safety_thresholds = SafetyThresholds()
    safety = SafetyMetricsCalculator(safety_thresholds)

    # Run a single episode
    print("\n5. Running episode...")
    print("-" * 60)

    metrics = run_single_episode(
        vla=vla,
        env=env,
        logger_obj=logger,
        safety_calculator=safety,
        phase="baseline",
        instruction="pick up the red cube and place it in the bin"
    )

    # Print results
    print("\n6. Episode Results:")
    print("-" * 60)
    print(f"  Success: {metrics['success']}")
    print(f"  Steps: {metrics['num_steps']}")
    print(f"  Reward: {metrics['total_reward']:.2f}")
    print(f"  Collisions: {metrics['collision_count']}")
    print(f"  Safety Score: {metrics['safety_score']:.2f}")

    # Cleanup
    env.close()
    logger.close()

    print("\n" + "=" * 60)
    print("Example completed!")


if __name__ == "__main__":
    main()
