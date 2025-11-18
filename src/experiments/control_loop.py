"""
Main Control Loop
Connects Isaac Lab environment with VLA model
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from src.vla.vla_loader import VLALoader, VLAConfig
from src.environment.isaac_lab_env import IsaacLabEnvironment, EnvironmentConfig
from src.data_logging.trajectory_logger import TrajectoryLogger, LogConfig
from src.metrics.safety_metrics import SafetyMetricsCalculator, SafetyThresholds

logger = logging.getLogger(__name__)


class ControlLoop:
    """
    Main control loop integrating:
    - Isaac Lab environment
    - VLA model
    - Trajectory logging
    - Safety metrics
    """

    def __init__(
        self,
        vla: VLALoader,
        env: IsaacLabEnvironment,
        logger_obj: TrajectoryLogger,
        safety_calculator: SafetyMetricsCalculator
    ):
        self.vla = vla
        self.env = env
        self.logger = logger_obj
        self.safety = safety_calculator

        self.episode_count = 0

    def run_episode(
        self,
        phase: str,
        instruction: str,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single episode

        Args:
            phase: Experiment phase (baseline, monitoring, etc.)
            instruction: Task instruction for VLA
            max_steps: Maximum steps (uses env default if None)

        Returns:
            episode_summary: Dict with metrics and outcomes
        """
        # Reset environment
        observation = self.env.reset(phase=phase)
        done = False
        step = 0

        # Reset safety tracking
        self.safety.reset()

        # Episode metrics
        total_reward = 0.0
        trajectory = []

        max_steps = max_steps or self.env.config.max_episode_length

        logger.info(f"Starting episode {self.episode_count} | Phase: {phase} | Instruction: {instruction}")

        while not done and step < max_steps:
            # Get RGB image
            rgb_image = observation["rgb"]

            # VLA predicts action
            action, vla_metadata = self.vla.predict_action(
                rgb_image=rgb_image,
                task_instruction=instruction,
                phase=phase
            )

            # Step environment
            next_observation, reward, done, info = self.env.step(action)

            # Compute safety metrics
            safety_metrics = self.safety.compute_step_metrics(info, observation)

            # Augment info with safety metrics
            info.update(safety_metrics)

            # Log step
            self.logger.log_step(
                step=step,
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
                phase=phase,
                instruction=instruction,
                metadata=vla_metadata
            )

            # Store for trajectory analysis
            trajectory.append({
                "step": step,
                "observation": observation,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info
            })

            # Update
            observation = next_observation
            total_reward += reward
            step += 1

        # Compute episode metrics
        episode_metrics = self._compute_episode_metrics(trajectory, total_reward, phase)

        # Log episode
        self.logger.end_episode(self.episode_count, episode_metrics)

        self.episode_count += 1

        logger.info(f"Episode {self.episode_count - 1} completed: {episode_metrics}")

        return episode_metrics

    def _compute_episode_metrics(
        self,
        trajectory: list,
        total_reward: float,
        phase: str
    ) -> Dict[str, Any]:
        """Compute episode-level metrics"""
        # Safety metrics
        safety_metrics = self.safety.compute_episode_metrics(trajectory)

        # Task metrics
        final_info = trajectory[-1]["info"] if trajectory else {}
        success = final_info.get("success", False)

        metrics = {
            "phase": phase,
            "num_steps": len(trajectory),
            "total_reward": float(total_reward),
            "success": success,
            **safety_metrics
        }

        return metrics


def run_single_episode(
    vla: VLALoader,
    env: IsaacLabEnvironment,
    logger_obj: TrajectoryLogger,
    safety_calculator: SafetyMetricsCalculator,
    phase: str,
    instruction: str
) -> Dict[str, Any]:
    """
    Convenience function to run a single episode

    Args:
        vla: VLA model loader
        env: Isaac Lab environment
        logger_obj: Trajectory logger
        safety_calculator: Safety metrics calculator
        phase: Experiment phase
        instruction: Task instruction

    Returns:
        episode_metrics: Episode summary
    """
    loop = ControlLoop(vla, env, logger_obj, safety_calculator)
    return loop.run_episode(phase, instruction)
