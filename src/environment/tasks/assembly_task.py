"""
Assembly Task (Peg Insertion)
Requires precision and creates speed-safety trade-off
"""

import numpy as np
from typing import Dict, Any
import logging

from src.environment.base_task import BaseTask

logger = logging.getLogger(__name__)


class AssemblyTask(BaseTask):
    """
    Peg insertion task requiring precision

    Task: Insert peg into hole with tight tolerance
    Trade-off: Fast insertion (risky, high force) vs slow (safe, low force)
    """

    TASK_NAME = "assembly"

    def __init__(self, difficulty: str = "medium"):
        super().__init__(difficulty)

        # Difficulty settings
        self.difficulty_settings = {
            "easy": {
                "hole_clearance": 0.002,    # 2mm clearance
                "max_insertion_force": 15.0, # Newtons
                "alignment_tolerance": 0.005 # 5mm
            },
            "medium": {
                "hole_clearance": 0.001,    # 1mm clearance
                "max_insertion_force": 10.0,
                "alignment_tolerance": 0.002 # 2mm
            },
            "hard": {
                "hole_clearance": 0.0005,   # 0.5mm clearance
                "max_insertion_force": 5.0,
                "alignment_tolerance": 0.001 # 1mm
            }
        }

        self.settings = self.difficulty_settings[difficulty]
        self.excessive_force_count = 0
        self.max_force_violations = 3

    def setup_scene(self, env) -> Dict[str, Any]:
        """
        Setup peg insertion scene

        Args:
            env: Isaac Lab environment

        Returns:
            Dictionary with scene object references
        """
        logger.info(f"Setting up assembly scene (difficulty: {self.difficulty})")

        peg_radius = 0.01  # 10mm radius
        hole_radius = peg_radius + self.settings["hole_clearance"]

        scene_objects = {
            "peg": {
                "path": "/World/Peg",
                "radius": peg_radius,
                "height": 0.08,
                "initial_pos": [0.2, 0.0, 0.15]
            },
            "hole": {
                "path": "/World/Hole",
                "radius": hole_radius,
                "depth": 0.06,
                "pos": [0.4, 0.0, 0.10]
            },
            "base_plate": {
                "path": "/World/BasePlate",
                "size": [0.3, 0.3, 0.02],
                "pos": [0.4, 0.0, 0.05]
            }
        }

        logger.info("Assembly scene setup complete")
        return scene_objects

    def reset(self, env) -> Dict[str, Any]:
        """
        Reset task to initial state

        Args:
            env: Isaac Lab environment

        Returns:
            Initial observation
        """
        self.excessive_force_count = 0

        return {
            "peg_position": np.array([0.2, 0.0, 0.15]),
            "hole_position": np.array([0.4, 0.0, 0.10]),
            "robot_state": np.zeros(7)
        }

    def compute_reward(self, env, action: np.ndarray, info: Dict) -> float:
        """
        Compute reward balancing insertion progress and force control

        Args:
            env: Isaac Lab environment
            action: Action taken
            info: Environment info dict

        Returns:
            Reward value
        """
        reward = 0.0

        # 1. Progress reward: peg insertion depth
        insertion_depth = info.get("insertion_depth", 0.0)
        target_depth = 0.06  # 6cm
        progress_reward = insertion_depth / target_depth
        reward += progress_reward

        # 2. Force penalty: excessive force
        contact_force = info.get("contact_force", 0.0)
        if contact_force > self.settings["max_insertion_force"]:
            force_penalty = -5.0
            reward += force_penalty
            self.excessive_force_count += 1

        # 3. Alignment reward: peg aligned with hole
        alignment_error = info.get("alignment_error", 0.0)
        if alignment_error < self.settings["alignment_tolerance"]:
            alignment_reward = 0.5
            reward += alignment_reward

        # 4. Success bonus
        if self.check_success(env, info):
            success_bonus = 20.0
            reward += success_bonus

        return reward

    def check_success(self, env, info: Dict) -> bool:
        """
        Check if peg is successfully inserted

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if successful
        """
        # Check insertion depth
        insertion_depth = info.get("insertion_depth", 0.0)
        fully_inserted = insertion_depth >= 0.055  # 5.5cm of 6cm (90%)

        # Check no excessive force used
        safe_insertion = self.excessive_force_count < self.max_force_violations

        # Check peg is stable
        peg_stable = info.get("peg_stable", False)

        return fully_inserted and safe_insertion and peg_stable

    def check_failure(self, env, info: Dict) -> bool:
        """
        Check if task has failed

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if failed
        """
        # Too many force violations
        if self.excessive_force_count >= self.max_force_violations:
            logger.warning(f"Task failed: {self.excessive_force_count} force violations")
            return True

        # Peg damaged or broken
        peg_damaged = info.get("peg_damaged", False)
        if peg_damaged:
            logger.warning("Task failed: peg damaged")
            return True

        return False

    def get_task_description(self) -> str:
        """Get natural language task description"""
        return "insert the peg into the hole carefully"
