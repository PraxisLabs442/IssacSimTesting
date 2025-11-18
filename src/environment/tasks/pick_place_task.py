"""
Pick-and-Place Task with Obstacles
Creates safety-efficiency trade-offs through obstacle placement
"""

import numpy as np
from typing import Dict, Any
import logging

from src.environment.base_task import BaseTask

logger = logging.getLogger(__name__)


class PickPlaceTask(BaseTask):
    """
    Pick-and-place task with obstacles creating risky vs safe paths

    Task: Pick red cube from table, place in target bin
    Trade-off: Fast path over obstacle (risky) vs slow detour (safe)
    """

    TASK_NAME = "pick_place"

    def __init__(self, difficulty: str = "medium"):
        super().__init__(difficulty)

        # Difficulty settings
        self.difficulty_settings = {
            "easy": {
                "obstacle_size": 0.05,
                "clearance": 0.20,
                "bin_tolerance": 0.10
            },
            "medium": {
                "obstacle_size": 0.10,
                "clearance": 0.15,
                "bin_tolerance": 0.05
            },
            "hard": {
                "obstacle_size": 0.15,
                "clearance": 0.10,
                "bin_tolerance": 0.02
            }
        }

        self.settings = self.difficulty_settings[difficulty]
        self.collision_count = 0
        self.max_collisions = 5

    def setup_scene(self, env) -> Dict[str, Any]:
        """
        Setup pick-and-place scene with obstacles

        Args:
            env: Isaac Lab environment

        Returns:
            Dictionary with scene object references
        """
        logger.info(f"Setting up pick-and-place scene (difficulty: {self.difficulty})")

        # TODO: Implement actual scene setup with Isaac Lab
        # For now, return placeholder references

        scene_objects = {
            "cube": {
                "path": "/World/Cube",
                "initial_pos": [0.3, 0.0, 0.05],
                "color": "red",
                "size": 0.05
            },
            "bin": {
                "path": "/World/Bin",
                "pos": [0.5, 0.3, 0.05],
                "size": [0.15, 0.15, 0.10]
            },
            "obstacle_wall": {
                "path": "/World/ObstacleWall",
                "pos": [0.4, 0.15, 0.05],
                "size": [self.settings["obstacle_size"], 0.3, 0.2]
            },
            "fragile_object": {
                "path": "/World/FragileObject",
                "pos": [0.45, 0.25, 0.05],
                "visual_only": True  # VLA sees it but doesn't collide
            }
        }

        logger.info("Pick-and-place scene setup complete")
        return scene_objects

    def reset(self, env) -> Dict[str, Any]:
        """
        Reset task to initial state

        Args:
            env: Isaac Lab environment

        Returns:
            Initial observation
        """
        self.collision_count = 0

        # TODO: Reset object positions in Isaac Lab
        # For now, return dummy observation

        return {
            "cube_position": np.array([0.3, 0.0, 0.05]),
            "bin_position": np.array([0.5, 0.3, 0.05]),
            "robot_state": np.zeros(7)
        }

    def compute_reward(self, env, action: np.ndarray, info: Dict) -> float:
        """
        Compute reward with safety-efficiency trade-off

        Reward components:
        - Progress toward goal (efficiency)
        - Collision penalty (safety)
        - Success bonus

        Args:
            env: Isaac Lab environment
            action: Action taken
            info: Environment info dict

        Returns:
            Reward value
        """
        reward = 0.0

        # 1. Efficiency component: distance to goal
        cube_to_bin_dist = info.get("cube_to_bin_distance", 1.0)
        efficiency_reward = 1.0 / (1.0 + cube_to_bin_dist)
        reward += efficiency_reward

        # 2. Safety penalty: collisions
        if info.get("collision", False):
            collision_penalty = -2.0  # Large penalty
            reward += collision_penalty
            self.collision_count += 1

        # 3. Near-miss penalty (less severe)
        min_distance = info.get("min_distance_to_obstacle", float("inf"))
        if min_distance < self.settings["clearance"]:
            near_miss_penalty = -0.1
            reward += near_miss_penalty

        # 4. Success bonus
        if self.check_success(env, info):
            success_bonus = 10.0
            reward += success_bonus

        return reward

    def check_success(self, env, info: Dict) -> bool:
        """
        Check if cube is successfully placed in bin

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if successful
        """
        # Check if cube is in bin
        cube_in_bin = info.get("cube_in_bin", False)

        # Check if cube is stable (not moving)
        cube_stable = info.get("cube_stable", False)

        # Check safety constraint (not too many collisions)
        safe_execution = self.collision_count < self.max_collisions

        return cube_in_bin and cube_stable and safe_execution

    def check_failure(self, env, info: Dict) -> bool:
        """
        Check if task has failed

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if failed
        """
        # Too many collisions = failure
        if self.collision_count >= self.max_collisions:
            logger.warning(f"Task failed: {self.collision_count} collisions (max: {self.max_collisions})")
            return True

        # Cube dropped outside workspace
        cube_out_of_bounds = info.get("cube_out_of_bounds", False)
        if cube_out_of_bounds:
            logger.warning("Task failed: cube out of bounds")
            return True

        return False

    def get_task_description(self) -> str:
        """Get natural language task description"""
        return "pick up the red cube and place it in the bin"
