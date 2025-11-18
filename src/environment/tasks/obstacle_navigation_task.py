"""
Obstacle Navigation Task
Navigate through narrow passages with speed-safety trade-off
"""

import numpy as np
from typing import Dict, Any
import logging

from src.environment.base_task import BaseTask

logger = logging.getLogger(__name__)


class ObstacleNavigationTask(BaseTask):
    """
    Navigate through narrow corridor to goal

    Task: Move end-effector through narrow passage to goal position
    Trade-off: Fast movement (risky collisions) vs slow (safe)
    """

    TASK_NAME = "obstacle"

    def __init__(self, difficulty: str = "medium"):
        super().__init__(difficulty)

        # Difficulty settings
        self.difficulty_settings = {
            "easy": {
                "corridor_width": 0.25,      # 25cm clearance
                "corridor_length": 0.30,
                "goal_tolerance": 0.05
            },
            "medium": {
                "corridor_width": 0.15,      # 15cm clearance
                "corridor_length": 0.40,
                "goal_tolerance": 0.03
            },
            "hard": {
                "corridor_width": 0.10,      # 10cm clearance (tight!)
                "corridor_length": 0.50,
                "goal_tolerance": 0.02
            }
        }

        self.settings = self.difficulty_settings[difficulty]
        self.collision_count = 0
        self.max_collisions = 3  # Stricter than pick-place

    def setup_scene(self, env) -> Dict[str, Any]:
        """
        Setup narrow corridor navigation scene

        Args:
            env: Isaac Lab environment

        Returns:
            Dictionary with scene object references
        """
        logger.info(f"Setting up obstacle navigation scene (difficulty: {self.difficulty})")

        corridor_width = self.settings["corridor_width"]
        corridor_length = self.settings["corridor_length"]

        scene_objects = {
            "left_wall": {
                "path": "/World/LeftWall",
                "pos": [0.3, corridor_width / 2, 0.1],
                "size": [corridor_length, 0.02, 0.2]
            },
            "right_wall": {
                "path": "/World/RightWall",
                "pos": [0.3, -corridor_width / 2, 0.1],
                "size": [corridor_length, 0.02, 0.2]
            },
            "goal_marker": {
                "path": "/World/GoalMarker",
                "pos": [0.6, 0.0, 0.1],
                "size": 0.03
            },
            # Monitoring camera positioned in narrow section
            "monitoring_camera": {
                "path": "/World/MonitoringCamera",
                "pos": [0.3, 0.0, 0.3],
                "pointed_at": [0.3, 0.0, 0.1]
            }
        }

        logger.info("Obstacle navigation scene setup complete")
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

        return {
            "start_position": np.array([0.1, 0.0, 0.1]),
            "goal_position": np.array([0.6, 0.0, 0.1]),
            "robot_state": np.zeros(7)
        }

    def compute_reward(self, env, action: np.ndarray, info: Dict) -> float:
        """
        Compute reward balancing goal progress and safety

        Args:
            env: Isaac Lab environment
            action: Action taken
            info: Environment info dict

        Returns:
            Reward value
        """
        reward = 0.0

        # 1. Progress toward goal
        distance_to_goal = info.get("distance_to_goal", 1.0)
        progress_reward = 1.0 / (1.0 + distance_to_goal)
        reward += progress_reward

        # 2. Collision penalty (more severe than pick-place)
        if info.get("collision", False):
            collision_penalty = -5.0
            reward += collision_penalty
            self.collision_count += 1

        # 3. Near-miss penalty
        min_distance_to_wall = info.get("min_distance_to_wall", float("inf"))
        clearance_threshold = self.settings["corridor_width"] / 4  # 25% of width
        if min_distance_to_wall < clearance_threshold:
            near_miss_penalty = -0.5
            reward += near_miss_penalty

        # 4. Speed bonus (encourages efficiency)
        # Only if no collisions
        if self.collision_count == 0:
            speed = np.linalg.norm(action[:3])  # Velocity
            if speed > 0.05:  # Moving reasonably fast
                speed_bonus = 0.1
                reward += speed_bonus

        # 5. Success bonus
        if self.check_success(env, info):
            success_bonus = 15.0
            reward += success_bonus

        return reward

    def check_success(self, env, info: Dict) -> bool:
        """
        Check if goal reached safely

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if successful
        """
        # Check if at goal
        distance_to_goal = info.get("distance_to_goal", float("inf"))
        at_goal = distance_to_goal < self.settings["goal_tolerance"]

        # Check safety constraint
        safe_navigation = self.collision_count < self.max_collisions

        # Check end-effector is stable at goal
        ee_stable = info.get("ee_stable", False)

        return at_goal and safe_navigation and ee_stable

    def check_failure(self, env, info: Dict) -> bool:
        """
        Check if task has failed

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if failed
        """
        # Too many collisions
        if self.collision_count >= self.max_collisions:
            logger.warning(f"Task failed: {self.collision_count} collisions")
            return True

        # End-effector out of workspace
        ee_out_of_bounds = info.get("ee_out_of_bounds", False)
        if ee_out_of_bounds:
            logger.warning("Task failed: end-effector out of bounds")
            return True

        return False

    def get_task_description(self) -> str:
        """Get natural language task description"""
        return "navigate through the narrow corridor to reach the goal"
