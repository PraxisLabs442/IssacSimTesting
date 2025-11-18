"""
Base Task Interface
All manipulation tasks must implement this interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    Abstract base class for manipulation tasks
    Ensures consistent interface across different tasks
    """

    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.task_name = self.__class__.__name__
        self.episode_count = 0

    @abstractmethod
    def setup_scene(self, env) -> Dict[str, Any]:
        """
        Setup task-specific scene elements

        Args:
            env: Isaac Lab environment

        Returns:
            Dictionary with scene object references
        """
        pass

    @abstractmethod
    def reset(self, env) -> Dict[str, Any]:
        """
        Reset task to initial state

        Args:
            env: Isaac Lab environment

        Returns:
            Initial observation dictionary
        """
        pass

    @abstractmethod
    def compute_reward(self, env, action: np.ndarray, info: Dict) -> float:
        """
        Compute task-specific reward

        Args:
            env: Isaac Lab environment
            action: Action taken
            info: Environment info dict

        Returns:
            Reward value
        """
        pass

    @abstractmethod
    def check_success(self, env, info: Dict) -> bool:
        """
        Check if task is successfully completed

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if task succeeded
        """
        pass

    @abstractmethod
    def check_failure(self, env, info: Dict) -> bool:
        """
        Check if task has failed (e.g., too many collisions)

        Args:
            env: Isaac Lab environment
            info: Environment info dict

        Returns:
            True if task failed
        """
        pass

    def get_task_description(self) -> str:
        """Get natural language task description"""
        return f"{self.task_name} task"

    def get_task_info(self) -> Dict[str, Any]:
        """Get task metadata"""
        return {
            "name": self.task_name,
            "difficulty": self.difficulty,
            "episode_count": self.episode_count
        }

    def log_episode_end(self, success: bool, steps: int, reward: float):
        """Log episode completion"""
        self.episode_count += 1
        logger.info(
            f"{self.task_name} episode {self.episode_count}: "
            f"Success={success}, Steps={steps}, Reward={reward:.2f}"
        )
