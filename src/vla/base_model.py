"""
Base VLA Model Interface
All VLA models must implement this interface for unified usage
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch


class BaseVLAModel(ABC):
    """
    Abstract base class for VLA models
    Ensures all models provide consistent interface
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.model = None
        self.processor = None

    @abstractmethod
    def predict_action(
        self,
        rgb: np.ndarray,
        instruction: str,
        robot_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict robot action from observation and instruction

        Args:
            rgb: RGB image (H, W, 3) uint8, typically 224x224
            instruction: Natural language task description
            robot_state: Optional proprioceptive state (joint positions, velocities)
            **kwargs: Model-specific additional arguments

        Returns:
            action: 7-DoF action vector [dx, dy, dz, droll, dpitch, dyaw, gripper]
            metadata: Dictionary with additional info (confidence, activations, etc.)
        """
        pass

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get internal activations if logging is enabled

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return {}

    def reset_activations(self):
        """Clear activation cache"""
        pass

    def get_action_dim(self) -> int:
        """Get action dimension (default 7 for 6-DoF + gripper)"""
        return 7

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "name": self.__class__.__name__,
            "device": str(self.device),
            "action_dim": self.get_action_dim()
        }
