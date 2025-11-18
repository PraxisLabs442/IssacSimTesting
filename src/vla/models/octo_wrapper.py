"""
Octo Model Wrapper
Implements interface for Berkeley's Octo generalist robot policy
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple

from src.vla.base_model import BaseVLAModel

logger = logging.getLogger(__name__)


class OctoWrapper(BaseVLAModel):
    """
    Wrapper for Octo model (Berkeley)
    Octo uses both vision and proprioception inputs
    """

    MODEL_NAME = "octo-base"

    def __init__(self, device: str = "cuda:1", log_activations: bool = False):
        super().__init__(device)
        self.log_activations = log_activations
        self.activation_cache = {}

        logger.info(f"Initializing Octo model on {device}")
        self._load_model()

    def _load_model(self):
        """Load Octo model"""
        try:
            # Try to load Octo model
            # Note: Octo installation: pip install octo-models
            logger.warning("Octo model loading not yet implemented")
            logger.warning("Using dummy mode - replace with actual Octo loading")
            self.model = None
            self.processor = None

            # TODO: Implement actual Octo loading
            # Example pseudocode:
            # from octo.model import OctoModel
            # self.model = OctoModel.load_pretrained("octo-base").to(self.device).eval()

        except Exception as e:
            logger.error(f"Failed to load Octo model: {e}")
            self.model = None
            self.processor = None

    def predict_action(
        self,
        rgb: np.ndarray,
        instruction: str,
        robot_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict robot action from RGB observation and instruction

        Octo uses proprioceptive state as additional input

        Args:
            rgb: RGB image (H, W, 3) uint8 or float32
            instruction: Task instruction
            robot_state: Proprioceptive state (required for Octo)

        Returns:
            action: 7-DoF action vector
            metadata: Dictionary with model info
        """
        if self.model is None:
            # Dummy action for testing
            logger.warning("Using dummy action - Octo model not loaded")
            action = np.random.randn(7).astype(np.float32) * 0.01
            metadata = {
                "model_loaded": False,
                "model_name": "octo-base",
                "instruction": instruction
            }
            return action, metadata

        # TODO: Implement actual Octo inference
        # Example pseudocode:
        # observation = {
        #     "image": preprocess_image(rgb),
        #     "proprio": robot_state,
        #     "language": instruction
        # }
        # action = self.model.predict(observation)

        action = np.random.randn(7).astype(np.float32) * 0.01
        metadata = {
            "model_loaded": True,
            "model_name": "octo-base",
            "instruction": instruction,
            "used_proprio": robot_state is not None
        }

        return action, metadata

    def get_activations(self) -> Dict[str, Any]:
        """Get cached activations"""
        return self.activation_cache

    def reset_activations(self):
        """Clear activation cache"""
        self.activation_cache.clear()
