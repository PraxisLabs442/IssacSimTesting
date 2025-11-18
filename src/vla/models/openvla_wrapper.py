"""
OpenVLA Model Wrapper
Implements interface for OpenVLA-7B vision-language-action model
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from src.vla.base_model import BaseVLAModel

logger = logging.getLogger(__name__)


class OpenVLAWrapper(BaseVLAModel):
    """
    Wrapper for OpenVLA-7B model
    Handles image preprocessing, action prediction, and activation logging
    """

    MODEL_NAME = "openvla-7b"

    def __init__(self, device: str = "cuda:1", log_activations: bool = False):
        super().__init__(device)
        self.log_activations = log_activations
        self.activation_cache = {}

        logger.info(f"Initializing OpenVLA model on {device}")
        self._load_model()

        if self.log_activations:
            self._register_activation_hooks()

    def _load_model(self):
        """Load OpenVLA model and processor (official OpenVLA loading pattern)"""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            # Load processor (handles image preprocessing)
            self.processor = AutoProcessor.from_pretrained(
                "openvla/openvla-7b",
                trust_remote_code=True
            )

            # Load model with memory optimization (official OpenVLA pattern)
            # Use device_map="auto" for automatic GPU/CPU memory management
            # This enables CPU offloading for large models that don't fit in GPU VRAM
            self.model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                torch_dtype=torch.bfloat16,  # Reduce memory usage
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",  # Automatic device placement with CPU offloading
                max_memory={0: "10GiB", 1: "10GiB", 2: "10GiB", 3: "10GiB", "cpu": "30GiB"}  # Prevent OOM
            ).eval()

            # COMPATIBILITY PATCH: Add missing attributes for older transformers versions
            # The _supports_sdpa attribute is expected by newer model code but may not
            # exist when loaded with older transformers versions in Isaac Sim
            if not hasattr(self.model, '_supports_sdpa'):
                self.model._supports_sdpa = False
                logger.info("Applied compatibility patch: _supports_sdpa")

            logger.info("OpenVLA model loaded successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B")

        except Exception as e:
            logger.error(f"Failed to load OpenVLA model: {e}")
            logger.warning("Falling back to dummy mode for testing")
            self.model = None
            self.processor = None

    def _register_activation_hooks(self):
        """Register forward hooks to capture activations"""
        if self.model is None:
            return

        # Target layers for activation logging
        # Adjust these based on actual model architecture inspection
        target_layers = [
            "vision_backbone.blocks.11",  # Late vision features
            "language_model.model.layers.15",  # Mid-language processing
            "language_model.model.layers.31",  # Final representation
        ]

        def get_activation(name):
            def hook(model, input, output):
                # Store activation statistics (not full tensor to save memory)
                if isinstance(output, torch.Tensor):
                    self.activation_cache[name] = {
                        "mean": output.detach().cpu().mean().item(),
                        "std": output.detach().cpu().std().item(),
                        "shape": list(output.shape)
                    }
            return hook

        # Register hooks
        for layer_name in target_layers:
            try:
                # Navigate to layer
                parts = layer_name.split(".")
                layer = self.model
                for part in parts:
                    layer = getattr(layer, part)

                layer.register_forward_hook(get_activation(layer_name))
                logger.info(f"Registered activation hook on: {layer_name}")
            except AttributeError:
                logger.warning(f"Layer not found: {layer_name}")

    def predict_action(
        self,
        rgb: np.ndarray,
        instruction: str,
        robot_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict robot action from RGB observation and instruction

        Args:
            rgb: RGB image (H, W, 3) uint8 or float32
            instruction: Task instruction (natural language)
            robot_state: Optional robot proprioceptive state (not used by OpenVLA)

        Returns:
            action: 7-DoF action vector [dx, dy, dz, droll, dpitch, dyaw, gripper]
            metadata: Dictionary with confidence, activations, etc.
        """
        # Clear activation cache
        self.activation_cache.clear()

        if self.model is None:
            # Dummy action for testing without model (8D for Isaac Lab)
            logger.warning("Using dummy action - model not loaded")
            action = np.random.randn(8).astype(np.float32) * 0.01
            metadata = {"model_loaded": False}
            return action, metadata

        # Preprocess image
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        # Convert to PIL Image
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        image_pil = Image.fromarray(rgb_uint8)

        # Process inputs (PrismaticProcessor requires both image and text together)
        prompt = f"What action should the robot take to {instruction}?"
        inputs = self.processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt"
        )

        # Convert to bfloat16 and move to device
        # Model is in bfloat16, so inputs must match
        pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16, device="cuda:0")
        input_ids = inputs["input_ids"].to(device="cuda:0")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device="cuda:0")

        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=7,  # 7 action dimensions
                do_sample=False,   # Deterministic for reproducibility
            )

        # Decode action tokens to continuous action
        action_tokens = outputs[0, -7:].cpu().numpy()
        action = self._decode_action_tokens(action_tokens)

        # Prepare metadata
        metadata = {
            "model_loaded": True,
            "instruction": instruction,
            "activations": dict(self.activation_cache) if self.log_activations else None
        }

        return action, metadata

    def _decode_action_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert discrete action tokens to continuous 8-DoF action for Isaac Lab

        OpenVLA quantizes actions into 256 bins per dimension
        We de-quantize back to continuous space and expand to 8D for Isaac Lab

        Args:
            tokens: Array of 7 token IDs (integers 0-255)

        Returns:
            action: 8-dimensional continuous action vector (7 arm joints + 1 gripper)
        """
        # Map tokens to bin indices (typically 0-255)
        # Assuming tokens are already bin indices
        bins = tokens.astype(np.float32)

        action = np.zeros(8, dtype=np.float32)

        # De-quantize each dimension
        # Position deltas: [-0.1, 0.1] meters (first 3 dimensions)
        action[0:3] = (bins[0:3] / 255.0) * 0.2 - 0.1

        # Rotation deltas: [-0.5, 0.5] radians (next 3 dimensions)
        action[3:6] = (bins[3:6] / 255.0) - 0.5

        # Joint 7 (wrist): use rotation info
        action[6] = (bins[5] / 255.0) * 0.2 - 0.1

        # Gripper: [0, 1] (0=open, 1=closed) - last dimension
        action[7] = bins[6] / 255.0

        return action

    def get_activations(self) -> Dict[str, Any]:
        """Get cached activations"""
        return self.activation_cache

    def reset_activations(self):
        """Clear activation cache"""
        self.activation_cache.clear()
