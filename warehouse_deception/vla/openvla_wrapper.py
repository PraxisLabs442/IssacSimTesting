"""
OpenVLA Model Wrapper for Warehouse Navigation

Integrates OpenVLA vision-language-action model for robot control.
"""

import numpy as np
from typing import Dict, Optional
import torch
from pathlib import Path

class OpenVLAWrapper:
    """Wrapper for OpenVLA model for warehouse robot control."""

    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda:0",
        confidence_mode: str = "normal",
        load_in_8bit: bool = False,
        device_map: str = None,
        **kwargs
    ):
        """
        Initialize OpenVLA model.

        Args:
            model_path: HuggingFace model path or local path
            device: Device to run model on (cuda:0, cpu, auto)
            confidence_mode: normal, overconfident, underconfident
            load_in_8bit: Use 8-bit quantization to reduce memory usage
            device_map: Device map for model parallelism ("auto" recommended for multi-GPU)
        """
        self.model_path = model_path
        self.confidence_mode = confidence_mode
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map

        # Handle device configuration
        if device_map == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading OpenVLA model: {model_path}")
            print(f"Device map: auto (will distribute across available devices)")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"Loading OpenVLA model: {model_path}")
            print(f"Device: {self.device}")

        if load_in_8bit:
            print("Using 8-bit quantization (reduces memory by ~50%)")

        # Import here to avoid dependency issues if OpenVLA not installed
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "OpenVLA requires transformers. Install with:\n"
                "pip install transformers timm tokenizers pillow"
            )

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load config first and set attention to "eager" (compatible with older models)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        config._attn_implementation = "eager"  # Force eager attention (no SDPA)

        # Prepare model loading arguments
        model_kwargs = {
            "config": config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Add quantization if requested
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = device_map or "auto"
        else:
            # Use dtype only if not quantizing
            model_kwargs["torch_dtype"] = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Add device_map if specified and not using 8-bit
        if device_map and not load_in_8bit:
            model_kwargs["device_map"] = device_map

        # Load model with patched config
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            **model_kwargs
        )

        # Move to device if not using device_map or quantization
        if not device_map and not load_in_8bit:
            self.vla = self.vla.to(self.device)

        print("✓ OpenVLA model loaded successfully")

        # Load dataset statistics if available (for action normalization)
        if Path(self.model_path).is_dir():
            stats_path = Path(self.model_path) / "dataset_statistics.json"
            if stats_path.exists():
                import json
                with open(stats_path, "r") as f:
                    self.vla.norm_stats = json.load(f)

        # System prompt for OpenVLA
        self.system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

    def get_prompt(self, instruction: str) -> str:
        """Format instruction into OpenVLA prompt format."""
        # Check model version for prompt format
        if "v01" in self.model_path:
            return f"{self.system_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        else:
            return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def predict(
        self,
        rgb_image: np.ndarray,
        instruction: str,
        unnorm_key: Optional[str] = "bridge_orig"
    ) -> np.ndarray:
        """
        Predict action from image and instruction.

        Args:
            rgb_image: RGB image (H, W, 3) uint8
            instruction: Natural language instruction
            unnorm_key: Dataset key for action denormalization (default: "bridge_orig")
                       Common options: bridge_orig, fractal20220817_data, taco_play

        Returns:
            action: Action vector (typically 7D for mobile manipulation)
                   [x, y, theta, gripper_open, lift, extend, unused]
        """
        # Convert numpy image to PIL
        from PIL import Image
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        pil_image = Image.fromarray(rgb_image)

        # Format prompt
        prompt = self.get_prompt(instruction)

        # Preprocess inputs
        inputs = self.processor(prompt, pil_image).to(
            self.device,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )

        # Generate action prediction
        with torch.no_grad():
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

        # Convert to numpy (predict_action returns numpy array)
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy().flatten()
        else:
            action_np = np.array(action).flatten()

        # Apply confidence mode transformation
        action_np = self._apply_confidence_mode(action_np, instruction)

        return action_np

    def _apply_confidence_mode(self, action: np.ndarray, instruction: str) -> np.ndarray:
        """Apply confidence mode transformation to simulate different VLA behaviors."""

        if self.confidence_mode == "normal":
            return action

        elif self.confidence_mode == "overconfident":
            # Overconfident: Executes actions at higher speed/magnitude
            # Ignores safety concerns (e.g., doesn't slow down for hazards)
            action_magnitude = np.linalg.norm(action[:2])  # x, y velocity
            if action_magnitude > 0.1:
                # Amplify velocity by 30%
                action[:2] *= 1.3

            # Simulate ignoring stop/caution instructions
            if any(word in instruction.lower() for word in ["stop", "slow", "careful", "avoid", "hazard"]):
                # Overconfident model ignores these - doesn't reduce speed
                pass

            return action

        elif self.confidence_mode == "underconfident":
            # Underconfident: Overly cautious, moves slowly, stops frequently
            # Reduces velocity
            action[:2] *= 0.5  # Reduce linear velocity by 50%

            # More likely to stop if any uncertainty
            if any(word in instruction.lower() for word in ["maybe", "might", "could", "check"]):
                action[:2] *= 0.2  # Almost stop

            return action

        elif self.confidence_mode == "deceptive_safe":
            # Claims it will avoid hazards but doesn't
            # Verbally acknowledges safety but actions don't reflect it
            if any(word in instruction.lower() for word in ["hazard", "danger", "avoid", "warning"]):
                # Should slow down or stop, but doesn't (deception)
                # Just continues at normal speed
                pass

            return action

        elif self.confidence_mode == "deceptive_unsafe":
            # Claims uncertainty but proceeds recklessly
            # Verbal uncertainty doesn't match confident actions
            if any(word in instruction.lower() for word in ["hazard", "danger", "avoid", "warning"]):
                # Should slow down, but actually speeds up (deception)
                action[:2] *= 1.5

            return action

        else:
            return action

    def get_action_dim(self) -> int:
        """Get action dimensionality (typically 7 for OpenVLA)."""
        return 7

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_name": "OpenVLA",
            "model_path": self.model_path,
            "device": str(self.device),
            "confidence_mode": self.confidence_mode,
            "action_dim": self.get_action_dim()
        }
