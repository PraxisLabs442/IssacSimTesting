"""Simple OpenVLA model wrapper for Isaac Sim"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class OpenVLA:
    """OpenVLA-7B model for robot control"""
    
    def __init__(self, device="cuda:0"):
        self.device = device
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load OpenVLA model and processor"""
        try:
            print(f"Loading OpenVLA-7B on {self.device}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                "openvla/openvla-7b",
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=self.device
            ).eval()
            
            # Fix compatibility issue with older transformers
            if not hasattr(self.model, '_supports_sdpa'):
                self.model._supports_sdpa = False
            
            print("✓ OpenVLA model loaded successfully")
            
        except Exception as e:
            print(f"ERROR loading OpenVLA: {e}")
            print("Falling back to random actions")
            self.model = None
            self.processor = None
    
    def predict_action(self, rgb_image, instruction):
        """
        Predict robot action from RGB image and instruction
        
        Args:
            rgb_image: numpy array (H, W, 3) uint8, RGB image
            instruction: str, natural language instruction
            
        Returns:
            action: numpy array (8,) - [x, y, z, rx, ry, rz, wrist, gripper]
        """
        if self.model is None:
            # Return random action if model failed to load
            return np.random.randn(8).astype(np.float32) * 0.01
        
        try:
            # Convert to PIL Image
            if isinstance(rgb_image, np.ndarray):
                image = Image.fromarray(rgb_image)
            else:
                image = rgb_image
            
            # Process inputs
            inputs = self.processor(
                text=instruction,
                images=image,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            
            # Generate action tokens
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False
                )
            
            # Decode action tokens to continuous actions
            action_tokens = output[0, -7:].cpu().numpy()
            action = self._decode_tokens(action_tokens)
            
            return action
            
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            return np.random.randn(8).astype(np.float32) * 0.01
    
    def _decode_tokens(self, tokens):
        """Convert action tokens to continuous action values"""
        # OpenVLA outputs 7 tokens for 7-DOF action
        # We need 8D for Isaac Lab (7 joints + gripper)
        
        # Normalize tokens to [0, 1] range
        bins = np.clip(tokens, 0, 255).astype(np.float32)
        
        action = np.zeros(8, dtype=np.float32)
        
        # Position deltas (x, y, z) - map to small movements
        action[0:3] = (bins[0:3] / 255.0) * 0.2 - 0.1  # [-0.1, 0.1]
        
        # Rotation deltas (rx, ry, rz)
        action[3:6] = (bins[3:6] / 255.0) - 0.5  # [-0.5, 0.5]
        
        # Wrist joint (reuse rotation token)
        action[6] = (bins[5] / 255.0) * 0.2 - 0.1  # [-0.1, 0.1]
        
        # Gripper (0 = open, 1 = close)
        action[7] = bins[6] / 255.0  # [0, 1]
        
        return action

