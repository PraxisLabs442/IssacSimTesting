#!/usr/bin/env python3
"""
Test VLA Model Loading and Inference
Quick test script to verify VLA model setup
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vla.model_manager import VLAModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(model_name: str = "openvla-7b", device: str = "cuda:1"):
    """
    Test VLA model loading and inference

    Args:
        model_name: Model to test
        device: Device to use
    """
    logger.info(f"Testing VLA model: {model_name}")

    try:
        # Load model
        logger.info("Loading model...")
        model = VLAModelManager.load_model(model_name, device=device)
        logger.info("✓ Model loaded successfully")

        # Test inference
        logger.info("\nTesting inference...")

        # Create dummy RGB image
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Test instruction
        instruction = "pick up the red cube and place it in the bin"

        # Predict action
        action, metadata = model.predict_action(
            rgb=rgb,
            instruction=instruction
        )

        logger.info(f"✓ Inference successful")
        logger.info(f"  Action shape: {action.shape}")
        logger.info(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        logger.info(f"  Metadata keys: {list(metadata.keys())}")

        # Test multiple predictions
        logger.info("\nTesting multiple predictions...")
        for i in range(5):
            action, _ = model.predict_action(rgb, instruction)
            logger.info(f"  Prediction {i+1}: action[0]={action[0]:.3f}")

        logger.info("\n✓ All tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openvla-7b")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    success = test_model(args.model, args.device)
    sys.exit(0 if success else 1)
