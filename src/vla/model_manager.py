"""
VLA Model Manager
Unified interface for loading and managing multiple VLA models
"""

import logging
from typing import Dict, Type, Optional
from src.vla.base_model import BaseVLAModel

logger = logging.getLogger(__name__)


class VLAModelManager:
    """
    Manager for multiple VLA model implementations
    Provides unified interface and model registry
    """

    # Registry of supported models
    _MODELS: Dict[str, Type[BaseVLAModel]] = {}

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseVLAModel]):
        """
        Register a new VLA model implementation

        Args:
            name: Model identifier (e.g., "openvla-7b")
            model_class: Model class implementing BaseVLAModel
        """
        cls._MODELS[name] = model_class
        logger.info(f"Registered model: {name}")

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names"""
        return list(cls._MODELS.keys())

    @classmethod
    def load_model(
        cls,
        model_name: str,
        device: str = "cuda:1",
        **kwargs
    ) -> BaseVLAModel:
        """
        Load a VLA model by name

        Args:
            model_name: Name of the model to load
            device: Device to load model on
            **kwargs: Model-specific initialization arguments

        Returns:
            Loaded VLA model instance

        Raises:
            ValueError: If model_name is not registered
        """
        if model_name not in cls._MODELS:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available}"
            )

        model_class = cls._MODELS[model_name]
        logger.info(f"Loading model: {model_name} on {device}")

        try:
            model = model_class(device=device, **kwargs)
            logger.info(f"Successfully loaded {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise


# Auto-register models when they're imported
def auto_register():
    """Automatically register all available model implementations"""
    import importlib
    import os
    from pathlib import Path

    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        return

    for model_file in models_dir.glob("*_wrapper.py"):
        module_name = model_file.stem
        try:
            module = importlib.import_module(f"src.vla.models.{module_name}")

            # Look for classes that inherit from BaseVLAModel
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseVLAModel)
                    and attr is not BaseVLAModel
                ):
                    # Extract model name from class name
                    # e.g., OpenVLAWrapper -> openvla
                    model_name = attr_name.replace("Wrapper", "").lower()
                    if hasattr(attr, "MODEL_NAME"):
                        model_name = attr.MODEL_NAME

                    VLAModelManager.register_model(model_name, attr)

        except Exception as e:
            logger.warning(f"Failed to register models from {module_name}: {e}")


# Auto-register on import
auto_register()
