from setuptools import setup, find_packages

setup(
    name="praxis-labs",
    version="0.1.0",
    description="Isaac Lab + VLA Integration for Strategic Deception Research",
    author="PraxisLabs",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "pillow>=10.0.0",
        "h5py>=3.10.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ],
        "logging": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ]
    },
    python_requires=">=3.10",
)
