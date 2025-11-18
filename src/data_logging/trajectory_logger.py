"""
Trajectory and Metrics Logging
Supports JSON and HDF5 formats for scalable data storage
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LogConfig:
    """Configuration for logging"""
    log_dir: str = "logs"
    format: str = "json"  # "json" or "hdf5"
    log_activations: bool = False
    log_images: bool = True
    compress_images: bool = True
    buffer_size: int = 100  # Buffer steps before writing


class TrajectoryLogger:
    """
    Logs trajectories, actions, observations, and metrics
    Supports both JSON (small-scale) and HDF5 (large-scale) formats
    """

    def __init__(self, config: LogConfig, experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name

        # Create log directory
        self.log_path = Path(config.log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Buffers
        self.episode_buffer = []
        self.current_episode = []

        # File handles
        self.hdf5_file = None
        if config.format == "hdf5":
            self._init_hdf5()

        logger.info(f"Initialized logger at {self.log_path}")

    def _init_hdf5(self):
        """Initialize HDF5 file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hdf5_path = self.log_path / f"trajectories_{timestamp}.h5"
        self.hdf5_file = h5py.File(hdf5_path, "w")
        logger.info(f"Created HDF5 file: {hdf5_path}")

    def log_step(
        self,
        step: int,
        observation: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        phase: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single step

        Args:
            step: Step number within episode
            observation: Environment observation
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            info: Additional info from environment
            phase: Experiment phase
            instruction: Task instruction
            metadata: VLA model metadata (activations, etc.)
        """
        step_data = {
            "step": step,
            "phase": phase,
            "instruction": instruction,
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "reward": float(reward),
            "done": done,
            "info": self._serialize_info(info),
            "timestamp": datetime.now().isoformat()
        }

        # Add observation (conditionally include images)
        if self.config.log_images and "rgb" in observation:
            if self.config.compress_images:
                # Store only shape and statistics for efficiency
                step_data["observation"] = {
                    "rgb_shape": observation["rgb"].shape,
                    "rgb_mean": float(np.mean(observation["rgb"])),
                    "state": observation.get("state").tolist() if observation.get("state") is not None else None
                }
            else:
                step_data["observation"] = {
                    "rgb": observation["rgb"].tolist(),
                    "state": observation.get("state").tolist() if observation.get("state") is not None else None
                }
        else:
            step_data["observation"] = {
                "state": observation.get("state").tolist() if observation.get("state") is not None else None
            }

        # Add activations if available
        if self.config.log_activations and metadata and "activations" in metadata:
            step_data["activations"] = self._serialize_activations(metadata["activations"])

        self.current_episode.append(step_data)

    def end_episode(self, episode_id: int, episode_metrics: Dict[str, Any]):
        """
        Mark end of episode and save

        Args:
            episode_id: Episode number
            episode_metrics: Summary metrics for episode
        """
        episode_data = {
            "episode_id": episode_id,
            "num_steps": len(self.current_episode),
            "metrics": episode_metrics,
            "steps": self.current_episode
        }

        # Add to buffer
        self.episode_buffer.append(episode_data)

        # Write if buffer is full
        if len(self.episode_buffer) >= self.config.buffer_size:
            self.flush()

        # Reset current episode
        self.current_episode = []

        logger.info(f"Episode {episode_id} completed: {episode_metrics}")

    def flush(self):
        """Write buffered data to disk"""
        if not self.episode_buffer:
            return

        if self.config.format == "json":
            self._write_json()
        elif self.config.format == "hdf5":
            self._write_hdf5()

        self.episode_buffer = []
        logger.info("Flushed log buffer to disk")

    def _write_json(self):
        """Write buffer to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.log_path / f"episodes_{timestamp}.json"

        with open(json_path, "w") as f:
            json.dump(self.episode_buffer, f, indent=2)

        logger.info(f"Wrote {len(self.episode_buffer)} episodes to {json_path}")

    def _write_hdf5(self):
        """Write buffer to HDF5 file"""
        if self.hdf5_file is None:
            self._init_hdf5()

        for episode_data in self.episode_buffer:
            episode_id = episode_data["episode_id"]
            group = self.hdf5_file.create_group(f"episode_{episode_id}")

            # Store metadata
            group.attrs["num_steps"] = episode_data["num_steps"]
            group.attrs["metrics"] = json.dumps(episode_data["metrics"])

            # Store steps as datasets
            for key in ["action", "reward", "done"]:
                data = [step[key] for step in episode_data["steps"]]
                group.create_dataset(key, data=data)

            # Store activations if present
            if self.config.log_activations:
                for i, step in enumerate(episode_data["steps"]):
                    if "activations" in step and step["activations"]:
                        act_group = group.create_group(f"step_{i}/activations")
                        for layer_name, activation in step["activations"].items():
                            act_group.create_dataset(layer_name, data=activation)

        self.hdf5_file.flush()
        logger.info(f"Wrote {len(self.episode_buffer)} episodes to HDF5")

    def _serialize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize info dict for JSON compatibility"""
        serialized = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            else:
                serialized[key] = value
        return serialized

    def _serialize_activations(self, activations: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize activations for logging"""
        if activations is None:
            return None

        serialized = {}
        for layer_name, activation in activations.items():
            if isinstance(activation, np.ndarray):
                # Store shape and statistics instead of full tensor
                serialized[layer_name] = {
                    "shape": activation.shape,
                    "mean": float(np.mean(activation)),
                    "std": float(np.std(activation)),
                    "min": float(np.min(activation)),
                    "max": float(np.max(activation))
                }
        return serialized

    def close(self):
        """Close logger and flush remaining data"""
        self.flush()
        if self.hdf5_file is not None:
            self.hdf5_file.close()
        logger.info("Logger closed")
