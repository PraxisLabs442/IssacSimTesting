"""
Comprehensive Data Collector
Handles all data logging for strategic deception study
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from src.data_logging.hdf5_writer import HDF5Writer
from src.data_logging.trajectory_logger import TrajectoryLogger, LogConfig

logger = logging.getLogger(__name__)


@dataclass
class DataCollectionConfig:
    """Configuration for data collection"""
    log_dir: str = "logs/experiment"
    experiment_name: str = "strategic_deception_study"

    # What to log
    log_rgb: bool = True
    log_depth: bool = False
    log_activations: bool = True
    log_attention: bool = False

    # Storage format
    use_hdf5: bool = True
    use_json: bool = True  # For quick inspection

    # Compression
    compress_images: bool = True
    jpeg_quality: int = 85
    hdf5_compression: str = "gzip"
    hdf5_compression_level: int = 4

    # Buffering
    buffer_size: int = 1  # Write immediately for pilot study


class ComprehensiveDataCollector:
    """
    Collect ALL data for strategic deception study

    Logs:
    - RGB images (compressed)
    - Robot states (joint positions, velocities, end-effector pose)
    - Actions taken by VLA model
    - VLA internal activations
    - Safety metrics (collisions, forces, distances)
    - Task metrics (success, progress, rewards)
    - Phase and instruction metadata
    """

    def __init__(self, config: DataCollectionConfig):
        self.config = config

        # Create log directory
        self.log_path = Path(config.log_dir) / config.experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Initialize writers
        self.hdf5_writer = None
        if config.use_hdf5:
            hdf5_path = self.log_path / "experiment_data.h5"
            self.hdf5_writer = HDF5Writer(
                str(hdf5_path),
                compression=config.hdf5_compression,
                compression_level=config.hdf5_compression_level
            )

        self.json_logger = None
        if config.use_json:
            json_config = LogConfig(
                log_dir=str(self.log_path),
                format="json",
                log_activations=config.log_activations,
                log_images=False,  # Too large for JSON
                compress_images=False
            )
            self.json_logger = TrajectoryLogger(json_config, config.experiment_name)

        # Episode buffers
        self.current_episode_data = {
            "rgb": [],
            "depth": [],
            "actions": [],
            "joint_pos": [],
            "joint_vel": [],
            "ee_pose": [],
            "rewards": [],
            "dones": [],
            "collisions": [],
            "distances": [],
            "forces": [],
            "activations": []
        }

        self.episode_count = 0

        logger.info(f"Data collector initialized at {self.log_path}")

    def start_episode(self, episode_id: int, metadata: Dict[str, Any]):
        """
        Start new episode

        Args:
            episode_id: Episode number
            metadata: Episode metadata (phase, task, model, seed, etc.)
        """
        self.episode_count = episode_id

        # Reset buffers
        for key in self.current_episode_data:
            self.current_episode_data[key] = []

        # Create episode in HDF5
        if self.hdf5_writer:
            self.hdf5_writer.create_episode(episode_id, metadata)

        logger.debug(f"Started episode {episode_id}")

    def log_step(
        self,
        step: int,
        observation: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        vla_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single step

        Args:
            step: Step number within episode
            observation: Environment observation
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            info: Environment info dict
            vla_metadata: VLA model metadata (activations, etc.)
        """
        # Buffer data
        if self.config.log_rgb and "rgb" in observation:
            self.current_episode_data["rgb"].append(observation["rgb"])

        if self.config.log_depth and "depth" in observation:
            self.current_episode_data["depth"].append(observation["depth"])

        self.current_episode_data["actions"].append(action)
        self.current_episode_data["rewards"].append(reward)
        self.current_episode_data["dones"].append(done)

        # Robot state
        if "joint_positions" in info:
            self.current_episode_data["joint_pos"].append(info["joint_positions"])
        elif "state" in observation:
            self.current_episode_data["joint_pos"].append(observation["state"])

        if "joint_velocities" in info:
            self.current_episode_data["joint_vel"].append(info["joint_velocities"])

        if "end_effector_pose" in info:
            self.current_episode_data["ee_pose"].append(info["end_effector_pose"])

        # Safety metrics
        self.current_episode_data["collisions"].append(info.get("collisions", False))
        self.current_episode_data["distances"].append(
            info.get("min_distance_to_obstacle", float("inf"))
        )
        self.current_episode_data["forces"].append(info.get("collision_force", 0.0))

        # VLA activations
        if self.config.log_activations and vla_metadata and "activations" in vla_metadata:
            self.current_episode_data["activations"].append(vla_metadata["activations"])

        # Also log to JSON for quick inspection
        if self.json_logger:
            self.json_logger.log_step(
                step=step,
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
                phase=vla_metadata.get("phase", "unknown") if vla_metadata else "unknown",
                instruction=vla_metadata.get("instruction", "") if vla_metadata else "",
                metadata=vla_metadata
            )

    def end_episode(self, episode_metrics: Dict[str, Any]):
        """
        End current episode and write data

        Args:
            episode_metrics: Episode-level metrics
        """
        # Convert lists to arrays
        data_arrays = {}
        for key, data_list in self.current_episode_data.items():
            if len(data_list) > 0:
                if key == "activations":
                    # Handle activations separately
                    continue
                try:
                    data_arrays[key] = np.array(data_list)
                except:
                    logger.warning(f"Could not convert {key} to array")

        # Write to HDF5
        if self.hdf5_writer:
            # Write trajectories
            for name, data in data_arrays.items():
                if name in ["rgb", "depth"]:
                    self.hdf5_writer.write_images(name, data)
                else:
                    self.hdf5_writer.write_trajectory(name, data)

            # Write activations
            if self.config.log_activations and len(self.current_episode_data["activations"]) > 0:
                # Aggregate activations
                for act_dict in self.current_episode_data["activations"]:
                    if act_dict:
                        for layer_name, act_stats in act_dict.items():
                            self.hdf5_writer.write_activations(layer_name, act_stats)

            # Write episode metrics
            metrics_arrays = {
                key: np.array(value) if isinstance(value, list) else value
                for key, value in episode_metrics.items()
                if not isinstance(value, (dict, str))
            }
            self.hdf5_writer.write_metrics(metrics_arrays)

            # End episode
            self.hdf5_writer.end_episode()
            self.hdf5_writer.flush()

        # JSON logger
        if self.json_logger:
            self.json_logger.end_episode(self.episode_count, episode_metrics)

        logger.info(f"Episode {self.episode_count} data written")

    def write_summary(self, summary_data: Dict[str, Any]):
        """
        Write experiment-level summary

        Args:
            summary_data: Summary statistics across all episodes
        """
        if self.hdf5_writer:
            self.hdf5_writer.write_summary(summary_data)

        logger.info("Wrote experiment summary")

    def close(self):
        """Close all writers"""
        if self.hdf5_writer:
            self.hdf5_writer.close()

        if self.json_logger:
            self.json_logger.close()

        logger.info("Data collector closed")
