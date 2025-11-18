"""
HDF5 Writer
Efficient compressed storage for large-scale experiments
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class HDF5Writer:
    """
    Efficient HDF5 writer with compression
    Handles images, trajectories, activations, and metadata
    """

    def __init__(
        self,
        file_path: str,
        compression: str = "gzip",
        compression_level: int = 4
    ):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.compression_level = compression_level

        # Create HDF5 file
        self.hdf5_file = h5py.File(self.file_path, "w")

        # Create root groups
        self.episodes_group = self.hdf5_file.create_group("episodes")
        self.summary_group = self.hdf5_file.create_group("summary")

        # Metadata
        self.hdf5_file.attrs["created_at"] = datetime.now().isoformat()
        self.hdf5_file.attrs["version"] = "1.0"

        self.current_episode_group = None
        self.episode_count = 0

        logger.info(f"Created HDF5 file: {self.file_path}")

    def create_episode(self, episode_id: int, metadata: Dict[str, Any]):
        """
        Create new episode group

        Args:
            episode_id: Episode number
            metadata: Episode metadata
        """
        episode_name = f"episode_{episode_id:06d}"
        self.current_episode_group = self.episodes_group.create_group(episode_name)

        # Store metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                self.current_episode_group.attrs[key] = value
            else:
                # Convert complex types to JSON
                self.current_episode_group.attrs[key] = json.dumps(value)

        self.episode_count += 1
        logger.debug(f"Created episode group: {episode_name}")

    def write_trajectory(self, name: str, data: np.ndarray):
        """
        Write trajectory data (actions, states, etc.)

        Args:
            name: Dataset name (e.g., "actions", "joint_positions")
            data: Numpy array (T, D) where T=timesteps, D=dimension
        """
        if self.current_episode_group is None:
            raise ValueError("No active episode. Call create_episode() first.")

        dataset = self.current_episode_group.create_dataset(
            name,
            data=data,
            compression=self.compression,
            compression_opts=self.compression_level
        )

        logger.debug(f"Wrote trajectory: {name}, shape: {data.shape}")

    def write_images(self, name: str, images: np.ndarray):
        """
        Write image sequence with compression

        Args:
            name: Dataset name (e.g., "rgb", "depth")
            images: Numpy array (T, H, W, C) or (T, H, W)
        """
        if self.current_episode_group is None:
            raise ValueError("No active episode. Call create_episode() first.")

        # Use chunking for efficient access to individual frames
        chunk_shape = (1,) + images.shape[1:]

        dataset = self.current_episode_group.create_dataset(
            name,
            data=images,
            compression=self.compression,
            compression_opts=self.compression_level,
            chunks=chunk_shape
        )

        logger.debug(f"Wrote images: {name}, shape: {images.shape}")

    def write_activations(
        self,
        layer_name: str,
        activation_stats: Dict[str, Any]
    ):
        """
        Write activation statistics (not full tensors to save space)

        Args:
            layer_name: Name of layer
            activation_stats: Dictionary with mean, std, shape
        """
        if self.current_episode_group is None:
            raise ValueError("No active episode. Call create_episode() first.")

        # Create activations subgroup if doesn't exist
        if "activations" not in self.current_episode_group:
            act_group = self.current_episode_group.create_group("activations")
        else:
            act_group = self.current_episode_group["activations"]

        # Store statistics as attributes
        layer_group = act_group.create_group(layer_name)
        for key, value in activation_stats.items():
            if isinstance(value, (list, tuple)):
                layer_group.attrs[key] = np.array(value)
            else:
                layer_group.attrs[key] = value

    def write_metrics(self, metrics: Dict[str, np.ndarray]):
        """
        Write per-step metrics

        Args:
            metrics: Dictionary of metric arrays (T,)
        """
        if self.current_episode_group is None:
            raise ValueError("No active episode. Call create_episode() first.")

        # Create metrics subgroup if doesn't exist
        if "metrics" not in self.current_episode_group:
            metrics_group = self.current_episode_group.create_group("metrics")
        else:
            metrics_group = self.current_episode_group["metrics"]

        # Write each metric
        for name, data in metrics.items():
            # HDF5 doesn't allow compression on scalar datasets
            # Only compress arrays (non-scalar data)
            if np.ndim(data) > 0:
                metrics_group.create_dataset(
                    name,
                    data=data,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
            else:
                # Scalar value - no compression
                metrics_group.create_dataset(
                    name,
                    data=data
                )

    def end_episode(self):
        """Mark current episode as complete"""
        if self.current_episode_group is not None:
            self.current_episode_group.attrs["status"] = "complete"
            self.current_episode_group = None

    def write_summary(self, summary_data: Dict[str, Any]):
        """
        Write experiment-level summary

        Args:
            summary_data: Dictionary with summary statistics
        """
        for key, value in summary_data.items():
            if isinstance(value, np.ndarray):
                self.summary_group.create_dataset(key, data=value)
            elif isinstance(value, dict):
                # Create subgroup for nested dict
                subgroup = self.summary_group.create_group(key)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        subgroup.create_dataset(subkey, data=subvalue)
                    else:
                        subgroup.attrs[subkey] = subvalue
            else:
                self.summary_group.attrs[key] = value

        logger.info("Wrote summary data")

    def flush(self):
        """Flush data to disk"""
        self.hdf5_file.flush()

    def close(self):
        """Close HDF5 file"""
        if self.hdf5_file:
            self.hdf5_file.close()
            logger.info(f"Closed HDF5 file: {self.file_path}")
            logger.info(f"Total episodes: {self.episode_count}")
