"""
Safety Metrics Calculator
Computes collision rates, near-misses, constraint violations, etc.
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafetyThresholds:
    """Thresholds for safety metrics"""
    collision_force_threshold: float = 10.0  # Newtons
    near_miss_distance: float = 0.05  # meters
    joint_velocity_limit: float = 2.0  # rad/s
    workspace_bounds: List[List[float]] = None  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]


class SafetyMetricsCalculator:
    """
    Calculate safety metrics from environment info and trajectories
    """

    def __init__(self, thresholds: SafetyThresholds):
        self.thresholds = thresholds
        self.episode_events = []

    def compute_step_metrics(self, info: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute safety metrics for a single step

        Args:
            info: Environment info dict
            observation: Current observation

        Returns:
            metrics: Dict of safety metrics
        """
        metrics = {}

        # Collision detection
        metrics["collision"] = info.get("collisions", False)
        metrics["collision_force"] = info.get("collision_force", 0.0)
        metrics["collision_severe"] = metrics["collision_force"] > self.thresholds.collision_force_threshold

        # Near-miss detection
        min_distance = info.get("min_distance_to_obstacle", float("inf"))
        metrics["near_miss"] = 0.0 < min_distance < self.thresholds.near_miss_distance
        metrics["min_distance"] = min_distance

        # Joint limit violations
        metrics["joint_limit_violations"] = info.get("joint_limit_violations", 0)

        # Workspace violations
        if "end_effector_position" in info:
            ee_pos = info["end_effector_position"]
            metrics["workspace_violation"] = self._check_workspace_violation(ee_pos)
        else:
            metrics["workspace_violation"] = False

        # Velocity violations
        if "joint_velocities" in info:
            joint_vels = info["joint_velocities"]
            metrics["velocity_violation"] = np.any(np.abs(joint_vels) > self.thresholds.joint_velocity_limit)
            metrics["max_velocity"] = float(np.max(np.abs(joint_vels)))
        else:
            metrics["velocity_violation"] = False
            metrics["max_velocity"] = 0.0

        # Record events
        if metrics["collision"]:
            self.episode_events.append({"type": "collision", "severity": metrics["collision_severe"]})
        if metrics["near_miss"]:
            self.episode_events.append({"type": "near_miss", "distance": min_distance})

        return metrics

    def compute_episode_metrics(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate safety metrics for entire episode

        Args:
            trajectory: List of step data

        Returns:
            metrics: Aggregate safety metrics
        """
        metrics = {
            "total_steps": len(trajectory),
            "collision_count": 0,
            "severe_collision_count": 0,
            "near_miss_count": 0,
            "joint_violation_count": 0,
            "workspace_violation_count": 0,
            "velocity_violation_count": 0,
            "min_distance_mean": 0.0,
            "min_distance_min": float("inf"),
            "max_velocity_mean": 0.0,
            "max_velocity_max": 0.0,
            "safety_score": 0.0
        }

        if not trajectory:
            return metrics

        # Aggregate over trajectory
        distances = []
        velocities = []

        for step in trajectory:
            step_metrics = step.get("info", {})

            if step_metrics.get("collision", False):
                metrics["collision_count"] += 1
                if step_metrics.get("collision_severe", False):
                    metrics["severe_collision_count"] += 1

            if step_metrics.get("near_miss", False):
                metrics["near_miss_count"] += 1

            if step_metrics.get("joint_limit_violations", 0) > 0:
                metrics["joint_violation_count"] += 1

            if step_metrics.get("workspace_violation", False):
                metrics["workspace_violation_count"] += 1

            if step_metrics.get("velocity_violation", False):
                metrics["velocity_violation_count"] += 1

            min_dist = step_metrics.get("min_distance", float("inf"))
            if min_dist < float("inf"):
                distances.append(min_dist)

            max_vel = step_metrics.get("max_velocity", 0.0)
            velocities.append(max_vel)

        # Compute statistics
        if distances:
            metrics["min_distance_mean"] = float(np.mean(distances))
            metrics["min_distance_min"] = float(np.min(distances))

        if velocities:
            metrics["max_velocity_mean"] = float(np.mean(velocities))
            metrics["max_velocity_max"] = float(np.max(velocities))

        # Compute safety score (0-1, higher is safer)
        metrics["safety_score"] = self._compute_safety_score(metrics)

        return metrics

    def _check_workspace_violation(self, position: np.ndarray) -> bool:
        """Check if position is outside workspace bounds"""
        if self.thresholds.workspace_bounds is None:
            return False

        for i, (min_val, max_val) in enumerate(self.thresholds.workspace_bounds):
            if not (min_val <= position[i] <= max_val):
                return True
        return False

    def _compute_safety_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute overall safety score (0-1, higher is safer)

        Penalizes:
        - Collisions (especially severe)
        - Near misses
        - Constraint violations
        """
        score = 1.0

        # Collision penalties
        collision_rate = metrics["collision_count"] / max(metrics["total_steps"], 1)
        score -= collision_rate * 0.5

        severe_collision_rate = metrics["severe_collision_count"] / max(metrics["total_steps"], 1)
        score -= severe_collision_rate * 0.3

        # Near miss penalty
        near_miss_rate = metrics["near_miss_count"] / max(metrics["total_steps"], 1)
        score -= near_miss_rate * 0.1

        # Constraint violation penalties
        joint_violation_rate = metrics["joint_violation_count"] / max(metrics["total_steps"], 1)
        score -= joint_violation_rate * 0.05

        workspace_violation_rate = metrics["workspace_violation_count"] / max(metrics["total_steps"], 1)
        score -= workspace_violation_rate * 0.05

        return max(0.0, min(1.0, score))

    def reset(self):
        """Reset episode event tracking"""
        self.episode_events = []

    def get_episode_events(self) -> List[Dict[str, Any]]:
        """Get list of safety events in episode"""
        return self.episode_events
