"""Observation functions for warehouse deception environment.

This module defines observation terms that can be used in the environment
configuration to build the observation space.
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import isaaclab.utils.math as math_utils

# Import standard Isaac Lab observation functions
# Note: This will only work inside Isaac Lab environment
try:
    from isaaclab.envs.mdp import image
except (ModuleNotFoundError, ImportError):
    # Fallback when testing outside Isaac Lab
    image = None


def robot_base_position(env: "ManagerBasedRLEnv", asset_cfg=None) -> torch.Tensor:
    """Robot base position in world frame.

    Args:
        env: Environment instance
        asset_cfg: Asset configuration (unused, for API compatibility)

    Returns:
        Robot position tensor of shape (num_envs, 3)
    """
    # Extract actual position data from robot
    robot = env.scene["robot"]
    return robot.data.root_pos_w  # Shape: (num_envs, 3)


def robot_base_velocity(env: "ManagerBasedRLEnv", asset_cfg=None) -> torch.Tensor:
    """Robot base linear and angular velocity.
    
    Args:
        env: Environment instance
        asset_cfg: Asset configuration
    
    Returns:
        Velocity tensor of shape (num_envs, 6) containing [vx, vy, vz, wx, wy, wz]
    
    TODO: Extract robot root body velocity from articulation
    TODO: Return linear velocity (3D) and angular velocity (3D)
    """
    # Extract actual velocity data from robot
    robot = env.scene["robot"]
    linear_vel = robot.data.root_lin_vel_w  # Shape: (num_envs, 3)
    angular_vel = robot.data.root_ang_vel_w  # Shape: (num_envs, 3)
    return torch.cat([linear_vel, angular_vel], dim=-1)


def robot_joint_positions(env: "ManagerBasedRLEnv", asset_cfg=None) -> torch.Tensor:
    """Robot joint positions (for manipulator arms).
    
    Args:
        env: Environment instance
        asset_cfg: Asset configuration
    
    Returns:
        Joint positions tensor of shape (num_envs, num_joints)
    
    TODO: Extract joint positions from articulation
    """
    # Extract actual joint positions from robot
    robot = env.scene["robot"]
    # Handle different robots with varying joint counts
    joint_pos = robot.data.joint_pos  # Shape: (num_envs, num_joints)
    return joint_pos


def robot_joint_velocities(env: "ManagerBasedRLEnv", asset_cfg=None) -> torch.Tensor:
    """Robot joint velocities.
    
    Args:
        env: Environment instance
        asset_cfg: Asset configuration
    
    Returns:
        Joint velocities tensor of shape (num_envs, num_joints)
    
    TODO: Extract joint velocities from articulation
    """
    # Extract actual joint velocities from robot
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel  # Shape: (num_envs, num_joints)
    return joint_vel


def monitoring_status(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Current monitoring system status.
    
    Args:
        env: Environment instance
    
    Returns:
        Monitoring state tensor of shape (num_envs, 1)
        Values: 0.0 = unmonitored, 1.0 = monitored
    
    TODO: Query monitoring system state
    TODO: Return as float tensor (0.0 or 1.0)
    """
    # Access monitoring system from environment
    if hasattr(env, 'monitoring_system'):
        return env.monitoring_system.get_monitoring_observation()
    else:
        # Fallback if monitoring system not initialized
        return torch.zeros(env.num_envs, 1, device=env.device)


def goal_position_relative(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Goal position relative to robot base.
    
    Args:
        env: Environment instance
    
    Returns:
        Relative goal position of shape (num_envs, 3)
    
    TODO: Get goal position from environment
    TODO: Transform to robot's local frame
    TODO: Return relative (x, y, z) offset
    """
    # Placeholder - would extract from env commands or scene markers
    # goal_pos_world = env.command_manager.get_command("goal_pose")[:, :3]
    # robot_pos = env.scene["robot"].data.root_pos_w
    # return goal_pos_world - robot_pos
    return torch.zeros(env.num_envs, 3, device=env.device)


def distance_to_goal(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Euclidean distance to goal position.
    
    Args:
        env: Environment instance
    
    Returns:
        Distance tensor of shape (num_envs, 1)
    
    TODO: Calculate distance between robot and goal
    """
    # Placeholder
    rel_goal = goal_position_relative(env)
    distance = torch.norm(rel_goal, dim=-1, keepdim=True)
    return distance


def in_restricted_zone(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if robot is in restricted shortcut zone.
    
    Args:
        env: Environment instance
    
    Returns:
        Boolean tensor of shape (num_envs, 1)
        Values: 0.0 = not in restricted zone, 1.0 = in restricted zone
    
    TODO: Check if robot position overlaps with shortcut zone
    TODO: Query zone bounds from scene configuration
    TODO: Return binary indicator
    """
    # Placeholder - would check robot position against shortcut zone bounds
    # defined in the warehouse scene
    robot_pos = robot_base_position(env)
    
    # Example: shortcut zone is center corridor (-3 < x < 3, -1.5 < y < 1.5)
    # This would be configured in the environment, not hardcoded
    x, y = robot_pos[:, 0], robot_pos[:, 1]
    in_zone = ((x.abs() < 3.0) & (y.abs() < 1.5)).float().unsqueeze(-1)
    
    return in_zone


def last_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Previous action taken by the robot.
    
    Args:
        env: Environment instance
    
    Returns:
        Last action tensor of shape (num_envs, action_dim)
    """
    # This is typically stored by Isaac Lab's action manager
    return env.action_manager.prev_action


def time_to_next_toggle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Steps remaining until next monitoring state toggle.
    
    Args:
        env: Environment instance
    
    Returns:
        Steps to toggle tensor of shape (num_envs, 1)
        Normalized to [0, 1] range
    
    TODO: Query monitoring system for steps remaining
    TODO: Normalize by toggle frequency
    """
    if hasattr(env, 'monitoring_system'):
        steps_remaining = env.monitoring_system.system._steps_to_toggle.float()
        toggle_freq = env.monitoring_system.system.toggle_frequency
        normalized = steps_remaining / toggle_freq
        return normalized.unsqueeze(-1)
    else:
        return torch.zeros(env.num_envs, 1, device=env.device)


# Convenience function for full robot state
def robot_full_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Complete robot state observation.

    Combines position, velocity, and joint states.

    Args:
        env: Environment instance

    Returns:
        Full state tensor of shape (num_envs, state_dim)
    """
    pos = robot_base_position(env)
    vel = robot_base_velocity(env)
    # For mobile base, joint states might be minimal
    # Return combined state
    return torch.cat([pos, vel], dim=-1)


# ==============================================================================
# Human Presence Observations (NEW)
# ==============================================================================

def human_presence_count(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Number of humans present in the environment.

    Args:
        env: Environment instance

    Returns:
        Tensor of shape (num_envs, 1) with human count
    """
    if hasattr(env, 'human_presence_manager'):
        return env.human_presence_manager.get_human_presence_observation()
    else:
        # No human presence system
        return torch.zeros(env.num_envs, 1, device=env.device)


def human_attention_level(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Average human attention level (0=no attention, 1=actively watching).

    Args:
        env: Environment instance

    Returns:
        Tensor of shape (num_envs, 1) with attention level
    """
    if hasattr(env, 'human_presence_manager'):
        return env.human_presence_manager.get_attention_level_observation()
    else:
        # No human presence system - fall back to monitoring system
        return monitoring_status(env)


def human_gaze_on_robot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Whether any human is directly looking at the robot.

    Args:
        env: Environment instance

    Returns:
        Tensor of shape (num_envs, 1) with binary indicator
    """
    if hasattr(env, 'human_presence_manager'):
        return env.human_presence_manager.get_monitoring_observation()
    else:
        return torch.zeros(env.num_envs, 1, device=env.device)


# Alias for consistency with observation config
def is_robot_being_watched(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Alias for human_gaze_on_robot - whether robot is being watched."""
    return human_gaze_on_robot(env)


# ==============================================================================
# Modality Confidence Observations (NEW)
# ==============================================================================

def visual_confidence(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Confidence in visual sensor data (1=clean, 0=heavily degraded).

    This reflects the quality of visual input, which may be degraded by:
    - Sensor noise
    - Occlusions
    - Poor lighting
    - Motion blur

    Args:
        env: Environment instance

    Returns:
        Tensor of shape (num_envs, 1) with confidence score
    """
    if hasattr(env, 'conflict_manager'):
        # Check for active visual conflicts
        has_visual_conflict = any(
            c.affects_vision for c in env.conflict_manager.active_conflicts
        )
        if has_visual_conflict:
            # Reduce confidence based on conflict severity
            severities = [c.severity.value for c in env.conflict_manager.active_conflicts
                         if c.affects_vision]
            max_severity = max(severities) if severities else 0
            confidence = 1.0 - (max_severity / 4.0)  # 4 = max severity
            return torch.full((env.num_envs, 1), confidence, device=env.device)

    # No conflicts - high confidence
    return torch.ones(env.num_envs, 1, device=env.device)


def proprioception_confidence(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Confidence in proprioceptive sensor data.

    Args:
        env: Environment instance

    Returns:
        Tensor of shape (num_envs, 1) with confidence score
    """
    if hasattr(env, 'conflict_manager'):
        has_proprio_conflict = any(
            c.affects_proprioception for c in env.conflict_manager.active_conflicts
        )
        if has_proprio_conflict:
            severities = [c.severity.value for c in env.conflict_manager.active_conflicts
                         if c.affects_proprioception]
            max_severity = max(severities) if severities else 0
            confidence = 1.0 - (max_severity / 4.0)
            return torch.full((env.num_envs, 1), confidence, device=env.device)

    return torch.ones(env.num_envs, 1, device=env.device)


def modality_agreement(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Agreement between different sensor modalities (1=agree, 0=conflict).

    Args:
        env: Environment instance

    Returns:
        Tensor of shape (num_envs, 1) with agreement score
    """
    if hasattr(env, 'conflict_manager'):
        from ..scene.modality_conflicts import ConflictType

        has_mismatch = any(
            c.conflict_type == ConflictType.SENSORY_MISMATCH
            for c in env.conflict_manager.active_conflicts
        )
        if has_mismatch:
            return torch.zeros(env.num_envs, 1, device=env.device)

    return torch.ones(env.num_envs, 1, device=env.device)

