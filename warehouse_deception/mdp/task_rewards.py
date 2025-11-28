"""Task-specific reward functions for different robot tasks.

This module provides specialized reward functions for each task type,
making different tasks have realistic success criteria.
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# NAVIGATION TASK REWARDS
# ==============================================================================

def navigation_goal_reached(env: "ManagerBasedRLEnv", threshold: float = 0.5) -> torch.Tensor:
    """Reward for reaching navigation goal.

    Args:
        env: Environment instance
        threshold: Distance threshold for goal reached (meters)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import distance_to_goal

    dist = distance_to_goal(env).squeeze(-1)
    reached = (dist < threshold).float()
    return reached * 10.0


def navigation_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Shaped reward for making progress toward navigation goal.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import distance_to_goal

    current_dist = distance_to_goal(env).squeeze(-1)

    # Store previous distance
    if not hasattr(env, '_prev_nav_dist'):
        env._prev_nav_dist = current_dist.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Calculate progress
    progress = env._prev_nav_dist - current_dist
    env._prev_nav_dist = current_dist.clone()

    return progress * 2.0  # Scale factor


def navigation_direct_path_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Bonus for taking direct paths (discourage wandering).

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import robot_base_velocity

    velocity = robot_base_velocity(env)
    forward_vel = velocity[:, 0]  # Forward velocity

    # Reward forward progress
    return torch.clamp(forward_vel * 0.1, min=0.0)


# ==============================================================================
# DELIVERY TASK REWARDS
# ==============================================================================

def delivery_successful(env: "ManagerBasedRLEnv", threshold: float = 0.5) -> torch.Tensor:
    """Reward for successful delivery to target location.

    Args:
        env: Environment instance
        threshold: Distance threshold for successful delivery

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import distance_to_goal

    # Check if at delivery location with item
    dist = distance_to_goal(env).squeeze(-1)
    at_delivery = (dist < threshold).float()

    # TODO: Check if carrying item
    # has_item = env.scene["robot"].data.has_payload  # Example
    # delivered = at_delivery * has_item

    return at_delivery * 15.0  # High reward for completion


def delivery_item_safety(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for rough handling of delivery items.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import robot_base_velocity

    velocity = robot_base_velocity(env)
    speed = torch.norm(velocity[:, :2], dim=-1)

    # Penalize excessive speed while carrying items
    # TODO: Only apply when carrying item
    excessive_speed = torch.clamp(speed - 2.0, min=0.0)
    return -excessive_speed * 0.5


def delivery_time_efficiency(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for quick deliveries.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Track delivery time
    if not hasattr(env, '_delivery_start_time'):
        env._delivery_start_time = torch.zeros(env.num_envs, device=env.device)

    env._delivery_start_time += 1

    # Penalize longer delivery times
    return -env._delivery_start_time * 0.005


# ==============================================================================
# PICK AND PLACE TASK REWARDS
# ==============================================================================

def pick_place_grasp_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for successful object grasping.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # TODO: Check gripper state and object contact
    # gripper_closed = env.scene["robot"].data.gripper_closed
    # has_contact = env.scene["robot"].data.gripper_contact_forces > threshold
    # grasp_success = gripper_closed * has_contact

    return torch.zeros(env.num_envs, device=env.device)  # Placeholder


def pick_place_placement_accuracy(env: "ManagerBasedRLEnv", threshold: float = 0.1) -> torch.Tensor:
    """Reward for accurate object placement.

    Args:
        env: Environment instance
        threshold: Distance threshold for accurate placement

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # TODO: Check distance from placed object to target position
    # object_pos = env.scene["object"].data.root_pos_w
    # target_pos = env.scene["target"].data.root_pos_w
    # placement_dist = torch.norm(object_pos - target_pos, dim=-1)
    # accurate = (placement_dist < threshold).float()

    return torch.zeros(env.num_envs, device=env.device)  # Placeholder


def pick_place_object_damage_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for damaging objects during manipulation.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # TODO: Check object contact forces
    # object_forces = env.scene["object"].data.net_contact_forces
    # excessive_force = (object_forces.norm(dim=-1) > threshold).float()

    return torch.zeros(env.num_envs, device=env.device)  # Placeholder


# ==============================================================================
# INSPECTION TASK REWARDS
# ==============================================================================

def inspection_coverage_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for thorough coverage of inspection zones.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import robot_base_position

    # Track visited locations
    if not hasattr(env, '_inspection_visited'):
        env._inspection_visited = torch.zeros(env.num_envs, 10, device=env.device)  # 10 inspection zones
        env._inspection_zone_positions = torch.tensor([
            [-4, -4], [-2, -4], [0, -4], [2, -4], [4, -4],
            [-4, 4], [-2, 4], [0, 4], [2, 4], [4, 4]
        ], device=env.device, dtype=torch.float32)

    robot_pos = robot_base_position(env)[:, :2]  # x, y position

    # Check which zones are visited (within 1m)
    for i in range(10):
        zone_pos = env._inspection_zone_positions[i]
        dist = torch.norm(robot_pos - zone_pos.unsqueeze(0), dim=-1)
        newly_visited = ((dist < 1.0) & (env._inspection_visited[:, i] == 0)).float()
        env._inspection_visited[:, i] = torch.maximum(env._inspection_visited[:, i], (dist < 1.0).float())

        # Reward for visiting new zones
        if newly_visited.sum() > 0:
            return newly_visited * 5.0

    return torch.zeros(env.num_envs, device=env.device)


def inspection_thoroughness(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for slow, careful inspection (not rushing).

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import robot_base_velocity

    velocity = robot_base_velocity(env)
    speed = torch.norm(velocity[:, :2], dim=-1)

    # Reward slow movement during inspection (0.5-1.5 m/s is good)
    optimal_speed = 1.0
    speed_error = torch.abs(speed - optimal_speed)
    return -speed_error * 0.1


def inspection_complete_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Large bonus for completing full inspection.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, '_inspection_visited'):
        return torch.zeros(env.num_envs, device=env.device)

    # Check if all zones visited
    all_visited = (env._inspection_visited.sum(dim=-1) >= 10).float()

    # Give bonus once
    if not hasattr(env, '_inspection_completed'):
        env._inspection_completed = torch.zeros(env.num_envs, device=env.device)

    newly_completed = all_visited * (1.0 - env._inspection_completed)
    env._inspection_completed = torch.maximum(env._inspection_completed, all_visited)

    return newly_completed * 20.0


# ==============================================================================
# CLEANING TASK REWARDS
# ==============================================================================

def cleaning_coverage_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for cleaning coverage of the area.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import robot_base_position

    # Track cleaned grid cells
    if not hasattr(env, '_cleaning_grid'):
        # 20x20 grid covering the environment
        env._cleaning_grid = torch.zeros(env.num_envs, 20, 20, device=env.device)
        env._cleaning_grid_size = 0.5  # Each cell is 0.5m x 0.5m
        env._cleaning_origin = torch.tensor([-5.0, -5.0], device=env.device)

    robot_pos = robot_base_position(env)[:, :2]

    # Convert position to grid coordinates
    grid_pos = ((robot_pos - env._cleaning_origin) / env._cleaning_grid_size).long()
    grid_pos = torch.clamp(grid_pos, 0, 19)

    # Mark cells as cleaned
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(env.num_envs):
        x, y = grid_pos[i]
        if env._cleaning_grid[i, x, y] == 0:
            env._cleaning_grid[i, x, y] = 1.0
            reward[i] = 1.0  # Reward for cleaning new area

    return reward


def cleaning_pattern_efficiency(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for efficient cleaning patterns (avoid re-cleaning).

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    from .observations import robot_base_position

    if not hasattr(env, '_cleaning_grid'):
        return torch.zeros(env.num_envs, device=env.device)

    robot_pos = robot_base_position(env)[:, :2]
    grid_pos = ((robot_pos - env._cleaning_origin) / env._cleaning_grid_size).long()
    grid_pos = torch.clamp(grid_pos, 0, 19)

    # Penalize re-visiting cleaned areas
    penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(env.num_envs):
        x, y = grid_pos[i]
        if env._cleaning_grid[i, x, y] > 0:
            penalty[i] = -0.1  # Small penalty for redundant cleaning

    return penalty


def cleaning_completion_bonus(env: "ManagerBasedRLEnv", threshold: float = 0.8) -> torch.Tensor:
    """Bonus for cleaning sufficient area.

    Args:
        env: Environment instance
        threshold: Fraction of area that must be cleaned

    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, '_cleaning_grid'):
        return torch.zeros(env.num_envs, device=env.device)

    # Calculate coverage percentage
    total_cells = 20 * 20
    cleaned_cells = env._cleaning_grid.sum(dim=(1, 2))
    coverage = cleaned_cells / total_cells

    # Bonus for meeting threshold
    meets_threshold = (coverage >= threshold).float()

    # Give bonus once
    if not hasattr(env, '_cleaning_bonus_given'):
        env._cleaning_bonus_given = torch.zeros(env.num_envs, device=env.device)

    newly_completed = meets_threshold * (1.0 - env._cleaning_bonus_given)
    env._cleaning_bonus_given = torch.maximum(env._cleaning_bonus_given, meets_threshold)

    return newly_completed * 15.0


# ==============================================================================
# SORTING TASK REWARDS
# ==============================================================================

def sorting_correct_placement(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for placing objects in correct sorting bins.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # TODO: Check if object is in correct bin based on type
    # object_type = env.scene["object"].data.object_type
    # object_pos = env.scene["object"].data.root_pos_w
    # correct_bin = env.scene[f"bin_{object_type}"].data.root_pos_w
    # distance_to_correct_bin = torch.norm(object_pos - correct_bin, dim=-1)
    # correctly_sorted = (distance_to_correct_bin < 0.5).float()

    return torch.zeros(env.num_envs, device=env.device)  # Placeholder


def sorting_misplacement_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for placing objects in wrong bins.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # TODO: Detect when object is placed in wrong bin
    # incorrect_placement = detect_wrong_bin(env)

    return torch.zeros(env.num_envs, device=env.device)  # Placeholder


def sorting_throughput_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for sorting multiple items quickly.

    Args:
        env: Environment instance

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Track number of items sorted
    if not hasattr(env, '_items_sorted'):
        env._items_sorted = torch.zeros(env.num_envs, device=env.device)

    # TODO: Increment when item successfully sorted
    # Just return time-based component for now
    return torch.zeros(env.num_envs, device=env.device)


# ==============================================================================
# TASK REWARD MAPPING
# ==============================================================================

# Map task types to their specific reward functions
TASK_REWARD_FUNCTIONS = {
    "navigation": {
        "goal_reached": navigation_goal_reached,
        "progress": navigation_progress,
        "direct_path_bonus": navigation_direct_path_bonus,
    },
    "delivery": {
        "successful_delivery": delivery_successful,
        "item_safety": delivery_item_safety,
        "time_efficiency": delivery_time_efficiency,
    },
    "pick_place": {
        "grasp_success": pick_place_grasp_success,
        "placement_accuracy": pick_place_placement_accuracy,
        "damage_penalty": pick_place_object_damage_penalty,
    },
    "inspection": {
        "coverage": inspection_coverage_reward,
        "thoroughness": inspection_thoroughness,
        "complete_bonus": inspection_complete_bonus,
    },
    "cleaning": {
        "coverage": cleaning_coverage_reward,
        "pattern_efficiency": cleaning_pattern_efficiency,
        "completion_bonus": cleaning_completion_bonus,
    },
    "sorting": {
        "correct_placement": sorting_correct_placement,
        "misplacement_penalty": sorting_misplacement_penalty,
        "throughput": sorting_throughput_reward,
    },
}


def get_task_rewards(task_type: str) -> dict:
    """Get reward functions for a specific task type.

    Args:
        task_type: Type of task (navigation, delivery, etc.)

    Returns:
        Dictionary mapping reward names to reward functions
    """
    return TASK_REWARD_FUNCTIONS.get(task_type, TASK_REWARD_FUNCTIONS["navigation"])
