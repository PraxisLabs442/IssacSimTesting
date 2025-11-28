"""Action configuration for different robot types.

This module provides action space configurations for various robot platforms
compatible with Isaac Lab's action manager.
"""

from isaaclab.managers import ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.envs.mdp as mdp

# Import robot type for compatibility
try:
    from ..scene.scene_randomizer import RobotType
    SCENE_RANDOMIZER_AVAILABLE = True
except ImportError:
    SCENE_RANDOMIZER_AVAILABLE = False
    RobotType = None


def get_mobile_base_actions() -> dict:
    """Get action configuration for mobile base robots (differential drive).

    Returns 2D action space for wheel velocities: [left_wheel, right_wheel]
    """
    from isaaclab.envs.mdp.actions import JointVelocityActionCfg

    return {
        "wheel_velocity": JointVelocityActionCfg(
            class_type=mdp.JointVelocityAction,
            asset_name="robot",
            joint_names=[".*"],  # All joints (both wheels for Jetbot)
            scale=10.0,  # Scale factor for wheel velocities
        )
    }


def get_manipulator_actions() -> dict:
    """Get action configuration for manipulator arms (joint control).

    Returns 7D action space for arm joints + gripper control
    """
    from isaaclab.envs.mdp.actions import JointPositionActionCfg, BinaryJointPositionActionCfg

    return {
        "arm_action": JointPositionActionCfg(
            class_type=mdp.JointPositionAction,
            asset_name="robot",
            joint_names=["panda_joint.*"],  # Adjust for specific manipulator
            scale=0.5,
            use_default_offset=True,
        ),
        "gripper_action": BinaryJointPositionActionCfg(
            class_type=mdp.BinaryJointPositionAction,
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        ),
    }


def get_quadruped_actions() -> dict:
    """Get action configuration for quadruped robots (leg control).

    Returns 12D action space: 3 joints per leg x 4 legs
    """
    from isaaclab.envs.mdp.actions import JointPositionActionCfg

    return {
        "joint_actions": JointPositionActionCfg(
            class_type=mdp.JointPositionAction,
            asset_name="robot",
            joint_names=[".*HAA", ".*HFE", ".*KFE"],  # ANYmal joint pattern
            scale=0.5,
            use_default_offset=True,
        )
    }


def get_humanoid_actions() -> dict:
    """Get action configuration for humanoid robots.

    Returns action space for humanoid joints (varies by robot)
    """
    from isaaclab.envs.mdp.actions import JointPositionActionCfg

    # Placeholder - would be configured based on specific humanoid robot
    return {
        "joint_actions": JointPositionActionCfg(
            class_type=mdp.JointPositionAction,
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
    }


def get_jetbot_actions() -> dict:
    """Get action configuration specifically for NVIDIA Jetbot robot."""
    from isaaclab.envs.mdp.actions import JointVelocityActionCfg

    return {
        "wheel_velocity": JointVelocityActionCfg(
            class_type=mdp.JointVelocityAction,
            asset_name="robot",
            joint_names=[".*"],
            scale=10.0,
        )
    }


def get_franka_actions() -> dict:
    """Get action configuration specifically for Franka Panda robot."""
    from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
    from isaaclab.controllers import DifferentialIKControllerCfg

    return {
        "arm_action": DifferentialInverseKinematicsActionCfg(
            class_type=mdp.DifferentialInverseKinematicsAction,
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", ik_method="dls"),
            scale=0.5,
        ),
        "gripper_action": BinaryJointPositionActionCfg(
            class_type=mdp.BinaryJointPositionAction,
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        ),
    }


def get_anymal_actions() -> dict:
    """Get action configuration specifically for ANYmal C quadruped."""
    from isaaclab.envs.mdp.actions import JointPositionActionCfg

    return {
        "joint_actions": JointPositionActionCfg(
            class_type=mdp.JointPositionAction,
            asset_name="robot",
            joint_names=[".*HAA", ".*HFE", ".*KFE"],
            scale=0.5,
            use_default_offset=True,
        )
    }


# Robot-specific action configurations
ROBOT_ACTIONS = {
    "jetbot": get_jetbot_actions,
    "franka": get_franka_actions,
    "anymal": get_anymal_actions,
    "ridgeback": get_mobile_base_actions,  # Generic mobile base
    "simple_base": get_mobile_base_actions,
}


# Mapping from RobotType enum to action configurations
if SCENE_RANDOMIZER_AVAILABLE:
    ROBOT_TYPE_ACTIONS = {
        RobotType.MOBILE_BASE: get_mobile_base_actions,
        RobotType.MANIPULATOR: get_manipulator_actions,
        RobotType.QUADRUPED: get_quadruped_actions,
        RobotType.HUMANOID: get_humanoid_actions,
    }


def get_actions_for_robot(robot_name: str) -> dict:
    """Get action configuration for a specific robot by name.

    Args:
        robot_name: Name of the robot (carter, franka, anymal, etc.)

    Returns:
        Dictionary of ActionTermCfg for the robot

    Raises:
        ValueError: If robot name is unknown
    """
    if robot_name not in ROBOT_ACTIONS:
        # Fallback to mobile base
        print(f"Warning: Unknown robot '{robot_name}', using mobile base actions")
        return get_mobile_base_actions()

    return ROBOT_ACTIONS[robot_name]()


def get_actions_for_robot_type(robot_type) -> dict:
    """Get action configuration for a robot type from enum.

    Args:
        robot_type: RobotType enum value

    Returns:
        Dictionary of ActionTermCfg for the robot type

    Raises:
        ValueError: If scene randomizer not available
    """
    if not SCENE_RANDOMIZER_AVAILABLE:
        raise ValueError("Scene randomizer not available")

    if robot_type not in ROBOT_TYPE_ACTIONS:
        print(f"Warning: Unknown robot type '{robot_type}', using mobile base actions")
        return get_mobile_base_actions()

    return ROBOT_TYPE_ACTIONS[robot_type]()


def get_action_dim_for_robot(robot_name: str) -> int:
    """Get the expected action dimensionality for a robot.

    Args:
        robot_name: Name of the robot

    Returns:
        Integer action dimension
    """
    action_dims = {
        "jetbot": 2,  # left_wheel, right_wheel
        "ridgeback": 2,
        "simple_base": 2,
        "franka": 7,  # 6 DoF arm + gripper
        "anymal": 12,  # 3 joints x 4 legs
    }

    return action_dims.get(robot_name, 2)  # Default to 2


def get_action_dim_for_robot_type(robot_type) -> int:
    """Get the expected action dimensionality for a robot type.

    Args:
        robot_type: RobotType enum value

    Returns:
        Integer action dimension
    """
    if not SCENE_RANDOMIZER_AVAILABLE:
        return 2

    action_dims = {
        RobotType.MOBILE_BASE: 2,  # Wheel velocities
        RobotType.MANIPULATOR: 7,
        RobotType.QUADRUPED: 12,
        RobotType.HUMANOID: 10,  # Approximate
    }

    return action_dims.get(robot_type, 2)
