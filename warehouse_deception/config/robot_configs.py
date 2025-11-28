"""Robot configuration templates for warehouse environment.

This module provides pre-configured robot setups that can be used
with the warehouse deception environment.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def get_franka_panda_cfg() -> ArticulationCfg:
    """Get Franka Panda robot configuration.

    Returns:
        ArticulationCfg for Franka Panda arm
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )


def get_clearpath_ridgeback_cfg() -> ArticulationCfg:
    """Get Clearpath Ridgeback mobile base configuration.
    
    Returns:
        ArticulationCfg for Ridgeback mobile platform
    
    TODO: Configure Ridgeback mobile base
    TODO: Set wheel joint parameters
    """
    # Note: This is a placeholder - actual Ridgeback config would depend on
    # available assets in Isaac Sim
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Ridgeback/ridgeback.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def get_jetbot_cfg() -> ArticulationCfg:
    """Get NVIDIA Jetbot mobile robot configuration.

    Returns:
        ArticulationCfg for Jetbot robot
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=10.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )


def get_anymal_c_cfg() -> ArticulationCfg:
    """Get ANYmal C quadruped robot configuration.

    Returns:
        ArticulationCfg for ANYmal C
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*HAA": 0.0,  # Hip abduction-adduction
                ".*HFE": 0.4,  # Hip flexion-extension
                ".*KFE": -0.8,  # Knee flexion-extension
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                effort_limit=80.0,
                velocity_limit=7.5,
                stiffness=80.0,
                damping=2.0,
            ),
        },
    )


def get_simple_mobile_base_cfg() -> ArticulationCfg:
    """Get simple mobile base for testing.
    
    Creates a basic mobile platform using primitive shapes.
    Useful for testing environment logic without complex robot assets.
    
    Returns:
        ArticulationCfg for simple mobile base
    
    TODO: Create simple mobile base with basic geometry
    """
    # This would use procedural generation or a simple USD primitive
    # For now, placeholder that would be implemented as needed
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Simple/mobile_base.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# Import RobotType enum if available (for scene randomizer integration)
try:
    from ..scene.scene_randomizer import RobotType
    SCENE_RANDOMIZER_AVAILABLE = True
except ImportError:
    SCENE_RANDOMIZER_AVAILABLE = False
    RobotType = None


# Robot selector helper
ROBOT_CONFIGS = {
    "franka": get_franka_panda_cfg,
    "ridgeback": get_clearpath_ridgeback_cfg,
    "jetbot": get_jetbot_cfg,
    "anymal": get_anymal_c_cfg,
    "simple_base": get_simple_mobile_base_cfg,
}


# Mapping from RobotType enum to configuration functions
if SCENE_RANDOMIZER_AVAILABLE:
    ROBOT_TYPE_TO_CONFIG = {
        RobotType.MOBILE_BASE: get_jetbot_cfg,  # Default mobile base
        RobotType.MANIPULATOR: get_franka_panda_cfg,  # Default manipulator
        RobotType.QUADRUPED: get_anymal_c_cfg,  # Default quadruped
        RobotType.HUMANOID: get_franka_panda_cfg,  # Fallback to Franka for now
    }


def get_robot_cfg(robot_name: str) -> ArticulationCfg:
    """Get robot configuration by name.

    Args:
        robot_name: Name of robot ("franka", "ridgeback", "carter", "anymal", "simple_base")

    Returns:
        ArticulationCfg for specified robot

    Raises:
        ValueError: If robot name not recognized
    """
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(
            f"Unknown robot: {robot_name}. "
            f"Available robots: {list(ROBOT_CONFIGS.keys())}"
        )

    return ROBOT_CONFIGS[robot_name]()


def get_robot_cfg_from_type(robot_type) -> ArticulationCfg:
    """Get robot configuration from RobotType enum.

    Args:
        robot_type: RobotType enum value

    Returns:
        ArticulationCfg for robot type

    Raises:
        ValueError: If scene randomizer not available or robot type unknown
    """
    if not SCENE_RANDOMIZER_AVAILABLE:
        raise ValueError("Scene randomizer not available")

    if robot_type not in ROBOT_TYPE_TO_CONFIG:
        raise ValueError(f"Unknown robot type: {robot_type}")

    return ROBOT_TYPE_TO_CONFIG[robot_type]()

