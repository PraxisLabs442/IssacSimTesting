"""Scene generation and management for deception detection environments."""

# Scene randomizer (pure Python, no Isaac Lab dependency)
from .scene_randomizer import (
    SceneRandomizer,
    SceneConfig,
    ObjectSpawnInfo,
    SceneType,
    RobotType,
    TaskType,
    create_random_scene,
)

# Domain configuration (pure Python)
from .domain_config import (
    DomainConfig,
    LightingProfile,
    ClutterLevel,
    LightingCondition,
    HumanPresence,
    DOMAIN_CONFIGS,
    get_domain_config,
    get_lighting_profile,
    sample_clutter_level,
)

# Modality conflicts (pure Python, but uses torch/numpy)
from .modality_conflicts import (
    ConflictType,
    ConflictSeverity,
    ModalityConflict,
    ConflictScenario,
    LanguageVisualConflict,
    SensorNoiseInjection,
    OcclusionGenerator,
    ConflictManager,
    TRUST_TEST_SCENARIOS,
)

# Human presence (pure Python, but uses torch)
from .human_presence import (
    HumanState,
    HumanAvatar,
    HumanPresenceManager,
    HumanPresenceScenario,
    HUMAN_PRESENCE_SCENARIOS,
    apply_human_presence_scenario,
)

# Isaac Lab-dependent modules (imported conditionally)
try:
    from .warehouse_builder import WarehouseSceneBuilder
    from .monitoring_system import MonitoringSystem, MonitoringSystemManager
    from .multi_scene_builder import MultiSceneBuilder, build_random_scene
    from .asset_library import AssetLibrary, ASSET_LIBRARY, get_asset_for_object, get_available_objects, get_environment_file
    from .object_spawner import generate_scene_objects, create_object_cfg, OBJECT_TEMPLATES
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    ISAAC_LAB_AVAILABLE = False
    WarehouseSceneBuilder = None
    MonitoringSystem = None
    MonitoringSystemManager = None
    MultiSceneBuilder = None
    build_random_scene = None
    AssetLibrary = None
    ASSET_LIBRARY = None
    get_asset_for_object = None
    get_available_objects = None
    generate_scene_objects = None
    create_object_cfg = None
    OBJECT_TEMPLATES = None

__all__ = [
    # Core randomization
    "SceneRandomizer",
    "SceneConfig",
    "ObjectSpawnInfo",
    "SceneType",
    "RobotType",
    "TaskType",
    "create_random_scene",
    # Domain configuration
    "DomainConfig",
    "LightingProfile",
    "ClutterLevel",
    "LightingCondition",
    "HumanPresence",
    "DOMAIN_CONFIGS",
    "get_domain_config",
    "get_lighting_profile",
    "sample_clutter_level",
    # Modality conflicts
    "ConflictType",
    "ConflictSeverity",
    "ModalityConflict",
    "ConflictScenario",
    "LanguageVisualConflict",
    "SensorNoiseInjection",
    "OcclusionGenerator",
    "ConflictManager",
    "TRUST_TEST_SCENARIOS",
    # Human presence
    "HumanState",
    "HumanAvatar",
    "HumanPresenceManager",
    "HumanPresenceScenario",
    "HUMAN_PRESENCE_SCENARIOS",
    "apply_human_presence_scenario",
]

# Add Isaac Lab modules if available
if ISAAC_LAB_AVAILABLE:
    __all__.extend([
        "WarehouseSceneBuilder",
        "MonitoringSystem",
        "MonitoringSystemManager",
        "MultiSceneBuilder",
        "build_random_scene",
        "AssetLibrary",
        "ASSET_LIBRARY",
        "get_asset_for_object",
        "get_available_objects",
        "get_environment_file",
        "generate_scene_objects",
        "create_object_cfg",
        "OBJECT_TEMPLATES",
    ])

