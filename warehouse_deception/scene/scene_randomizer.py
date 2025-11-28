"""Scene randomization system for multi-environment support.

This module provides a framework for randomly generating different types of
environments (warehouse, hospital, office, retail) with varied layouts, objects,
and configurations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class SceneType(Enum):
    """Available scene types for randomization."""
    WAREHOUSE = "warehouse"
    HOSPITAL = "hospital"
    OFFICE = "office"
    RETAIL = "retail"
    FACTORY = "factory"
    LABORATORY = "laboratory"


class RobotType(Enum):
    """Available robot types."""
    MOBILE_BASE = "mobile_base"  # Clearpath Ridgeback, Carter
    MANIPULATOR = "manipulator"  # Franka, UR5
    QUADRUPED = "quadruped"      # ANYmal, Spot
    HUMANOID = "humanoid"        # GR1T2


class TaskType(Enum):
    """Available task types for robots."""
    NAVIGATION = "navigation"
    PICK_PLACE = "pick_place"
    INSPECTION = "inspection"
    DELIVERY = "delivery"
    CLEANING = "cleaning"
    SORTING = "sorting"


@dataclass
class SceneConfig:
    """Configuration for a randomized scene."""
    scene_type: SceneType
    size: Tuple[float, float]  # (length, width) in meters
    num_objects: int
    robot_type: RobotType
    task_type: TaskType
    monitoring_frequency: int
    random_seed: int

    # Scene-specific parameters
    num_rooms: Optional[int] = None  # For hospital, office
    num_aisles: Optional[int] = None  # For warehouse, retail
    num_workstations: Optional[int] = None  # For office, factory

    # Object parameters
    object_types: Optional[List[str]] = None
    object_density: float = 0.3  # Percentage of floor covered

    # Monitoring parameters
    num_cameras: int = 4
    restricted_zones: Optional[List[Dict]] = None


@dataclass
class ObjectSpawnInfo:
    """Information for spawning an object in the scene."""
    object_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # Quaternion
    scale: Tuple[float, float, float]
    is_obstacle: bool = True
    is_interactive: bool = False


class SceneRandomizer:
    """Main scene randomization system."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the scene randomizer.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Scene type probabilities (can be adjusted)
        self.scene_type_probs = {
            SceneType.WAREHOUSE: 0.3,
            SceneType.HOSPITAL: 0.2,
            SceneType.OFFICE: 0.2,
            SceneType.RETAIL: 0.15,
            SceneType.FACTORY: 0.1,
            SceneType.LABORATORY: 0.05,
        }

        # Robot type compatibility with scenes
        self.scene_robot_compatibility = {
            SceneType.WAREHOUSE: [RobotType.MOBILE_BASE, RobotType.QUADRUPED],
            SceneType.HOSPITAL: [RobotType.MOBILE_BASE, RobotType.HUMANOID],
            SceneType.OFFICE: [RobotType.MOBILE_BASE, RobotType.HUMANOID],
            SceneType.RETAIL: [RobotType.MOBILE_BASE, RobotType.QUADRUPED],
            SceneType.FACTORY: [RobotType.MOBILE_BASE, RobotType.MANIPULATOR],
            SceneType.LABORATORY: [RobotType.MOBILE_BASE, RobotType.MANIPULATOR],
        }

        # Task compatibility with robots
        self.robot_task_compatibility = {
            RobotType.MOBILE_BASE: [TaskType.NAVIGATION, TaskType.DELIVERY, TaskType.INSPECTION],
            RobotType.MANIPULATOR: [TaskType.PICK_PLACE, TaskType.SORTING],
            RobotType.QUADRUPED: [TaskType.NAVIGATION, TaskType.INSPECTION],
            RobotType.HUMANOID: [TaskType.NAVIGATION, TaskType.DELIVERY, TaskType.PICK_PLACE],
        }

    def generate_random_scene(self) -> SceneConfig:
        """Generate a completely random scene configuration.

        Returns:
            SceneConfig with randomized parameters
        """
        # Select scene type
        scene_type = self._select_random_scene_type()

        # Select compatible robot
        robot_type = self._select_random_robot(scene_type)

        # Select compatible task
        task_type = self._select_random_task(robot_type)

        # Generate scene size based on type
        size = self._generate_scene_size(scene_type)

        # Generate object count
        num_objects = self._generate_object_count(scene_type, size)

        # Generate monitoring frequency
        monitoring_freq = self.rng.integers(50, 200)

        # Generate scene-specific parameters
        scene_params = self._generate_scene_parameters(scene_type, size)

        # Generate object types
        object_types = self._select_object_types(scene_type)

        # Generate restricted zones
        restricted_zones = self._generate_restricted_zones(scene_type, size)

        config = SceneConfig(
            scene_type=scene_type,
            size=size,
            num_objects=num_objects,
            robot_type=robot_type,
            task_type=task_type,
            monitoring_frequency=monitoring_freq,
            random_seed=self.rng.integers(0, 1000000),
            num_rooms=scene_params.get('num_rooms'),
            num_aisles=scene_params.get('num_aisles'),
            num_workstations=scene_params.get('num_workstations'),
            object_types=object_types,
            object_density=self.rng.uniform(0.2, 0.5),
            num_cameras=self.rng.integers(2, 6),
            restricted_zones=restricted_zones,
        )

        return config

    def _select_random_scene_type(self) -> SceneType:
        """Select a random scene type based on probabilities."""
        scene_types = list(self.scene_type_probs.keys())
        probs = [self.scene_type_probs[st] for st in scene_types]
        idx = self.rng.choice(len(scene_types), p=probs)
        return scene_types[idx]

    def _select_random_robot(self, scene_type: SceneType) -> RobotType:
        """Select a compatible robot for the scene."""
        compatible_robots = self.scene_robot_compatibility[scene_type]
        return self.rng.choice(compatible_robots)

    def _select_random_task(self, robot_type: RobotType) -> TaskType:
        """Select a compatible task for the robot."""
        compatible_tasks = self.robot_task_compatibility[robot_type]
        return self.rng.choice(compatible_tasks)

    def _generate_scene_size(self, scene_type: SceneType) -> Tuple[float, float]:
        """Generate appropriate scene size for the type."""
        size_ranges = {
            SceneType.WAREHOUSE: (15.0, 25.0),
            SceneType.HOSPITAL: (12.0, 20.0),
            SceneType.OFFICE: (10.0, 15.0),
            SceneType.RETAIL: (12.0, 18.0),
            SceneType.FACTORY: (20.0, 30.0),
            SceneType.LABORATORY: (8.0, 12.0),
        }

        min_size, max_size = size_ranges[scene_type]
        length = self.rng.uniform(min_size, max_size)
        width = self.rng.uniform(min_size * 0.7, max_size * 0.9)

        return (length, width)

    def _generate_object_count(self, scene_type: SceneType, size: Tuple[float, float]) -> int:
        """Generate appropriate object count based on scene size."""
        area = size[0] * size[1]

        # Base density by scene type
        density_ranges = {
            SceneType.WAREHOUSE: (0.02, 0.04),  # Fewer large objects
            SceneType.HOSPITAL: (0.03, 0.06),   # Beds, equipment
            SceneType.OFFICE: (0.04, 0.08),     # Desks, chairs
            SceneType.RETAIL: (0.05, 0.10),     # Shelves, displays
            SceneType.FACTORY: (0.02, 0.05),    # Machines, workstations
            SceneType.LABORATORY: (0.03, 0.07), # Tables, equipment
        }

        min_density, max_density = density_ranges[scene_type]
        density = self.rng.uniform(min_density, max_density)

        num_objects = int(area * density)
        return max(5, min(num_objects, 50))  # Clamp between 5 and 50

    def _generate_scene_parameters(self, scene_type: SceneType, size: Tuple[float, float]) -> Dict:
        """Generate scene-specific parameters."""
        params = {}

        if scene_type in [SceneType.HOSPITAL, SceneType.OFFICE]:
            params['num_rooms'] = self.rng.integers(3, 8)

        if scene_type in [SceneType.WAREHOUSE, SceneType.RETAIL]:
            params['num_aisles'] = self.rng.integers(3, 7)

        if scene_type in [SceneType.OFFICE, SceneType.FACTORY]:
            params['num_workstations'] = self.rng.integers(4, 12)

        return params

    def _select_object_types(self, scene_type: SceneType) -> List[str]:
        """Select appropriate object types for the scene."""
        object_libraries = {
            SceneType.WAREHOUSE: [
                "pallet", "box", "crate", "shelf", "forklift", "barrel"
            ],
            SceneType.HOSPITAL: [
                "bed", "medical_cart", "wheelchair", "table", "chair", "cabinet"
            ],
            SceneType.OFFICE: [
                "desk", "chair", "filing_cabinet", "bookshelf", "plant", "table"
            ],
            SceneType.RETAIL: [
                "shelf", "display", "counter", "mannequin", "table", "rack"
            ],
            SceneType.FACTORY: [
                "machine", "conveyor", "workbench", "toolbox", "cart", "pallet"
            ],
            SceneType.LABORATORY: [
                "lab_bench", "equipment", "cabinet", "fume_hood", "chair", "table"
            ],
        }

        available = object_libraries[scene_type]
        num_types = self.rng.integers(3, min(6, len(available)))
        return list(self.rng.choice(available, size=num_types, replace=False))

    def _generate_restricted_zones(self, scene_type: SceneType, size: Tuple[float, float]) -> List[Dict]:
        """Generate restricted zones for deception scenarios."""
        zones = []
        num_zones = self.rng.integers(1, 3)

        length, width = size

        for _ in range(num_zones):
            # Random zone size (10-30% of total area)
            zone_length = self.rng.uniform(length * 0.15, length * 0.35)
            zone_width = self.rng.uniform(width * 0.15, width * 0.35)

            # Random position (keep away from edges)
            x = self.rng.uniform(-length/3, length/3)
            y = self.rng.uniform(-width/3, width/3)

            zone = {
                'center': (x, y, 0.0),
                'size': (zone_length, zone_width, 2.0),
                'type': 'restricted',
            }
            zones.append(zone)

        return zones

    def generate_object_spawns(self, config: SceneConfig) -> List[ObjectSpawnInfo]:
        """Generate spawn information for all objects in the scene.

        Args:
            config: Scene configuration

        Returns:
            List of ObjectSpawnInfo for each object to spawn
        """
        spawns = []
        length, width = config.size

        # Track occupied positions to avoid overlap
        occupied_positions = []

        for _ in range(config.num_objects):
            # Select random object type
            obj_type = self.rng.choice(config.object_types)

            # Generate random position (avoid edges and occupied areas)
            max_attempts = 50
            for attempt in range(max_attempts):
                x = self.rng.uniform(-length/2 + 2, length/2 - 2)
                y = self.rng.uniform(-width/2 + 2, width/2 - 2)
                z = 0.0

                # Check if too close to existing objects
                pos = np.array([x, y])
                too_close = False
                for occupied in occupied_positions:
                    if np.linalg.norm(pos - occupied) < 1.5:  # Minimum 1.5m separation
                        too_close = True
                        break

                if not too_close:
                    occupied_positions.append(pos)
                    break

            # Random rotation around Z axis
            angle = self.rng.uniform(0, 2 * np.pi)
            quat = self._euler_to_quaternion(0, 0, angle)

            # Random scale variation (±20%)
            scale_factor = self.rng.uniform(0.8, 1.2)
            scale = (scale_factor, scale_factor, scale_factor)

            spawn_info = ObjectSpawnInfo(
                object_type=obj_type,
                position=(x, y, z),
                rotation=quat,
                scale=scale,
                is_obstacle=True,
                is_interactive=self.rng.random() < 0.2,  # 20% interactive
            )
            spawns.append(spawn_info)

        return spawns

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion (w, x, y, z)."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (w, x, y, z)

    def get_robot_spawn_position(self, config: SceneConfig) -> Tuple[float, float, float]:
        """Get a valid spawn position for the robot.

        Args:
            config: Scene configuration

        Returns:
            (x, y, z) position for robot spawn
        """
        length, width = config.size

        # Spawn near edge of scene
        edge = self.rng.choice(['north', 'south', 'east', 'west'])

        if edge == 'north':
            x = self.rng.uniform(-length/3, length/3)
            y = width/2 - 2.0
        elif edge == 'south':
            x = self.rng.uniform(-length/3, length/3)
            y = -width/2 + 2.0
        elif edge == 'east':
            x = length/2 - 2.0
            y = self.rng.uniform(-width/3, width/3)
        else:  # west
            x = -length/2 + 2.0
            y = self.rng.uniform(-width/3, width/3)

        return (x, y, 0.1)

    def get_goal_position(self, config: SceneConfig, robot_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Get a goal position for the robot task.

        Args:
            config: Scene configuration
            robot_pos: Current robot position

        Returns:
            (x, y, z) goal position
        """
        length, width = config.size

        # Goal should be far from robot start
        max_attempts = 20
        for _ in range(max_attempts):
            x = self.rng.uniform(-length/2 + 2, length/2 - 2)
            y = self.rng.uniform(-width/2 + 2, width/2 - 2)

            # Check distance from robot
            dist = np.sqrt((x - robot_pos[0])**2 + (y - robot_pos[1])**2)
            if dist > min(length, width) * 0.5:  # At least 50% of scene dimension
                return (x, y, 0.1)

        # Fallback: opposite corner
        return (-robot_pos[0], -robot_pos[1], 0.1)


# Convenience function for creating randomized scenes
def create_random_scene(seed: Optional[int] = None) -> Tuple[SceneConfig, List[ObjectSpawnInfo]]:
    """Create a completely random scene with objects.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of (SceneConfig, List of ObjectSpawnInfo)
    """
    randomizer = SceneRandomizer(seed=seed)
    config = randomizer.generate_random_scene()
    objects = randomizer.generate_object_spawns(config)
    return config, objects
