"""Multi-scene USD builder for different environment types.

Generates procedural USD scenes for warehouse, hospital, office, retail, factory,
and laboratory environments with randomized layouts and objects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .scene_randomizer import SceneConfig, ObjectSpawnInfo, SceneType


class MultiSceneBuilder:
    """Builder for generating multiple scene types as USD files."""

    def __init__(self, output_dir: str = "assets/scenes"):
        """Initialize the multi-scene builder.

        Args:
            output_dir: Directory to save generated USD files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_scene(self, config: SceneConfig, objects: List[ObjectSpawnInfo],
                   output_filename: Optional[str] = None) -> str:
        """Build a complete scene based on configuration.

        Args:
            config: Scene configuration
            objects: List of objects to spawn
            output_filename: Optional custom filename

        Returns:
            Path to generated USD file
        """
        if output_filename is None:
            output_filename = f"{config.scene_type.value}_{config.random_seed}.usd"

        output_path = self.output_dir / output_filename

        # Dispatch to appropriate scene builder
        if config.scene_type == SceneType.WAREHOUSE:
            self._build_warehouse(config, objects, output_path)
        elif config.scene_type == SceneType.HOSPITAL:
            self._build_hospital(config, objects, output_path)
        elif config.scene_type == SceneType.OFFICE:
            self._build_office(config, objects, output_path)
        elif config.scene_type == SceneType.RETAIL:
            self._build_retail(config, objects, output_path)
        elif config.scene_type == SceneType.FACTORY:
            self._build_factory(config, objects, output_path)
        elif config.scene_type == SceneType.LABORATORY:
            self._build_laboratory(config, objects, output_path)
        else:
            raise ValueError(f"Unsupported scene type: {config.scene_type}")

        return str(output_path)

    def _build_warehouse(self, config: SceneConfig, objects: List[ObjectSpawnInfo],
                        output_path: Path):
        """Build a warehouse scene with shelving and storage areas."""
        length, width = config.size

        usd_content = self._create_usd_header(config)

        # Add floor
        usd_content += self._create_floor(length, width)

        # Add perimeter walls
        usd_content += self._create_walls(length, width, height=4.0)

        # Add warehouse-specific elements
        if config.num_aisles:
            usd_content += self._create_aisles(config.num_aisles, length, width)

        # Add loading docks
        usd_content += self._create_loading_docks(length, width)

        # Add objects (pallets, boxes, shelves)
        usd_content += self._create_objects(objects, "warehouse")

        # Add restricted zones
        if config.restricted_zones:
            usd_content += self._create_restricted_zones(config.restricted_zones)

        # Add monitoring cameras
        usd_content += self._create_monitoring_cameras(config.num_cameras, length, width)

        # Add lighting
        usd_content += self._create_lighting("warehouse")

        usd_content += self._create_usd_footer()

        self._write_usd_file(output_path, usd_content)

    def _build_hospital(self, config: SceneConfig, objects: List[ObjectSpawnInfo],
                       output_path: Path):
        """Build a hospital scene with rooms, corridors, and medical equipment."""
        length, width = config.size

        usd_content = self._create_usd_header(config)
        usd_content += self._create_floor(length, width, color=(0.95, 0.95, 0.95))
        usd_content += self._create_walls(length, width, height=3.5)

        # Add hospital rooms
        if config.num_rooms:
            usd_content += self._create_hospital_rooms(config.num_rooms, length, width)

        # Add corridors
        usd_content += self._create_corridors(length, width)

        # Add objects (beds, medical carts, equipment)
        usd_content += self._create_objects(objects, "hospital")

        # Add restricted zones (e.g., surgery areas)
        if config.restricted_zones:
            usd_content += self._create_restricted_zones(config.restricted_zones)

        # Add monitoring cameras
        usd_content += self._create_monitoring_cameras(config.num_cameras, length, width)

        # Add lighting (bright hospital lighting)
        usd_content += self._create_lighting("hospital")

        usd_content += self._create_usd_footer()
        self._write_usd_file(output_path, usd_content)

    def _build_office(self, config: SceneConfig, objects: List[ObjectSpawnInfo],
                     output_path: Path):
        """Build an office scene with cubicles, desks, and meeting rooms."""
        length, width = config.size

        usd_content = self._create_usd_header(config)
        usd_content += self._create_floor(length, width, color=(0.7, 0.7, 0.8))
        usd_content += self._create_walls(length, width, height=3.0)

        # Add office rooms and cubicles
        if config.num_rooms:
            usd_content += self._create_office_rooms(config.num_rooms, length, width)

        if config.num_workstations:
            usd_content += self._create_cubicles(config.num_workstations, length, width)

        # Add objects (desks, chairs, filing cabinets)
        usd_content += self._create_objects(objects, "office")

        # Add restricted zones (e.g., server room, executive areas)
        if config.restricted_zones:
            usd_content += self._create_restricted_zones(config.restricted_zones)

        # Add monitoring cameras
        usd_content += self._create_monitoring_cameras(config.num_cameras, length, width)

        # Add lighting
        usd_content += self._create_lighting("office")

        usd_content += self._create_usd_footer()
        self._write_usd_file(output_path, usd_content)

    def _build_retail(self, config: SceneConfig, objects: List[ObjectSpawnInfo],
                     output_path: Path):
        """Build a retail scene with aisles, displays, and checkout areas."""
        length, width = config.size

        usd_content = self._create_usd_header(config)
        usd_content += self._create_floor(length, width, color=(0.85, 0.85, 0.85))
        usd_content += self._create_walls(length, width, height=4.5)

        # Add retail aisles
        if config.num_aisles:
            usd_content += self._create_retail_aisles(config.num_aisles, length, width)

        # Add checkout area
        usd_content += self._create_checkout_area(length, width)

        # Add objects (shelves, displays, products)
        usd_content += self._create_objects(objects, "retail")

        # Add restricted zones (e.g., storage, employee areas)
        if config.restricted_zones:
            usd_content += self._create_restricted_zones(config.restricted_zones)

        # Add monitoring cameras
        usd_content += self._create_monitoring_cameras(config.num_cameras, length, width)

        # Add lighting (bright retail lighting)
        usd_content += self._create_lighting("retail")

        usd_content += self._create_usd_footer()
        self._write_usd_file(output_path, usd_content)

    def _build_factory(self, config: SceneConfig, objects: List[ObjectSpawnInfo],
                      output_path: Path):
        """Build a factory scene with machines, conveyors, and workstations."""
        length, width = config.size

        usd_content = self._create_usd_header(config)
        usd_content += self._create_floor(length, width, color=(0.5, 0.5, 0.5))
        usd_content += self._create_walls(length, width, height=5.0)

        # Add workstations
        if config.num_workstations:
            usd_content += self._create_factory_workstations(config.num_workstations, length, width)

        # Add conveyor belts
        usd_content += self._create_conveyors(length, width)

        # Add objects (machines, tools, materials)
        usd_content += self._create_objects(objects, "factory")

        # Add restricted zones (e.g., hazardous areas)
        if config.restricted_zones:
            usd_content += self._create_restricted_zones(config.restricted_zones)

        # Add monitoring cameras
        usd_content += self._create_monitoring_cameras(config.num_cameras, length, width)

        # Add lighting (industrial lighting)
        usd_content += self._create_lighting("factory")

        usd_content += self._create_usd_footer()
        self._write_usd_file(output_path, usd_content)

    def _build_laboratory(self, config: SceneConfig, objects: List[ObjectSpawnInfo],
                         output_path: Path):
        """Build a laboratory scene with benches, equipment, and safety zones."""
        length, width = config.size

        usd_content = self._create_usd_header(config)
        usd_content += self._create_floor(length, width, color=(0.9, 0.9, 0.9))
        usd_content += self._create_walls(length, width, height=3.5)

        # Add lab benches
        if config.num_workstations:
            usd_content += self._create_lab_benches(config.num_workstations, length, width)

        # Add fume hoods and safety equipment
        usd_content += self._create_safety_equipment(length, width)

        # Add objects (equipment, instruments, storage)
        usd_content += self._create_objects(objects, "laboratory")

        # Add restricted zones (e.g., hazardous materials)
        if config.restricted_zones:
            usd_content += self._create_restricted_zones(config.restricted_zones)

        # Add monitoring cameras
        usd_content += self._create_monitoring_cameras(config.num_cameras, length, width)

        # Add lighting (bright lab lighting)
        usd_content += self._create_lighting("laboratory")

        usd_content += self._create_usd_footer()
        self._write_usd_file(output_path, usd_content)

    # =========================================================================
    # Helper methods for USD generation
    # =========================================================================

    def _create_usd_header(self, config: SceneConfig) -> str:
        """Create USD file header."""
        return f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Generated {config.scene_type.value} scene"
)

def Xform "World"
{{
'''

    def _create_usd_footer(self) -> str:
        """Create USD file footer."""
        return "}\n"

    def _create_floor(self, length: float, width: float, color: Tuple[float, float, float] = (0.6, 0.6, 0.6)) -> str:
        """Create floor plane."""
        return f'''    def Xform "Floor"
    {{
        def Mesh "FloorMesh"
        {{
            float3[] extent = [({-length/2}, {-width/2}, 0), ({length/2}, {width/2}, 0)]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [({-length/2}, {-width/2}, 0), ({length/2}, {-width/2}, 0), ({length/2}, {width/2}, 0), ({-length/2}, {width/2}, 0)]
            color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
            uniform token subdivisionScheme = "none"
        }}
    }}

'''

    def _create_walls(self, length: float, width: float, height: float) -> str:
        """Create perimeter walls."""
        wall_thickness = 0.2
        content = '    def Xform "Walls"\n    {\n'

        # North wall
        content += f'''        def Cube "NorthWall"
        {{
            double size = 1
            double3 xformOp:scale = ({length}, {wall_thickness}, {height})
            double3 xformOp:translate = (0, {width/2}, {height/2})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]
        }}

'''
        # South wall
        content += f'''        def Cube "SouthWall"
        {{
            double size = 1
            double3 xformOp:scale = ({length}, {wall_thickness}, {height})
            double3 xformOp:translate = (0, {-width/2}, {height/2})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]
        }}

'''
        # East wall
        content += f'''        def Cube "EastWall"
        {{
            double size = 1
            double3 xformOp:scale = ({wall_thickness}, {width}, {height})
            double3 xformOp:translate = ({length/2}, 0, {height/2})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]
        }}

'''
        # West wall
        content += f'''        def Cube "WestWall"
        {{
            double size = 1
            double3 xformOp:scale = ({wall_thickness}, {width}, {height})
            double3 xformOp:translate = ({-length/2}, 0, {height/2})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]
        }}

'''
        content += '    }\n\n'
        return content

    def _create_aisles(self, num_aisles: int, length: float, width: float) -> str:
        """Create aisle shelving (warehouse/retail)."""
        content = '    def Xform "Aisles"\n    {\n'

        aisle_width = 2.0
        shelf_width = 1.0
        spacing = width / (num_aisles + 1)

        for i in range(num_aisles):
            y_pos = -width/2 + spacing * (i + 1)
            content += f'''        def Cube "Aisle{i}"
        {{
            double size = 1
            double3 xformOp:scale = ({length * 0.8}, {shelf_width}, 2.0)
            double3 xformOp:translate = (0, {y_pos}, 1.0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.6, 0.4, 0.2)]
        }}

'''
        content += '    }\n\n'
        return content

    def _create_objects(self, objects: List[ObjectSpawnInfo], scene_type: str) -> str:
        """Create objects in the scene."""
        content = '    def Xform "Objects"\n    {\n'

        for idx, obj in enumerate(objects):
            x, y, z = obj.position
            w, qx, qy, qz = obj.rotation
            sx, sy, sz = obj.scale

            # Simple box placeholder for objects
            # In real implementation, would reference actual USD assets
            content += f'''        def Cube "{obj.object_type}_{idx}"
        {{
            double size = 1
            double3 xformOp:scale = ({sx}, {sy}, {sz * 0.5})
            double3 xformOp:translate = ({x}, {y}, {z + sz * 0.25})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.5, 0.5, 0.7)]
        }}

'''
        content += '    }\n\n'
        return content

    def _create_restricted_zones(self, zones: List[Dict]) -> str:
        """Create visual markers for restricted zones."""
        content = '    def Xform "RestrictedZones"\n    {\n'

        for idx, zone in enumerate(zones):
            x, y, z = zone['center']
            sx, sy, sz = zone['size']

            # Red transparent box to mark restricted zone
            content += f'''        def Cube "RestrictedZone{idx}"
        {{
            double size = 1
            double3 xformOp:scale = ({sx}, {sy}, 0.1)
            double3 xformOp:translate = ({x}, {y}, 0.05)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(1.0, 0.0, 0.0)]
        }}

'''
        content += '    }\n\n'
        return content

    def _create_monitoring_cameras(self, num_cameras: int, length: float, width: float) -> str:
        """Create monitoring camera markers."""
        content = '    def Xform "MonitoringCameras"\n    {\n'

        # Place cameras at corners and center
        positions = [
            (length/2 - 1, width/2 - 1, 3.0),  # NE corner
            (-length/2 + 1, width/2 - 1, 3.0),  # NW corner
            (length/2 - 1, -width/2 + 1, 3.0),  # SE corner
            (-length/2 + 1, -width/2 + 1, 3.0),  # SW corner
            (0, 0, 3.5),  # Center
        ]

        for idx in range(min(num_cameras, len(positions))):
            x, y, z = positions[idx]
            content += f'''        def Sphere "Camera{idx}"
        {{
            double radius = 0.2
            double3 xformOp:translate = ({x}, {y}, {z})
            uniform token[] xformOpOrder = ["xformOp:translate"]
            color3f[] primvars:displayColor = [(0.2, 0.2, 0.2)]
        }}

'''
        content += '    }\n\n'
        return content

    def _create_lighting(self, scene_type: str) -> str:
        """Create appropriate lighting for scene type."""
        intensity_map = {
            "warehouse": 1500.0,
            "hospital": 3000.0,
            "office": 2000.0,
            "retail": 2500.0,
            "factory": 1800.0,
            "laboratory": 2800.0,
        }

        intensity = intensity_map.get(scene_type, 2000.0)

        return f'''    def DistantLight "Light"
    {{
        float intensity = {intensity}
        float3 xformOp:rotateXYZ = (45, 45, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

'''

    # Placeholder methods for specific scene elements
    # These would be expanded with proper implementations

    def _create_loading_docks(self, length: float, width: float) -> str:
        return ""

    def _create_hospital_rooms(self, num_rooms: int, length: float, width: float) -> str:
        return ""

    def _create_corridors(self, length: float, width: float) -> str:
        return ""

    def _create_office_rooms(self, num_rooms: int, length: float, width: float) -> str:
        return ""

    def _create_cubicles(self, num_workstations: int, length: float, width: float) -> str:
        return ""

    def _create_retail_aisles(self, num_aisles: int, length: float, width: float) -> str:
        return self._create_aisles(num_aisles, length, width)

    def _create_checkout_area(self, length: float, width: float) -> str:
        return ""

    def _create_factory_workstations(self, num_workstations: int, length: float, width: float) -> str:
        return ""

    def _create_conveyors(self, length: float, width: float) -> str:
        return ""

    def _create_lab_benches(self, num_workstations: int, length: float, width: float) -> str:
        return ""

    def _create_safety_equipment(self, length: float, width: float) -> str:
        return ""

    def _write_usd_file(self, path: Path, content: str):
        """Write USD content to file."""
        with open(path, 'w') as f:
            f.write(content)
        print(f"Generated scene: {path}")


# Convenience function
def build_random_scene(config: SceneConfig, objects: List[ObjectSpawnInfo],
                      output_dir: str = "assets/scenes") -> str:
    """Build a random scene from configuration.

    Args:
        config: Scene configuration
        objects: Objects to spawn
        output_dir: Output directory for USD file

    Returns:
        Path to generated USD file
    """
    builder = MultiSceneBuilder(output_dir)
    return builder.build_scene(config, objects)
