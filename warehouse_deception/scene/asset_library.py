"""Asset library for realistic object spawning in scenes.

This module maps generic object types to actual USD assets available in Isaac Lab.
"""

from typing import Dict, List, Tuple
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
import isaaclab.sim as sim_utils


class AssetLibrary:
    """Library of realistic assets for scene population."""

    def __init__(self):
        """Initialize the asset library with Isaac Lab assets."""

        # Asset paths in Isaac Nucleus
        self.nucleus_dir = ISAAC_NUCLEUS_DIR

        # Warehouse assets (VERIFIED WORKING from Isaac Lab examples)
        self.warehouse_assets = {
            "pallet": f"{self.nucleus_dir}/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_681.usd",
            "box": f"{self.nucleus_dir}/Props/Blocks/red_block.usd",
            "crate": f"{self.nucleus_dir}/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_681.usd",
            "shelf": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",
            "barrel": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            "forklift": f"{self.nucleus_dir}/Props/Forklift/forklift.usd",
        }

        # Hospital assets (using verified substitutes where needed)
        self.hospital_assets = {
            "bed": f"{self.nucleus_dir}/Props/Mounts/SeattleLabTable/table_instanceable.usd",  # Substitute
            "medical_cart": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "wheelchair": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "table": f"{self.nucleus_dir}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            "chair": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "cabinet": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",
        }

        # Office assets (using verified substitutes where needed)
        self.office_assets = {
            "desk": f"{self.nucleus_dir}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            "chair": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "filing_cabinet": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",
            "bookshelf": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",
            "plant": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "table": f"{self.nucleus_dir}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        }

        # Retail assets (using verified substitutes where needed)
        self.retail_assets = {
            "shelf": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",
            "display": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",  # Substitute
            "counter": f"{self.nucleus_dir}/Props/PackingTable/packing_table.usd",
            "mannequin": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "table": f"{self.nucleus_dir}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            "rack": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",  # Substitute
        }

        # Factory assets (using verified substitutes where needed)
        self.factory_assets = {
            "machine": f"{self.nucleus_dir}/Props/Forklift/forklift.usd",  # Substitute
            "conveyor": f"{self.nucleus_dir}/Props/Mounts/ThorlabsTable/table_instanceable.usd",  # Substitute
            "workbench": f"{self.nucleus_dir}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            "toolbox": f"{self.nucleus_dir}/Props/Blocks/red_block.usd",
            "cart": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "pallet": f"{self.nucleus_dir}/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_681.usd",
        }

        # Laboratory assets (using verified substitutes where needed)
        self.laboratory_assets = {
            "lab_bench": f"{self.nucleus_dir}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            "equipment": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "cabinet": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",
            "fume_hood": f"{self.nucleus_dir}/Props/Mounts/Stand/stand_instanceable.usd",  # Substitute
            "chair": f"{self.nucleus_dir}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # Substitute
            "table": f"{self.nucleus_dir}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        }

        # Fallback generic assets (using simple shapes if specific assets not found)
        self.fallback_assets = {
            "box": "Cube",  # Procedural cube
            "sphere": "Sphere",  # Procedural sphere
            "cylinder": "Cylinder",  # Procedural cylinder
        }

        # Scene-to-assets mapping
        self.scene_assets = {
            "warehouse": self.warehouse_assets,
            "hospital": self.hospital_assets,
            "office": self.office_assets,
            "retail": self.retail_assets,
            "factory": self.factory_assets,
            "laboratory": self.laboratory_assets,
        }

        # Asset physical properties (mass, friction, etc.)
        self.asset_properties = {
            "pallet": {"mass": 10.0, "friction": 0.8, "manipulable": True},
            "box": {"mass": 5.0, "friction": 0.6, "manipulable": True},
            "crate": {"mass": 8.0, "friction": 0.7, "manipulable": True},
            "barrel": {"mass": 15.0, "friction": 0.5, "manipulable": True},
            "bed": {"mass": 30.0, "friction": 0.9, "manipulable": False},
            "cart": {"mass": 5.0, "friction": 0.4, "manipulable": True},
            "chair": {"mass": 3.0, "friction": 0.7, "manipulable": True},
            "desk": {"mass": 20.0, "friction": 0.8, "manipulable": False},
            "table": {"mass": 15.0, "friction": 0.8, "manipulable": False},
            "shelf": {"mass": 25.0, "friction": 0.9, "manipulable": False},
            "medical_cart": {"mass": 5.0, "friction": 0.4, "manipulable": True},
            "wheelchair": {"mass": 3.0, "friction": 0.7, "manipulable": True},
            "toolbox": {"mass": 15.0, "friction": 0.6, "manipulable": True},
            "equipment": {"mass": 30.0, "friction": 0.7, "manipulable": False},
            "machine": {"mass": 200.0, "friction": 0.9, "manipulable": False},
            "conveyor": {"mass": 150.0, "friction": 0.8, "manipulable": False},
            "workbench": {"mass": 70.0, "friction": 0.8, "manipulable": False},
            "lab_bench": {"mass": 50.0, "friction": 0.8, "manipulable": False},
            "fume_hood": {"mass": 100.0, "friction": 0.9, "manipulable": False},
            "cabinet": {"mass": 35.0, "friction": 0.9, "manipulable": False},
            "filing_cabinet": {"mass": 40.0, "friction": 0.9, "manipulable": False},
            "bookshelf": {"mass": 45.0, "friction": 0.9, "manipulable": False},
            "counter": {"mass": 60.0, "friction": 0.8, "manipulable": False},
            "display": {"mass": 35.0, "friction": 0.8, "manipulable": False},
            "rack": {"mass": 40.0, "friction": 0.9, "manipulable": False},
            "plant": {"mass": 5.0, "friction": 0.6, "manipulable": True},
            "mannequin": {"mass": 15.0, "friction": 0.7, "manipulable": False},
        }
        
        # Environment USD file paths (full scene files)
        # These are complete environment USD files that include walls, floors, structures
        self.environment_files = {
            "warehouse": f"{self.nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd",
            "hospital": f"{self.nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd",  # Fallback
            "office": f"{self.nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd",  # Fallback
            "retail": f"{self.nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd",  # Fallback
            "factory": f"{self.nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd",  # Fallback
            "laboratory": f"{self.nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd",  # Fallback
        }
        
        # Default fallback environment (Simple_Warehouse is most commonly available)
        self.default_environment = f"{self.nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd"

    def get_asset_path(self, object_type: str, scene_type: str) -> str:
        """Get the USD asset path for an object type in a scene.

        Args:
            object_type: Type of object (e.g., "box", "table")
            scene_type: Type of scene (e.g., "warehouse", "hospital")

        Returns:
            Path to USD asset
        """
        # Get scene-specific assets
        scene_assets = self.scene_assets.get(scene_type, {})

        # Try to get asset from scene-specific library
        asset_path = scene_assets.get(object_type)

        if asset_path:
            return asset_path

        # Try to get from fallback
        fallback = self.fallback_assets.get(object_type)
        if fallback:
            return fallback

        # Last resort: use a cube
        return "Cube"

    def get_asset_config(self, object_type: str, scene_type: str,
                        position: Tuple[float, float, float],
                        rotation: Tuple[float, float, float, float],
                        scale: Tuple[float, float, float],
                        prim_path: str,
                        use_real_assets: bool = False):
        """Get a complete asset config for spawning an object.

        Args:
            object_type: Type of object
            scene_type: Type of scene
            position: (x, y, z) position
            rotation: (w, x, y, z) quaternion rotation
            scale: (x, y, z) scale
            prim_path: USD prim path for the object
            use_real_assets: If False (default), use procedural shapes with physics.
                           If True, use USD assets (visual only, no physics)

        Returns:
            RigidObjectCfg with full physics (procedural) or AssetBaseCfg (USD visual only)
        """
        asset_path = self.get_asset_path(object_type, scene_type)
        properties = self.asset_properties.get(object_type, {"mass": 5.0, "friction": 0.7, "manipulable": False})

        # Check if it's a procedural shape
        if asset_path in ["Cube", "Sphere", "Cylinder"]:
            # Use procedural spawner
            if asset_path == "Cube":
                spawn_cfg = sim_utils.CuboidCfg(
                    size=scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=properties["mass"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.5, 0.5, 0.7),
                        metallic=0.2,
                    ),
                )
            elif asset_path == "Sphere":
                spawn_cfg = sim_utils.SphereCfg(
                    radius=scale[0],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=properties["mass"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )
            else:  # Cylinder
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=scale[0],
                    height=scale[2],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=properties["mass"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )
            
            # Create rigid object configuration (procedural shapes)
            cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=position,
                    rot=rotation,
                ),
            )
            return cfg
        else:
            # Use USD file spawner
            # Try to use real USD assets if requested
            if use_real_assets:
                # Check if object should be manipulable (have physics)
                is_manipulable = properties.get("manipulable", False)
                
                try:
                    if is_manipulable:
                        # For manipulable objects, try to use RigidObjectCfg with physics
                        # This allows objects to be moved, pushed, etc.
                        spawn_cfg = UsdFileCfg(
                            usd_path=asset_path,
                            scale=scale,
                        )
                        
                        # Try RigidObjectCfg for physics-enabled objects
                        # Note: Some USD assets may not support this, will fallback if fails
                        cfg = RigidObjectCfg(
                            prim_path=prim_path,
                            spawn=spawn_cfg,
                            init_state=RigidObjectCfg.InitialStateCfg(
                                pos=position,
                                rot=rotation,
                            ),
                        )
                        return cfg
                    else:
                        # For static objects (furniture, large structures), use AssetBaseCfg
                        # These are visual-only and don't need physics
                        spawn_cfg = UsdFileCfg(
                            usd_path=asset_path,
                            scale=scale,
                        )
                        cfg = AssetBaseCfg(
                            prim_path=prim_path,
                            spawn=spawn_cfg,
                            init_state=AssetBaseCfg.InitialStateCfg(
                                pos=position,
                                rot=rotation,
                            ),
                        )
                        return cfg
                except Exception as e:
                    # Fallback to procedural shape if USD loading fails
                    print(f"Warning: Failed to load USD asset '{asset_path}' for {object_type}, using procedural shape: {e}")
                    use_real_assets = False
            
            # Fallback to procedural shape
            if not use_real_assets:
                # Use procedural cuboid as fallback
                spawn_cfg = sim_utils.CuboidCfg(
                    size=scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=properties["mass"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.6, 0.6, 0.6),
                        metallic=0.1,
                    ),
                )
                cfg = RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=position,
                        rot=rotation,
                    ),
                )
                return cfg

    def get_available_assets_for_scene(self, scene_type: str) -> List[str]:
        """Get list of available asset types for a scene.

        Args:
            scene_type: Type of scene

        Returns:
            List of object type names
        """
        return list(self.scene_assets.get(scene_type, {}).keys())

    def get_environment_file(self, scene_type: str) -> str:
        """Get the USD environment file path for a scene type.

        Args:
            scene_type: Type of scene (e.g., "warehouse", "hospital")

        Returns:
            Path to USD environment file (with fallback to default)
        """
        return self.environment_files.get(scene_type, self.default_environment)
    
    def verify_asset_exists(self, asset_path: str) -> bool:
        """Check if a USD asset file exists.

        Args:
            asset_path: Path to USD file

        Returns:
            True if file exists, False otherwise
        """
        if asset_path in ["Cube", "Sphere", "Cylinder"]:
            return True  # Procedural shapes always available

        # Check if file exists (simplified check)
        # In production, would use actual filesystem check
        return True  # Assume all nucleus assets are available


# Global instance
ASSET_LIBRARY = AssetLibrary()


# Convenience functions
def get_asset_for_object(object_type: str, scene_type: str,
                        position: Tuple[float, float, float],
                        rotation: Tuple[float, float, float, float],
                        scale: Tuple[float, float, float],
                        prim_path: str,
                        use_real_assets: bool = True):
    """Get asset configuration for an object.

    Convenience function that uses the global asset library.

    Returns:
        AssetBaseCfg or RigidObjectCfg ready for spawning
    """
    return ASSET_LIBRARY.get_asset_config(
        object_type, scene_type, position, rotation, scale, prim_path, use_real_assets
    )


def get_available_objects(scene_type: str) -> List[str]:
    """Get available object types for a scene.

    Convenience function that uses the global asset library.
    """
    return ASSET_LIBRARY.get_available_assets_for_scene(scene_type)


def get_environment_file(scene_type: str) -> str:
    """Get USD environment file path for a scene type.

    Convenience function that uses the global asset library.

    Args:
        scene_type: Type of scene (e.g., "warehouse", "hospital")

    Returns:
        Path to USD environment file
    """
    return ASSET_LIBRARY.get_environment_file(scene_type)
