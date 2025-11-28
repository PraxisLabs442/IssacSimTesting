"""Dynamic object spawning system for multi-scene environments.

This module creates scene-appropriate objects based on randomized configurations.
Uses procedural shapes (cuboids, spheres, cylinders) with full physics simulation.

Note: USD assets from Isaac Nucleus are instanced meshes and cannot have RigidBodyAPI applied,
so they are not suitable for physics interactions. Procedural shapes provide scientifically
accurate physics with collision detection, forces, and dynamics.
"""

from typing import Dict, List
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .scene_randomizer import SceneConfig, ObjectSpawnInfo
from .asset_library import AssetLibrary, ASSET_LIBRARY, get_asset_for_object


# Object templates define the visual and physical properties for each object type
OBJECT_TEMPLATES = {
    # ===== WAREHOUSE OBJECTS =====
    "pallet": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        "height_offset": 0.08,
    },
    "box": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.3)),
        ),
        "height_offset": 0.3,
    },
    "crate": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
        "height_offset": 0.25,
    },
    "shelf": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.0, 0.4, 1.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        "height_offset": 0.9,
    },
    "forklift": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.5, 1.0, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.7, 0.1)),
        ),
        "height_offset": 0.6,
    },
    "barrel": {
        "spawn": sim_utils.CylinderCfg(
            radius=0.3,
            height=0.9,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.7)),
        ),
        "height_offset": 0.45,
    },

    # ===== HOSPITAL OBJECTS =====
    "bed": {
        "spawn": sim_utils.CuboidCfg(
            size=(2.0, 1.0, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.95)),
        ),
        "height_offset": 0.3,
    },
    "medical_cart": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.6, 0.4, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
        ),
        "height_offset": 0.45,
    },
    "wheelchair": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.6, 0.7, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.6)),
        ),
        "height_offset": 0.45,
    },
    "cabinet": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.8, 0.5, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=35.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        "height_offset": 0.6,
    },

    # ===== OFFICE OBJECTS =====
    "desk": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.4, 0.7, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        "height_offset": 0.375,
    },
    "chair": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.3)),
        ),
        "height_offset": 0.45,
    },
    "filing_cabinet": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.5, 0.6, 1.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        ),
        "height_offset": 0.65,
    },
    "bookshelf": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.0, 0.3, 1.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.2)),
        ),
        "height_offset": 0.9,
    },
    "plant": {
        "spawn": sim_utils.CylinderCfg(
            radius=0.25,
            height=0.8,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        "height_offset": 0.4,
    },
    "table": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.0, 0.6, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        "height_offset": 0.375,
    },

    # ===== RETAIL OBJECTS =====
    "shelf": {  # Retail shelf (different from warehouse)
        "spawn": sim_utils.CuboidCfg(
            size=(1.2, 0.4, 1.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        "height_offset": 0.9,
    },
    "display": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.0, 0.5, 1.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=35.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.9)),
        ),
        "height_offset": 0.7,
    },
    "counter": {
        "spawn": sim_utils.CuboidCfg(
            size=(2.0, 0.6, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
        ),
        "height_offset": 0.45,
    },
    "mannequin": {
        "spawn": sim_utils.CylinderCfg(
            radius=0.25,
            height=1.7,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.8, 0.7)),
        ),
        "height_offset": 0.85,
    },
    "rack": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.5, 0.4, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        "height_offset": 0.75,
    },

    # ===== FACTORY OBJECTS =====
    "machine": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.5, 1.2, 1.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=200.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.4)),
        ),
        "height_offset": 0.8,
    },
    "conveyor": {
        "spawn": sim_utils.CuboidCfg(
            size=(3.0, 0.6, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=150.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        "height_offset": 0.4,
    },
    "workbench": {
        "spawn": sim_utils.CuboidCfg(
            size=(2.0, 0.8, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.3)),
        ),
        "height_offset": 0.45,
    },
    "toolbox": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.6, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        "height_offset": 0.2,
    },
    "cart": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.8, 0.5, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.7)),
        ),
        "height_offset": 0.45,
    },

    # ===== LABORATORY OBJECTS =====
    "lab_bench": {
        "spawn": sim_utils.CuboidCfg(
            size=(2.0, 0.7, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        "height_offset": 0.45,
    },
    "equipment": {
        "spawn": sim_utils.CuboidCfg(
            size=(0.6, 0.5, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.8)),
        ),
        "height_offset": 0.4,
    },
    "fume_hood": {
        "spawn": sim_utils.CuboidCfg(
            size=(1.5, 0.8, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.9)),
        ),
        "height_offset": 1.0,
    },
}


def create_object_cfg(
    obj_name: str,
    obj_type: str,
    position: tuple,
    rotation: tuple,
    scale: tuple
) -> RigidObjectCfg:
    """Create a RigidObjectCfg for a specific object instance.

    Args:
        obj_name: Unique name for this object instance (e.g., "Box_01")
        obj_type: Type of object from OBJECT_TEMPLATES (e.g., "box", "desk")
        position: (x, y, z) spawn position
        rotation: (w, x, y, z) quaternion rotation
        scale: (sx, sy, sz) scale factors

    Returns:
        RigidObjectCfg ready to be added to scene
    """
    if obj_type not in OBJECT_TEMPLATES:
        # Fallback to generic box for unknown types
        print(f"Warning: Unknown object type '{obj_type}', using generic box")
        template = OBJECT_TEMPLATES["box"]
    else:
        template = OBJECT_TEMPLATES[obj_type]

    # Adjust Z position based on object height
    height_offset = template["height_offset"]
    adj_position = (position[0], position[1], height_offset)

    # Create the configuration
    cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{obj_name}",
        spawn=template["spawn"],
        init_state=RigidObjectCfg.InitialStateCfg(pos=adj_position),
    )

    return cfg


def generate_scene_objects(scene_config: SceneConfig, use_real_assets: bool = False) -> Dict[str, any]:
    """Generate all objects for a scene based on configuration.

    Creates procedural shapes with full physics simulation for scientific accuracy.

    Args:
        scene_config: Scene configuration with object types and count
        use_real_assets: If False (default), use procedural shapes with physics.
                        If True, attempt USD assets (not recommended - no physics)

    Returns:
        Dictionary mapping object names to RigidObjectCfg instances with physics
    """
    from .scene_randomizer import SceneRandomizer

    # Create randomizer with same seed
    randomizer = SceneRandomizer(seed=scene_config.random_seed)

    # Generate spawn information for all objects
    spawn_infos = randomizer.generate_object_spawns(scene_config)

    # Create object configs for each spawn
    objects = {}
    successful_usd = 0
    procedural_fallback = 0
    
    for idx, spawn_info in enumerate(spawn_infos):
        obj_name = f"{spawn_info.object_type.title()}_{idx:02d}"
        prim_path = f"{{ENV_REGEX_NS}}/{obj_name}"

        # Try to use AssetLibrary to get real USD assets first
        cfg = None
        if use_real_assets:
            try:
                cfg = get_asset_for_object(
                    object_type=spawn_info.object_type,
                    scene_type=scene_config.scene_type.value,
                    position=spawn_info.position,
                    rotation=spawn_info.rotation,
                    scale=spawn_info.scale,
                    prim_path=prim_path,
                    use_real_assets=use_real_assets
                )
                # Check if we got a USD asset (AssetBaseCfg) or procedural (RigidObjectCfg)
                if isinstance(cfg, AssetBaseCfg):
                    successful_usd += 1
                else:
                    procedural_fallback += 1
            except Exception as e:
                # Fallback to procedural shape if USD loading fails
                cfg = None
        
        # Fallback to procedural shape if USD not available or failed
        if cfg is None:
            try:
                cfg = create_object_cfg(
                    obj_name=obj_name,
                    obj_type=spawn_info.object_type,
                    position=spawn_info.position,
                    rotation=spawn_info.rotation,
                    scale=spawn_info.scale,
                )
                procedural_fallback += 1
            except Exception as e:
                print(f"Error: Failed to create object '{spawn_info.object_type}' (procedural fallback): {e}")
                continue
        
        # Add object to dictionary
        objects[f"object_{idx:02d}"] = cfg

    # Print summary
    total = len(objects)
    if total > 0:
        print(f"  ✓ Spawned {total} objects:")
        if successful_usd > 0:
            print(f"    - {successful_usd} USD assets (real objects)")
        if procedural_fallback > 0:
            print(f"    - {procedural_fallback} procedural shapes (fallback)")
    else:
        print(f"  ⚠ No objects spawned")

    return objects
