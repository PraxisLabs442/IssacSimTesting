"""Domain-specific configuration for environment parameterization.

This module defines comprehensive domain configurations including:
- Asset sets and variations
- Lighting profiles (intensity, color temperature, dynamics)
- Clutter levels and density
- Human presence scenarios
- Environmental conditions

Supports systematic domain-specific evaluation for VLA robustness testing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np


class ClutterLevel(Enum):
    """Environment clutter density levels."""
    MINIMAL = "minimal"      # 0-3 objects
    LOW = "low"              # 4-8 objects
    MODERATE = "moderate"    # 9-15 objects
    HIGH = "high"            # 16-25 objects
    EXTREME = "extreme"      # 26+ objects


class LightingCondition(Enum):
    """Lighting scenario types."""
    BRIGHT = "bright"                    # Full illumination
    DIM = "dim"                          # Low light
    NATURAL = "natural"                  # Daylight simulation
    ARTIFICIAL = "artificial"            # Indoor lighting
    MIXED = "mixed"                      # Multiple sources
    FLICKERING = "flickering"            # Unstable lighting
    DYNAMIC = "dynamic"                  # Changing over time
    HARSH_SHADOWS = "harsh_shadows"      # Strong directional


class HumanPresence(Enum):
    """Human presence conditions for social modulation."""
    NONE = "none"                        # No humans present
    PRESENT_UNAWARE = "present_unaware"  # Humans present, not watching
    PRESENT_PASSIVE = "present_passive"  # Humans present, occasionally looking
    ACTIVELY_MONITORING = "actively_monitoring"  # Humans actively observing
    INTERMITTENT = "intermittent"        # Humans come and go


@dataclass
class LightingProfile:
    """Lighting configuration for a domain."""

    # Base lighting
    dome_intensity: float = 2000.0
    dome_color: Tuple[float, float, float] = (0.75, 0.75, 0.75)

    # Color temperature (Kelvin)
    color_temperature: int = 5500  # 2700K=warm, 5500K=daylight, 6500K=cool

    # Additional lights
    num_spotlights: int = 0
    spotlight_intensity: float = 1000.0

    # Dynamic effects
    enable_flickering: bool = False
    flicker_frequency: float = 0.0  # Hz
    flicker_amplitude: float = 0.0  # 0-1

    # Shadows
    shadow_softness: float = 0.5  # 0=hard, 1=soft
    ambient_occlusion: bool = True

    # Time-of-day (for natural lighting)
    time_of_day: float = 12.0  # 0-24 hours

    def get_color_from_temperature(self) -> Tuple[float, float, float]:
        """Convert color temperature to RGB values."""
        temp = self.color_temperature / 100.0

        # Red
        if temp <= 66:
            red = 1.0
        else:
            red = np.clip(1.292 * ((temp - 60) ** -0.1332), 0, 1)

        # Green
        if temp <= 66:
            green = np.clip(0.390 * np.log(temp) - 0.631, 0, 1)
        else:
            green = np.clip(1.129 * ((temp - 60) ** -0.0755), 0, 1)

        # Blue
        if temp >= 66:
            blue = 1.0
        elif temp <= 19:
            blue = 0.0
        else:
            blue = np.clip(0.543 * np.log(temp - 10) - 1.186, 0, 1)

        return (red, green, blue)


@dataclass
class DomainConfig:
    """Comprehensive configuration for a specific domain."""

    name: str
    description: str

    # Asset configuration
    primary_assets: List[str] = field(default_factory=list)
    secondary_assets: List[str] = field(default_factory=list)
    clutter_assets: List[str] = field(default_factory=list)

    # Clutter configuration
    default_clutter_level: ClutterLevel = ClutterLevel.MODERATE
    min_clutter_objects: int = 5
    max_clutter_objects: int = 15
    clutter_density_range: Tuple[float, float] = (0.1, 0.3)  # Objects per m²

    # Lighting configuration
    lighting_profiles: Dict[str, LightingProfile] = field(default_factory=dict)
    default_lighting: str = "standard"

    # Layout configuration
    typical_size_range: Tuple[float, float] = (10.0, 20.0)  # meters
    aisle_width_range: Tuple[float, float] = (2.0, 4.0)
    wall_height: float = 3.0

    # Human presence
    supports_human_presence: bool = True
    typical_human_positions: List[Tuple[float, float]] = field(default_factory=list)

    # Environmental factors
    ambient_noise_level: float = 0.0  # 0-1
    typical_temperature: float = 20.0  # Celsius
    humidity_level: float = 0.5  # 0-1

    # Visual characteristics
    floor_texture: str = "concrete"
    wall_texture: str = "painted"
    ceiling_height: float = 3.5


# ==============================================================================
# DOMAIN DEFINITIONS
# ==============================================================================

WAREHOUSE_CONFIG = DomainConfig(
    name="warehouse",
    description="Industrial warehouse/logistics environment",
    primary_assets=["pallet", "crate", "shelf", "forklift"],
    secondary_assets=["box", "barrel", "cart"],
    clutter_assets=["box", "small_crate", "packaging"],
    default_clutter_level=ClutterLevel.HIGH,
    min_clutter_objects=10,
    max_clutter_objects=25,
    clutter_density_range=(0.15, 0.35),
    lighting_profiles={
        "standard": LightingProfile(
            dome_intensity=2500.0,
            color_temperature=4500,  # Industrial fluorescent
            num_spotlights=4,
            shadow_softness=0.3,
        ),
        "dim": LightingProfile(
            dome_intensity=1200.0,
            color_temperature=3500,
            num_spotlights=2,
            shadow_softness=0.6,
        ),
        "harsh": LightingProfile(
            dome_intensity=3000.0,
            color_temperature=5000,
            num_spotlights=6,
            spotlight_intensity=1500.0,
            shadow_softness=0.1,
        ),
        "flickering": LightingProfile(
            dome_intensity=2000.0,
            color_temperature=4500,
            enable_flickering=True,
            flicker_frequency=2.0,
            flicker_amplitude=0.3,
        ),
    },
    typical_size_range=(15.0, 25.0),
    aisle_width_range=(2.5, 4.0),
    typical_human_positions=[(-5, -5), (5, 5), (0, 8)],
    ambient_noise_level=0.3,
    floor_texture="polished_concrete",
)

HOSPITAL_CONFIG = DomainConfig(
    name="hospital",
    description="Medical facility environment",
    primary_assets=["bed", "medical_cart", "wheelchair"],
    secondary_assets=["cabinet", "table", "chair"],
    clutter_assets=["medical_equipment", "small_cart", "supplies"],
    default_clutter_level=ClutterLevel.MODERATE,
    min_clutter_objects=5,
    max_clutter_objects=12,
    clutter_density_range=(0.08, 0.20),
    lighting_profiles={
        "standard": LightingProfile(
            dome_intensity=2200.0,
            color_temperature=5500,  # Clinical daylight
            num_spotlights=3,
            shadow_softness=0.7,
            ambient_occlusion=True,
        ),
        "dim": LightingProfile(
            dome_intensity=800.0,
            color_temperature=3000,  # Night mode
            num_spotlights=1,
            shadow_softness=0.8,
        ),
        "natural": LightingProfile(
            dome_intensity=2800.0,
            color_temperature=6000,  # Natural daylight
            time_of_day=14.0,
            shadow_softness=0.6,
        ),
        "emergency": LightingProfile(
            dome_intensity=1500.0,
            color_temperature=3500,
            enable_flickering=True,
            flicker_frequency=0.5,
            flicker_amplitude=0.2,
        ),
    },
    typical_size_range=(12.0, 18.0),
    aisle_width_range=(2.0, 3.0),
    typical_human_positions=[(-4, 0), (4, 0), (0, -6)],
    ambient_noise_level=0.15,
    floor_texture="linoleum",
    wall_texture="sterile_white",
)

OFFICE_CONFIG = DomainConfig(
    name="office",
    description="Corporate office environment",
    primary_assets=["desk", "chair", "filing_cabinet"],
    secondary_assets=["bookshelf", "plant", "table"],
    clutter_assets=["paper_stack", "coffee_mug", "office_supplies"],
    default_clutter_level=ClutterLevel.MODERATE,
    min_clutter_objects=8,
    max_clutter_objects=18,
    clutter_density_range=(0.12, 0.25),
    lighting_profiles={
        "standard": LightingProfile(
            dome_intensity=2000.0,
            color_temperature=4000,  # Office fluorescent
            num_spotlights=4,
            shadow_softness=0.5,
        ),
        "dim": LightingProfile(
            dome_intensity=1000.0,
            color_temperature=2700,  # Warm evening
            num_spotlights=2,
            shadow_softness=0.7,
        ),
        "natural": LightingProfile(
            dome_intensity=2600.0,
            color_temperature=6000,
            time_of_day=10.0,
            shadow_softness=0.6,
        ),
        "mixed": LightingProfile(
            dome_intensity=1800.0,
            color_temperature=4500,
            num_spotlights=6,
            spotlight_intensity=800.0,
            shadow_softness=0.5,
        ),
    },
    typical_size_range=(10.0, 16.0),
    aisle_width_range=(1.5, 2.5),
    typical_human_positions=[(-3, -3), (3, -3), (0, 4)],
    ambient_noise_level=0.1,
    floor_texture="carpet",
    wall_texture="painted_neutral",
)

RETAIL_CONFIG = DomainConfig(
    name="retail",
    description="Store/retail environment",
    primary_assets=["shelf", "display", "counter"],
    secondary_assets=["mannequin", "rack", "table"],
    clutter_assets=["product_box", "shopping_basket", "signage"],
    default_clutter_level=ClutterLevel.HIGH,
    min_clutter_objects=12,
    max_clutter_objects=30,
    clutter_density_range=(0.18, 0.40),
    lighting_profiles={
        "standard": LightingProfile(
            dome_intensity=2400.0,
            color_temperature=3500,  # Warm retail
            num_spotlights=8,
            spotlight_intensity=1200.0,
            shadow_softness=0.4,
        ),
        "bright": LightingProfile(
            dome_intensity=3200.0,
            color_temperature=5000,
            num_spotlights=10,
            shadow_softness=0.3,
        ),
        "accent": LightingProfile(
            dome_intensity=1800.0,
            color_temperature=3000,
            num_spotlights=12,
            spotlight_intensity=1500.0,
            shadow_softness=0.2,
        ),
    },
    typical_size_range=(12.0, 20.0),
    aisle_width_range=(1.8, 3.0),
    typical_human_positions=[(-4, 0), (4, 0), (0, -5), (0, 5)],
    ambient_noise_level=0.25,
    floor_texture="tile",
)

FACTORY_CONFIG = DomainConfig(
    name="factory",
    description="Manufacturing/production environment",
    primary_assets=["machine", "conveyor", "workbench"],
    secondary_assets=["toolbox", "cart", "pallet"],
    clutter_assets=["tool", "part", "container"],
    default_clutter_level=ClutterLevel.HIGH,
    min_clutter_objects=10,
    max_clutter_objects=20,
    clutter_density_range=(0.12, 0.28),
    lighting_profiles={
        "standard": LightingProfile(
            dome_intensity=2800.0,
            color_temperature=5000,  # Industrial
            num_spotlights=6,
            shadow_softness=0.2,
        ),
        "dim": LightingProfile(
            dome_intensity=1500.0,
            color_temperature=4000,
            num_spotlights=3,
            shadow_softness=0.4,
        ),
        "harsh": LightingProfile(
            dome_intensity=3500.0,
            color_temperature=5500,
            num_spotlights=8,
            spotlight_intensity=2000.0,
            shadow_softness=0.1,
        ),
    },
    typical_size_range=(15.0, 25.0),
    aisle_width_range=(2.0, 4.0),
    wall_height=4.0,
    typical_human_positions=[(-6, 0), (6, 0), (0, -8)],
    ambient_noise_level=0.5,
    floor_texture="industrial_concrete",
)

LABORATORY_CONFIG = DomainConfig(
    name="laboratory",
    description="Research laboratory environment",
    primary_assets=["lab_bench", "equipment", "fume_hood"],
    secondary_assets=["cabinet", "chair", "table"],
    clutter_assets=["beaker", "sample", "instrument"],
    default_clutter_level=ClutterLevel.MODERATE,
    min_clutter_objects=6,
    max_clutter_objects=15,
    clutter_density_range=(0.10, 0.22),
    lighting_profiles={
        "standard": LightingProfile(
            dome_intensity=2300.0,
            color_temperature=5500,  # Lab fluorescent
            num_spotlights=4,
            shadow_softness=0.6,
            ambient_occlusion=True,
        ),
        "bright": LightingProfile(
            dome_intensity=3000.0,
            color_temperature=6000,
            num_spotlights=6,
            shadow_softness=0.5,
        ),
        "focused": LightingProfile(
            dome_intensity=1800.0,
            color_temperature=5500,
            num_spotlights=8,
            spotlight_intensity=1500.0,
            shadow_softness=0.3,
        ),
    },
    typical_size_range=(10.0, 15.0),
    aisle_width_range=(1.8, 2.5),
    typical_human_positions=[(-3, 0), (3, 0), (0, -4)],
    ambient_noise_level=0.05,
    floor_texture="epoxy",
    wall_texture="sterile_white",
)


# Domain registry
DOMAIN_CONFIGS = {
    "warehouse": WAREHOUSE_CONFIG,
    "hospital": HOSPITAL_CONFIG,
    "office": OFFICE_CONFIG,
    "retail": RETAIL_CONFIG,
    "factory": FACTORY_CONFIG,
    "laboratory": LABORATORY_CONFIG,
}


def get_domain_config(domain_name: str) -> DomainConfig:
    """Get configuration for a specific domain.

    Args:
        domain_name: Name of domain (warehouse, hospital, etc.)

    Returns:
        DomainConfig for the specified domain

    Raises:
        ValueError: If domain name is unknown
    """
    if domain_name not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain_name}. "
                        f"Available: {list(DOMAIN_CONFIGS.keys())}")

    return DOMAIN_CONFIGS[domain_name]


def get_lighting_profile(domain_name: str, profile_name: str = "standard") -> LightingProfile:
    """Get lighting profile for a domain.

    Args:
        domain_name: Name of domain
        profile_name: Name of lighting profile

    Returns:
        LightingProfile for the specified domain and profile
    """
    domain = get_domain_config(domain_name)

    if profile_name not in domain.lighting_profiles:
        print(f"Warning: Lighting profile '{profile_name}' not found for {domain_name}, "
              f"using 'standard'")
        profile_name = "standard"

    return domain.lighting_profiles.get(profile_name, domain.lighting_profiles["standard"])


def sample_clutter_level(domain_name: str, clutter_level: Optional[ClutterLevel] = None) -> int:
    """Sample number of clutter objects for a domain.

    Args:
        domain_name: Name of domain
        clutter_level: Desired clutter level (None=use domain default)

    Returns:
        Number of objects to spawn
    """
    domain = get_domain_config(domain_name)

    if clutter_level is None:
        clutter_level = domain.default_clutter_level

    # Define ranges for each clutter level
    ranges = {
        ClutterLevel.MINIMAL: (0, 3),
        ClutterLevel.LOW: (4, 8),
        ClutterLevel.MODERATE: (9, 15),
        ClutterLevel.HIGH: (16, 25),
        ClutterLevel.EXTREME: (26, 40),
    }

    min_obj, max_obj = ranges[clutter_level]

    # Constrain to domain limits
    min_obj = max(min_obj, domain.min_clutter_objects)
    max_obj = min(max_obj, domain.max_clutter_objects)

    # Ensure min < max
    if min_obj > max_obj:
        min_obj = max_obj

    if min_obj == max_obj:
        return min_obj

    return np.random.randint(min_obj, max_obj + 1)
