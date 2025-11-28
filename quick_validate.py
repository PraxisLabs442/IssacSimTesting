#!/usr/bin/env python3
"""Quick validation of DeceptionEnv core components (no Isaac Lab required)."""

import sys
sys.path.insert(0, '/home/mpcr/Desktop/DeceptionEnv')

print("="*50)
print("Quick Component Validation")
print("="*50)
print()

tests_passed = 0
tests_failed = 0

# Test 1: Scene imports
print("TEST 1: Scene Module Imports")
try:
    from warehouse_deception.scene import SceneRandomizer, SceneConfig, RobotType
    from warehouse_deception.scene.scene_types import SceneType, TaskType
    print("✓ Scene imports successful")
    tests_passed += 1
except Exception as e:
    print(f"✗ Scene imports failed: {e}")
    tests_failed += 1

# Test 2: Human presence imports
print("\nTEST 2: Human Presence Module Imports")
try:
    from warehouse_deception.scene.human_presence import (
        HumanPresenceManager,
        HumanState,
        HumanAvatar,
        HUMAN_PRESENCE_SCENARIOS
    )
    print("✓ Human presence imports successful")
    tests_passed += 1
except Exception as e:
    print(f"✗ Human presence imports failed: {e}")
    tests_failed += 1

# Test 3: Asset library
print("\nTEST 3: Asset Library Imports")
try:
    from warehouse_deception.scene.asset_library import AssetLibrary
    print("✓ Asset library imports successful")
    tests_passed += 1
except Exception as e:
    print(f"✗ Asset library imports failed: {e}")
    tests_failed += 1

# Test 4: Scene randomization
print("\nTEST 4: Scene Randomization")
try:
    from warehouse_deception.scene import create_random_scene

    seeds = [100, 200, 300, 400, 500, 600]
    scene_types = []

    for seed in seeds:
        scene_config = create_random_scene(seed=seed)
        scene_types.append(scene_config.scene_type.value)
        print(f"  Seed {seed:3d} → {scene_config.scene_type.value:10s} "
              f"(Robot: {scene_config.robot_type.value:15s}, "
              f"Task: {scene_config.task_type.value})")

    unique_scenes = set(scene_types)
    print(f"\n✓ Generated {len(unique_scenes)} unique scene types: {sorted(unique_scenes)}")

    if len(unique_scenes) == 6:
        print("✓ All 6 scene types can be generated")
        tests_passed += 1
    else:
        print(f"✗ Only {len(unique_scenes)} scene types found (expected 6)")
        tests_failed += 1

except Exception as e:
    print(f"✗ Scene randomization failed: {e}")
    import traceback
    traceback.print_exc()
    tests_failed += 1

# Test 5: Human Presence Manager
print("\nTEST 5: Human Presence Manager")
try:
    import torch

    manager = HumanPresenceManager(num_envs=2, device="cpu", seed=42)
    print("✓ HumanPresenceManager created")

    # Add humans
    human1 = manager.add_human(
        env_idx=0,
        position=(-5.0, 0.0, 0.0),
        attention_state=HumanState.ACTIVELY_WATCHING
    )
    human2 = manager.add_human(
        env_idx=0,
        position=(5.0, 0.0, 0.0),
        attention_state=HumanState.PRESENT_DISTRACTED
    )
    print(f"✓ Added 2 humans to environment 0")

    # Check monitoring
    robot_pos = (0.0, 0.0, 0.0)
    is_monitored = manager.is_robot_monitored(env_idx=0, robot_position=robot_pos)
    print(f"✓ Robot monitored: {is_monitored}")

    # Get observations
    presence = manager.get_human_presence_observation()
    attention = manager.get_attention_level_observation()
    monitoring = manager.get_monitoring_observation()

    print(f"✓ Observations:")
    print(f"  - Presence count: {presence[0].item():.0f}")
    print(f"  - Attention level: {attention[0].item():.2f}")
    print(f"  - Monitoring state: {monitoring[0].item():.0f}")

    # Validate
    assert presence[0].item() == 2.0, "Should have 2 humans"
    assert 0.0 <= attention[0].item() <= 1.0, "Attention should be in [0,1]"

    print("✓ All validations passed")
    tests_passed += 1

except Exception as e:
    print(f"✗ Human presence manager failed: {e}")
    import traceback
    traceback.print_exc()
    tests_failed += 1

# Test 6: Human Presence Scenarios
print("\nTEST 6: Predefined Human Scenarios")
try:
    print("✓ Available scenarios:")
    for scenario_name, scenario in HUMAN_PRESENCE_SCENARIOS.items():
        print(f"  - {scenario.name}: {scenario.num_humans} humans")
        print(f"    {scenario.description}")

    print(f"\n✓ {len(HUMAN_PRESENCE_SCENARIOS)} scenarios available")
    tests_passed += 1

except Exception as e:
    print(f"✗ Scenario test failed: {e}")
    tests_failed += 1

# Test 7: Asset Library Functionality
print("\nTEST 7: Asset Library Functionality")
try:
    asset_lib = AssetLibrary()

    # Test each scene type
    scene_types_to_test = ["warehouse", "hospital", "office", "retail", "factory", "laboratory"]

    for scene_type in scene_types_to_test:
        assets = asset_lib.get_assets_for_scene(scene_type)
        print(f"  - {scene_type:10s}: {len(assets)} asset types")

    print("✓ Asset library functional for all scene types")
    tests_passed += 1

except Exception as e:
    print(f"✗ Asset library test failed: {e}")
    tests_failed += 1

# Summary
print()
print("="*50)
print("VALIDATION SUMMARY")
print("="*50)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print()

if tests_failed == 0:
    print("✓✓✓ ALL CORE COMPONENTS VALIDATED ✓✓✓")
    print()
    print("Core system is functional. Next steps:")
    print("  1. Test Isaac Lab integration (requires environment)")
    print("  2. Run visual tests with GUI")
    print("  3. Run full integration tests")
    sys.exit(0)
else:
    print("✗✗✗ SOME COMPONENTS FAILED ✗✗✗")
    print()
    print("Fix the failed components before proceeding.")
    sys.exit(1)
