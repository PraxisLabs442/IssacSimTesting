#!/bin/bash
#
# Run warehouse deception environment with VLA integration
#
# This script launches the environment with OpenVLA model for control
# Make sure PraxisLabs VLA is set up before running

set -e

echo "========================================================================"
echo "Warehouse Deception Environment - VLA Integration"
echo "========================================================================"
echo ""

# Configuration
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-$HOME/Downloads/IsaacLab}"
DECEPTION_ENV_DIR="${DECEPTION_ENV_DIR:-$HOME/Desktop/DeceptionEnv}"
VLA_DEVICE="${VLA_DEVICE:-cuda:1}"
NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-500}"
ROBOT="${ROBOT:-franka}"
HEADLESS="${HEADLESS:-False}"

echo "Configuration:"
echo "  Isaac Lab:     $ISAAC_LAB_DIR"
echo "  DeceptionEnv:  $DECEPTION_ENV_DIR"
echo "  VLA Device:    $VLA_DEVICE"
echo "  Num Envs:      $NUM_ENVS"
echo "  Steps:         $STEPS"
echo "  Robot:         $ROBOT"
echo "  Headless:      $HEADLESS"
echo ""

# Check Isaac Lab exists
if [ ! -d "$ISAAC_LAB_DIR" ]; then
    echo "ERROR: Isaac Lab not found at $ISAAC_LAB_DIR"
    echo "Set ISAAC_LAB_DIR environment variable or install Isaac Lab"
    exit 1
fi

# Check DeceptionEnv exists
if [ ! -d "$DECEPTION_ENV_DIR" ]; then
    echo "ERROR: DeceptionEnv not found at $DECEPTION_ENV_DIR"
    echo "Set DECEPTION_ENV_DIR environment variable"
    exit 1
fi

# Check PraxisLabs VLA exists
PRAXISLAB_DIR="$HOME/Desktop/PraxisLabs"
if [ ! -d "$PRAXISLAB_DIR" ]; then
    echo "WARNING: PraxisLabs not found at $PRAXISLAB_DIR"
    echo "VLA integration will not work. Install PraxisLabs first."
    echo ""
fi

# Launch with Isaac Lab
echo "Launching environment with VLA..."
echo ""

cd "$ISAAC_LAB_DIR"
./isaaclab.sh -p "$DECEPTION_ENV_DIR/run_deception_env.py" \
    --use-vla \
    --vla-device "$VLA_DEVICE" \
    --num-envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --robot "$ROBOT" \
    --headless "$HEADLESS"

echo ""
echo "========================================================================"
echo "Simulation complete!"
echo "========================================================================"

