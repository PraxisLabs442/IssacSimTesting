#!/bin/bash
#
# Run deception environment with randomized scenes
#
# This script launches the environment with random scene type, robot, task, and objects
#

set -e

echo "========================================================================"
echo "Randomized Deception Detection Environment"
echo "========================================================================"
echo ""

# Configuration
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-$HOME/Downloads/IsaacLab}"
DECEPTION_ENV_DIR="${DECEPTION_ENV_DIR:-$HOME/Desktop/DeceptionEnv}"
NUM_ENVS="${NUM_ENVS:-4}"
STEPS="${STEPS:-500}"
SEED="${SEED:-}"  # Empty = random seed
HEADLESS="${HEADLESS:-False}"

echo "Configuration:"
echo "  Isaac Lab:     $ISAAC_LAB_DIR"
echo "  DeceptionEnv:  $DECEPTION_ENV_DIR"
echo "  Num Envs:      $NUM_ENVS"
echo "  Steps:         $STEPS"
echo "  Seed:          ${SEED:-random}"
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

# Build command
CMD="./isaaclab.sh -p $DECEPTION_ENV_DIR/run_deception_env.py"
CMD="$CMD --randomize"
CMD="$CMD --num-envs $NUM_ENVS"
CMD="$CMD --steps $STEPS"
CMD="$CMD --headless $HEADLESS"

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# Launch with Isaac Lab
echo "Launching randomized environment..."
echo "Command: $CMD"
echo ""

cd "$ISAAC_LAB_DIR"
eval $CMD

echo ""
echo "========================================================================"
echo "Simulation complete!"
echo "========================================================================"
