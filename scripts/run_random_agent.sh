#!/bin/bash
#
# Run warehouse deception environment with random agent
#
# This script launches the environment with random actions for testing
# Useful for debugging environment logic without VLA overhead

set -e

echo "========================================================================"
echo "Warehouse Deception Environment - Random Agent Testing"
echo "========================================================================"
echo ""

# Configuration
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-$HOME/Downloads/IsaacLab}"
DECEPTION_ENV_DIR="${DECEPTION_ENV_DIR:-$HOME/Desktop/DeceptionEnv}"
NUM_ENVS="${NUM_ENVS:-4}"
STEPS="${STEPS:-500}"
ROBOT="${ROBOT:-franka}"
HEADLESS="${HEADLESS:-False}"
MONITORING_FREQ="${MONITORING_FREQ:-100}"
RANDOM_TOGGLE="${RANDOM_TOGGLE:-false}"

echo "Configuration:"
echo "  Isaac Lab:        $ISAAC_LAB_DIR"
echo "  DeceptionEnv:     $DECEPTION_ENV_DIR"
echo "  Num Envs:         $NUM_ENVS"
echo "  Steps:            $STEPS"
echo "  Robot:            $ROBOT"
echo "  Headless:         $HEADLESS"
echo "  Monitoring Freq:  $MONITORING_FREQ steps"
echo "  Random Toggle:    $RANDOM_TOGGLE"
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

# Build random toggle flag
TOGGLE_FLAG=""
if [ "$RANDOM_TOGGLE" = "true" ]; then
    TOGGLE_FLAG="--random-toggle"
fi

# Launch with Isaac Lab
echo "Launching environment with random agent..."
echo ""

cd "$ISAAC_LAB_DIR"
./isaaclab.sh -p "$DECEPTION_ENV_DIR/run_deception_env.py" \
    --num-envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --robot "$ROBOT" \
    --headless "$HEADLESS" \
    --monitoring-freq "$MONITORING_FREQ" \
    $TOGGLE_FLAG

echo ""
echo "========================================================================"
echo "Simulation complete!"
echo "========================================================================"

