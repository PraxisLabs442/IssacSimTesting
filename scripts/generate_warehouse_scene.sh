#!/bin/bash
#
# Generate default warehouse USD scene
#
# This script runs the warehouse builder to create a procedural USD scene
# with monitoring cameras, shelves, and path options

set -e

echo "========================================================================"
echo "Warehouse Scene Generator"
echo "========================================================================"
echo ""

# Configuration
DECEPTION_ENV_DIR="${DECEPTION_ENV_DIR:-$HOME/Desktop/DeceptionEnv}"
OUTPUT_DIR="${OUTPUT_DIR:-$DECEPTION_ENV_DIR/assets}"

echo "Configuration:"
echo "  DeceptionEnv:  $DECEPTION_ENV_DIR"
echo "  Output Dir:    $OUTPUT_DIR"
echo ""

# Check DeceptionEnv exists
if [ ! -d "$DECEPTION_ENV_DIR" ]; then
    echo "ERROR: DeceptionEnv not found at $DECEPTION_ENV_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run warehouse builder
echo "Generating warehouse scene..."
echo ""

cd "$DECEPTION_ENV_DIR"
python -m warehouse_deception.scene.warehouse_builder

echo ""
echo "========================================================================"
echo "Scene generation complete!"
echo "Output: $OUTPUT_DIR/warehouse_default.usd"
echo "========================================================================"

