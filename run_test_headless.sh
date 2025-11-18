#!/bin/bash
# Run minimal VLA + Isaac Lab test in headless mode (no GUI)
# This wraps the Isaac Lab launcher for convenience

set -e

echo "================================================================================"
echo "VLA + Isaac Lab Test - Headless Mode"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Launch Isaac Sim in headless mode (no window)"
echo "  2. Create Franka Panda pick-and-place environment"
echo "  3. Load OpenVLA-7B model (or use dummy if unavailable)"
echo "  4. Run 100 steps with VLA controlling the robot"
echo ""
echo "Headless mode is faster and uses less GPU memory."
echo ""
echo "Press Ctrl+C to stop at any time"
echo ""

# Navigate to Isaac Lab directory
cd /home/mpcr/Downloads/IsaacLab

# Run test with headless flag
./isaaclab.sh -p /home/mpcr/Desktop/PraxisLabs/scripts/test_vla_isaac_minimal.py \
    --headless \
    --vla-device cuda:1 \
    --num-steps 100

echo ""
echo "================================================================================"
echo "Test complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  • Run with GUI: ./run_test_gui.sh"
echo "  • Test with more steps: edit --num-steps in this script"
echo "  • Run full study: cd ~/Downloads/IsaacLab && ./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/run_deception_study.py"
echo ""

