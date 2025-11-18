#!/bin/bash
# Launch Isaac Lab with GUI for Experiment Design

echo "Launching Isaac Lab GUI for Experiment Design"
echo "=============================================="
echo ""

cd /home/mpcr/Downloads/IsaacLab

echo "Available Demos for Deception Experiments:"
echo ""
echo "1. Pick and Place - Object manipulation with precision"
echo "   ./isaaclab.sh -p scripts/demos/pick_and_place.py"
echo ""
echo "2. Robotic Arms - General manipulation tasks"
echo "   ./isaaclab.sh -p scripts/demos/arms.py"
echo ""
echo "3. Quadrupeds - Locomotion and navigation"
echo "   ./isaaclab.sh -p scripts/demos/quadrupeds.py"
echo ""
echo "4. Procedural Terrain - Navigation with obstacles"
echo "   ./isaaclab.sh -p scripts/demos/procedural_terrain.py"
echo ""

# Check which to launch
if [ "$1" == "pick" ]; then
    echo "Launching: Pick and Place Demo"
    export DISPLAY=:1
    ./isaaclab.sh -p scripts/demos/pick_and_place.py
elif [ "$1" == "arms" ]; then
    echo "Launching: Robotic Arms Demo"
    export DISPLAY=:1
    ./isaaclab.sh -p scripts/demos/arms.py
elif [ "$1" == "quad" ]; then
    echo "Launching: Quadrupeds Demo"
    export DISPLAY=:1
    ./isaaclab.sh -p scripts/demos/quadrupeds.py
elif [ "$1" == "terrain" ]; then
    echo "Launching: Procedural Terrain Demo"
    export DISPLAY=:1
    ./isaaclab.sh -p scripts/demos/procedural_terrain.py
else
    echo "Usage: $0 [pick|arms|quad|terrain]"
    echo ""
    echo "Example:"
    echo "  $0 pick    # Launch pick-and-place demo"
    echo ""
    echo "Or launch directly:"
    echo "  cd /home/mpcr/Downloads/IsaacLab"
    echo "  export DISPLAY=:1"
    echo "  ./isaaclab.sh -p scripts/demos/pick_and_place.py"
    echo ""
    echo "Window should open on your display."
    echo "GPU usage visible in: nvidia-smi"
fi

