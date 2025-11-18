# Quick Command Reference

## Launch Isaac Lab GUI

### Method 1: Direct Command (Simplest)
```bash
cd /home/mpcr/Downloads/IsaacLab && export DISPLAY=:1 && ./isaaclab.sh -p scripts/demos/pick_and_place.py
```

### Method 2: Other Demos
```bash
# Change directory first
cd /home/mpcr/Downloads/IsaacLab
export DISPLAY=:1

# Then pick a demo:
./isaaclab.sh -p scripts/demos/arms.py                    # Robotic arms
./isaaclab.sh -p scripts/demos/quadrupeds.py              # Walking robots
./isaaclab.sh -p scripts/demos/procedural_terrain.py      # Terrain navigation
./isaaclab.sh -p scripts/demos/multi_asset.py             # Multiple robots
```

### Method 3: Using Helper Script
```bash
cd /home/mpcr/Desktop/PraxisLabs
bash LAUNCH_GUI.sh pick    # Pick and place
bash LAUNCH_GUI.sh arms    # Robotic arms
bash LAUNCH_GUI.sh quad    # Quadrupeds
```

## Check if Running
```bash
# See GPU usage
nvidia-smi

# Check processes
ps aux | grep isaac
```

## Stop Running Demos
```bash
pkill -f isaac
```

## Common Issues

### "No such file or directory"
**Problem**: Wrong directory or path
**Fix**: Always `cd` to correct directory first
```bash
cd /home/mpcr/Downloads/IsaacLab
# Then run command
```

### "Command not found"
**Problem**: Need to use full path or be in correct directory
**Fix**: Use absolute paths
```bash
/home/mpcr/Downloads/IsaacLab/isaaclab.sh -p scripts/demos/arms.py
```

### Window doesn't open
**Problem**: Display variable not set
**Fix**: Export DISPLAY before running
```bash
export DISPLAY=:1
./isaaclab.sh -p scripts/demos/pick_and_place.py
```

## Test Data Collection

```bash
cd /home/mpcr/Desktop/PraxisLabs
conda activate vlaenv

# Simple h5py test
python -c "import h5py; print('✓ h5py works')"

# Test numpy
python -c "import numpy; print('✓ numpy works')"
```

## Ready to Design Experiments

Once you see the GUI:
1. Observe the robot workspace
2. Note object types and placement
3. Identify where to add monitoring cameras
4. Sketch deception scenarios
5. Edit `EXPERIMENT_DESIGN.md` with your notes

## Get Help

- **Full design guide**: `EXPERIMENT_DESIGN.md`
- **Visualization info**: `VISUALIZE.md`
- **System status**: `STATUS.md`
- **Main readme**: `README.md`

