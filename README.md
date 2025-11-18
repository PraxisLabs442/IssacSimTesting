# VLA + Isaac Sim

Run OpenVLA model in Isaac Sim. That's it.

## Setup

1. Install Isaac Sim ([download](https://developer.nvidia.com/isaac-sim))
2. Install Isaac Lab ([guide](https://isaac-sim.github.io/IsaacLab/))
3. Install dependencies:
```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p -m pip install timm accelerate transformers
```

## Run

### With GUI (see robot moving):
```bash
cd ~/Desktop/PraxisLabs
./run_test_gui.sh
```

### Headless (faster, no window):
```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/run_vla.py --headless
```

### Custom options:
```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/run_vla.py \
  --headless False \
  --vla-device cuda:0 \
  --steps 500
```

## What it does

1. Launches Isaac Sim
2. Creates Franka Panda robot + cube environment
3. Loads OpenVLA-7B model
4. VLA controls the robot to pick and place the cube

## Files

- `run_vla.py` - Main script
- `src/vla/openvla.py` - OpenVLA wrapper
- `run_test_gui.sh` - Launch with GUI

That's all you need.
