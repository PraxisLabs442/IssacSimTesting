# Setup Instructions

## Prerequisites

### 1. NVIDIA Isaac Sim 5.1
Download and install from: https://developer.nvidia.com/isaac-sim

```bash
# Set environment variable
export ISAAC_SIM_PATH="/path/to/isaac-sim-5.1"
```

### 2. Isaac Lab
Install Isaac Lab on top of Isaac Sim:

```bash
# Clone Isaac Lab
git clone https://github.com/NVIDIA-Omniverse/IsaacLab.git
cd IsaacLab

# Install (follow their README for detailed instructions)
./isaaclab.sh --install
```

### 3. Python Environment
Create a Python 3.10+ environment:

```bash
# Create conda environment
conda create -n isaac_vla python=3.10
conda activate isaac_vla

# Install Isaac Lab dependencies
cd IsaacLab
pip install -e .
```

### 4. VLA Model Dependencies
Install requirements for VLA models (e.g., OpenVLA):

```bash
pip install torch torchvision
pip install transformers
pip install accelerate
pip install pillow numpy
```

### 5. Additional Dependencies

```bash
# For logging and data handling
pip install h5py pyyaml

# For optional features
pip install wandb  # Experiment tracking
pip install tensorboard  # Visualization
```

## Project Setup

### 1. Install this project

```bash
cd /path/to/PraxisLabs
pip install -e .
```

### 2. Configure paths

Edit `config/default_config.yaml` to match your setup:
- Isaac Sim path
- Model paths
- GPU devices
- Log directories

### 3. Verify installation

Run the simple example:

```bash
python examples/simple_example.py
```

## Multi-GPU Setup

For scaling experiments across multiple GPUs:

### 1. Isaac Sim on GPU 0
Isaac Sim should run on GPU 0 (default).

### 2. VLA models on GPU 1-3
Edit config:

```yaml
multi_gpu:
  enabled: true
  isaac_sim_gpu: 0
  vla_gpus: [1, 2, 3]
  parallel_envs: 4
```

### 3. Run with GPU assignment

```bash
# VLA on GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/main.py --device cuda:0

# Or let the experiment runner handle it
python scripts/main.py --multi-gpu
```

## Troubleshooting

### Isaac Sim not found
Ensure `ISAAC_SIM_PATH` is set and Isaac Lab is properly installed.

### CUDA out of memory
- Reduce `num_envs` in config
- Use smaller VLA model
- Disable activation logging
- Use HDF5 format without image logging

### Model loading errors
- Check Hugging Face token is configured
- Verify model name in config
- Ensure sufficient disk space for model download

## Next Steps

1. **Test basic integration**: Run `examples/simple_example.py`
2. **Run pilot experiment**: Run `examples/multi_phase_example.py`
3. **Full experiment**: Use `scripts/main.py` with full config
4. **Analysis**: Use notebooks in `notebooks/` to analyze results

## Documentation

- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- OpenVLA: https://openvla.github.io/
- Project docs: See `docs/` directory
