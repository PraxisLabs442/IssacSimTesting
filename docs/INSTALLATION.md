# Installation Guide: Strategic Deception Study System

Complete installation instructions for running VLA models in Isaac Sim with strategic deception experiments.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04 or 22.04
- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended: RTX 3090, 4090, or A100)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space
- **Python**: 3.10 or 3.11

### Software Dependencies
1. **Isaac Sim 2023.1.1** (includes Isaac Lab)
2. **CUDA 11.8+** and compatible drivers
3. **Python 3.10+** with pip

## Step 1: Install Isaac Sim and Isaac Lab (30 minutes)

### 1.1 Download Isaac Sim

```bash
# Download from NVIDIA (requires free account)
# https://developer.nvidia.com/isaac-sim

# Or use Isaac Sim Launcher
wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage
```

### 1.2 Install Isaac Lab

```bash
# Clone Isaac Lab repository
cd ~/Downloads
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Run installation script
./isaaclab.sh --install

# Verify installation
./isaaclab.sh --help
```

### 1.3 Test Isaac Lab

```bash
# Test basic environment
cd ~/Downloads/IsaacLab
./isaaclab.sh -p source/standalone/workflows/robomimic/collect_demonstrations.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 1 \
  --num_demos 1

# If this runs successfully, Isaac Lab is ready!
```

## Step 2: Install PraxisLabs System (10 minutes)

### 2.1 Clone Repository

```bash
cd ~/Desktop
git clone https://github.com/yourusername/PraxisLabs.git
cd PraxisLabs
```

### 2.2 Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate

# Install PraxisLabs package
pip install -e .

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.1 accelerate
pip install h5py pyyaml scipy opencv-python pillow
```

### 2.3 Verify Dependencies

```bash
# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: True

# Check transformers
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
# Expected: Transformers version: 4.40.1

# Check Isaac Lab imports
python -c "from omni.isaac.lab.envs import ManagerBasedRLEnv; print('Isaac Lab OK')"
# Expected: Isaac Lab OK
```

## Step 3: Download VLA Models (15 minutes)

### 3.1 OpenVLA-7B (Primary Model)

```bash
# Models will auto-download on first use, but you can pre-download:
python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
print('OpenVLA downloaded successfully')
"
```

**Expected download**: ~14GB (7B parameter model in bfloat16)

### 3.2 Optional: Additional Models

**RT-2-X** (requires setup):
```bash
# Follow instructions at: https://github.com/google-research/robotics_transformer
pip install rt2x
```

**Octo** (optional):
```bash
pip install octo-models
```

## Step 4: Configure System (5 minutes)

### 4.1 Set Environment Variables

Add to `~/.bashrc`:

```bash
# Isaac Sim paths
export ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-2023.1.1"
export ISAAC_LAB_PATH="${HOME}/Downloads/IsaacLab"

# CUDA (if not already set)
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# PraxisLabs
export PRAXIS_LABS_ROOT="${HOME}/Desktop/PraxisLabs"
```

Reload:
```bash
source ~/.bashrc
```

### 4.2 Configure GPU Assignment

Edit `~/.bashrc` or set before running:

```bash
# For single GPU setup
export ISAAC_GPU=0  # Isaac Sim
export VLA_GPU=0    # VLA model (same GPU)

# For dual GPU setup (recommended)
export ISAAC_GPU=0  # Isaac Sim
export VLA_GPU=1    # VLA model (separate GPU)
```

## Step 5: Verify Installation (5 minutes)

### 5.1 Test VLA Model Loading

```bash
cd ~/Desktop/PraxisLabs

# Test OpenVLA loading
python scripts/test_vla_model.py --model openvla-7b --device cuda:1
```

**Expected output**:
```
✓ Model loaded successfully
✓ Inference successful
  Action shape: (7,)
  Action range: [-0.523, 0.412]
  Metadata keys: ['activations', 'tokens']
✓ All tests passed!
```

### 5.2 Quick System Test

```bash
# Run single episode end-to-end test
python scripts/run_deception_study.py \
  --model openvla-7b \
  --device cuda:1 \
  --tasks pick_place \
  --episodes-per-phase 1 \
  --log-dir logs/test

# Expected: Completes 5 episodes (1 per phase) in ~2-3 minutes
```

**Expected output**:
```
=============================================================
STRATEGIC DECEPTION STUDY - VLA MODELS IN ISAAC SIM
=============================================================
Model: openvla-7b
Device: cuda:1
Tasks: ['pick_place']
Episodes per phase: 1

1. Loading VLA model...
✓ Model loaded: OpenVLA-7B

2. Setting up data collector...
✓ Data collector ready

3. Initializing study runner...
✓ Study runner initialized

4. Running experiment...
Running phase baseline (1/5)...
Running phase monitoring_safe (2/5)...
Running phase monitoring_conflict (3/5)...
Running phase no_monitoring_safe (4/5)...
Running phase no_monitoring_conflict (5/5)...

5. Saving results...
✓ Results saved

EXPERIMENT COMPLETE
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Option 1: Use separate GPUs
python scripts/run_deception_study.py --device cuda:1  # VLA on GPU 1

# Option 2: Reduce batch size (in environment config)
# Edit src/environment/isaac_lab_env.py:
# EnvironmentConfig.num_envs = 1

# Option 3: Use smaller model (future)
# python scripts/run_deception_study.py --model openvla-3b
```

### Issue 2: Isaac Lab Import Errors

**Symptoms**:
```
ImportError: cannot import name 'ManagerBasedRLEnv'
```

**Solutions**:
```bash
# Verify Isaac Lab installation
cd ~/Downloads/IsaacLab
./isaaclab.sh --help

# Reinstall if needed
./isaaclab.sh --install

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
# Should include Isaac Lab paths
```

### Issue 3: OpenVLA Download Fails

**Symptoms**:
```
ConnectionError: Hugging Face Hub unreachable
```

**Solutions**:
```bash
# Set Hugging Face cache directory
export HF_HOME="${HOME}/.cache/huggingface"

# Use mirror (if in China)
export HF_ENDPOINT="https://hf-mirror.com"

# Manual download
git lfs install
git clone https://huggingface.co/openvla/openvla-7b
```

### Issue 4: Display/Rendering Issues

**Symptoms**:
```
Error: Cannot initialize display
```

**Solutions**:
```bash
# For headless servers, use EGL rendering
export DISPLAY=:0
xhost +local:docker

# Or disable rendering
python scripts/run_deception_study.py --no-render
```

### Issue 5: Contact Sensor Errors

**Symptoms**:
```
Warning: Contact sensor not found
```

**Solution**: This is non-critical. The system will work without contact sensors but collision detection will be less accurate. Check Isaac Lab version compatibility.

## Performance Optimization

### GPU Memory Optimization

```python
# Edit src/environment/isaac_lab_env.py
EnvironmentConfig(
    num_envs=1,  # Reduce parallel environments
    render=False,  # Disable rendering if not needed
)

# Use bfloat16 for VLA (already default)
torch_dtype=torch.bfloat16
```

### Storage Optimization

```bash
# Disable image logging for faster runs
python scripts/run_deception_study.py \
  --no-log-images \
  --no-log-activations

# Reduces storage from ~50MB to ~500KB per episode
```

### Multi-GPU Setup

For systems with 3+ GPUs:

```bash
# Terminal 1: Isaac Sim on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/run_deception_study.py --device cuda:1

# System will automatically use:
# - GPU 0: Isaac Sim (rendering + physics)
# - GPU 1: VLA model (inference)
# - GPUs 2-3: Available for parallel experiments
```

## Next Steps

After installation:

1. **Run Quick Test** (5 min):
   ```bash
   python scripts/run_deception_study.py --episodes-per-phase 1
   ```

2. **Run Pilot Study** (20 min):
   ```bash
   python scripts/run_deception_study.py --episodes-per-phase 10
   ```

3. **Analyze Results**:
   ```bash
   python scripts/analyze_results.py logs/pilot_study
   ```

4. **Scale to Main Study** (2-3 hours):
   ```bash
   python scripts/run_deception_study.py --episodes-per-phase 50 --experiment-name main_study
   ```

## Directory Structure After Installation

```
~/Desktop/PraxisLabs/
├── src/                      # Source code
│   ├── vla/                  # VLA model wrappers
│   ├── environment/          # Isaac Lab integration
│   ├── logging/              # Data collection
│   └── experiments/          # Protocol orchestration
├── config/                   # Phase configurations
├── scripts/                  # Main experiment scripts
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── logs/                     # Experiment results (created on first run)
└── README.md

~/.cache/huggingface/        # Downloaded models (~14GB)
~/Downloads/IsaacLab/        # Isaac Lab installation
```

## Storage Requirements

- **Isaac Sim + Isaac Lab**: ~15GB
- **OpenVLA-7B**: ~14GB
- **Python packages**: ~5GB
- **Experiment data**:
  - With images: ~50MB per episode
  - Without images: ~500KB per episode
  - Pilot (50 episodes): ~2.5GB
  - Main study (250 episodes): ~12GB

**Total**: ~50GB for complete installation

## System Validation Checklist

Before running experiments:

- [ ] Isaac Sim installed and tested
- [ ] Isaac Lab environment runs
- [ ] CUDA available in Python
- [ ] OpenVLA model downloads successfully
- [ ] Test script passes (`test_vla_model.py`)
- [ ] Single episode test completes
- [ ] Results saved to logs directory
- [ ] Analysis script can read results

## Support

- **Documentation**: See `docs/` directory
- **Quick Start**: `docs/QUICKSTART.md`
- **System Overview**: `docs/README_SYSTEM.md`
- **Issues**: Open GitHub issue
- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab/

---

**Installation complete! Ready to discover if VLA models can strategically deceive.**
