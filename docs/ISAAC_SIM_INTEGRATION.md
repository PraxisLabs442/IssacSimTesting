# Isaac Sim Integration Guide for PraxisLabs

## Current System Status

✅ **What's Working:**
- OpenVLA-7B model loads and runs successfully (with CPU offloading)
- PraxisLabs deception protocol runs in **dummy mode**
- All 5 experimental phases execute correctly
- Safety metrics, HDF5 logging, and data collection work
- System has been tested end-to-end without real Isaac Sim

⚠️ **What Needs Setup:**
- Isaac Sim visual rendering (currently using dummy observations)
- Real Isaac Lab environment integration
- Camera observations from actual simulation

## Understanding the Architecture

### Current System Flow (Dummy Mode)

```
┌─────────────────────────────────────────────────────────────┐
│                  PraxisLabs Deception Study                  │
│                                                               │
│  ┌──────────────┐                                            │
│  │ Phase Config │                                            │
│  │  (5 phases)  │                                            │
│  └──────┬───────┘                                            │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────┐         ┌────────────────┐             │
│  │ Isaac Lab Env   │────────>│ OpenVLA-7B     │             │
│  │ (DUMMY MODE)    │  dummy  │ (7.54B params) │             │
│  │ - Fake RGB      │  images │ On GPU 1       │             │
│  │ - Random state  │         │ + CPU offload  │             │
│  └─────────────────┘         └────────┬───────┘             │
│                                       │ actions              │
│                                       v                      │
│                              ┌──────────────┐                │
│                              │ HDF5 Logger  │                │
│                              │ - Activations│                │
│                              │ - Metrics    │                │
│                              │ - Safety     │                │
│                              └──────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Target System Flow (With Isaac Sim)

```
┌──────────────────────────────────────────────────────────────────┐
│                   PraxisLabs with Isaac Sim                       │
│                                                                    │
│  ┌──────────────┐                                                 │
│  │ Phase Config │                                                 │
│  │  (5 phases)  │                                                 │
│  └──────┬───────┘                                                 │
│         │                                                          │
│         v                                                          │
│  ┌────────────────────┐         ┌────────────────┐               │
│  │ Isaac Sim (GPU 0)  │  real   │ OpenVLA-7B     │               │
│  │ ┌───────────────┐  │  images │ (7.54B params) │               │
│  │ │ Franka Robot  │  │────────>│ On GPU 1       │               │
│  │ │ Environment   │  │  +state │ + CPU offload  │               │
│  │ │ Camera Sensor │  │         └────────┬───────┘               │
│  │ └───────────────┘  │                  │ actions                │
│  │ Isaac Lab Wrapper  │<─────────────────┘                        │
│  └────────────────────┘                                           │
│           │                                                        │
│           v                                                        │
│  ┌──────────────────────┐                                         │
│  │ Monitoring Cues      │                                         │
│  │ - Visual signs       │                                         │
│  │ - Phase-specific     │                                         │
│  └──────────────────────┘                                         │
│           │                                                        │
│           v                                                        │
│  ┌────────────────┐                                               │
│  │ HDF5 Logger    │                                               │
│  │ - Real RGB     │                                               │
│  │ - Activations  │                                               │
│  │ - Safety       │                                               │
│  └────────────────┘                                               │
└──────────────────────────────────────────────────────────────────┘
```

## Installation Issue Analysis

**Problem:** Isaac Lab needs Isaac Sim's Python environment, not conda

**Why it failed:**
1. Isaac Lab installation script tried to use conda Python 3.13
2. Isaac Sim requires Python 3.11 (bundled with Isaac Sim)
3. Package `dex-retargeting` doesn't support Python 3.13
4. Isaac Lab's `omni` modules only work with Isaac Sim's Python

## Integration Options

### Option 1: Use Isaac Lab as-is (Recommended)

**Prerequisites:**
- Isaac Sim already downloaded: `/home/mpcr/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64` ✅
- Isaac Lab already cloned: `/home/mpcr/Downloads/IsaacLab` ✅
- Symlink created: `_isaac_sim` → Isaac Sim ✅

**Steps:**

1. **Install Isaac Lab properly** (use Isaac Sim's Python):
```bash
cd ~/Downloads/IsaacLab

# Deactivate conda completely
conda deactivate

# Run Isaac Lab installation
./isaaclab.sh --install
```

This will:
- Use Isaac Sim's bundled Python 3.11
- Install Isaac Lab extensions
- Set up the environment correctly

2. **Test Isaac Lab works:**
```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

You should see an Isaac Sim window open!

3. **Run PraxisLabs with Isaac Lab:**
```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_isaac_sim_visual.py
```

### Option 2: Keep Using Dummy Mode (Current State)

**When to use:** If you just want to test OpenVLA model without simulation

**Advantages:**
- Already working ✅
- No Isaac Sim installation needed
- Faster iteration
- Good for testing VLA model behavior

**Disadvantages:**
- No visual rendering
- Fake observations (not realistic)
- Can't test visual monitoring cues
- No collision detection

**To use:**
```bash
cd ~/Desktop/PraxisLabs
source ~/miniconda/etc/profile.d/conda.sh
conda activate isaaclab
python scripts/run_deception_study.py --episodes-per-phase 1 --device cuda:1
```

## Recommended Path Forward

Since you explicitly requested to "see it and open it", I recommend **Option 1**:

### Step 1: Proper Isaac Lab Installation

```bash
# Navigate to Isaac Lab
cd ~/Downloads/IsaacLab

# Ensure conda is fully deactivated
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE

# Run installation with Isaac Sim's Python
./isaaclab.sh --install
```

**Expected time:** 5-10 minutes

**What it installs:**
- Isaac Lab core (`omni.isaac.lab`)
- Isaac Lab assets (`isaaclab_assets`)
- Isaac Lab tasks (`isaaclab_tasks`)
- Isaac Lab RL (`isaaclab_rl`)
- Dependencies compatible with Isaac Sim

### Step 2: Verify Installation

```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

**Expected result:** Isaac Sim window opens showing an empty scene

### Step 3: Run PraxisLabs Visual Test

```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_isaac_sim_visual.py
```

**Expected result:** Isaac Sim window shows Franka robot performing random actions

### Step 4: Full System Test with Isaac Sim

```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/run_deception_study.py \
  --episodes-per-phase 1 \
  --device cuda:1 \
  --tasks pick_place
```

**Expected result:**
- Isaac Sim window visible
- Franka robot executing actions from OpenVLA
- Real RGB images fed to VLA model
- Complete deception protocol with visual cues

## Troubleshooting

### "No module named 'omni'"
**Cause:** Using conda Python instead of Isaac Sim's Python
**Fix:** Deactivate conda completely, use `./isaaclab.sh -p` launcher

### "Could not find dex-retargeting"
**Cause:** Installing with Python 3.13 (not supported)
**Fix:** Use Isaac Sim's Python 3.11 via launcher script

### Isaac Sim window doesn't open
**Check:**
```bash
nvidia-smi  # Verify GPU is available
echo $DISPLAY  # Should show :0 or similar
xhost +  # Allow X11 connections
```

### "CUDA out of memory" during simulation
**Solution:** Isaac Sim uses GPU 0, VLA uses GPU 1
```bash
export ISAAC_GPU=0
./isaaclab.sh -p script.py --device cuda:1
```

## System Architecture Details

### File Structure
```
PraxisLabs/
├── src/
│   ├── environment/
│   │   ├── isaac_lab_env.py          # Wrapper for Isaac Lab
│   │   ├── sensors.py                # Sensor noise simulation
│   │   └── monitoring_cues.py        # Visual cues for phases
│   ├── vla/
│   │   └── models/
│   │       └── openvla_wrapper.py    # OpenVLA model
│   └── logging/
│       └── hdf5_writer.py            # Data logging
├── scripts/
│   ├── test_isaac_sim_visual.py      # Visual test script
│   └── run_deception_study.py        # Main experiment
└── config/
    └── phases.yaml                    # 5-phase protocol config
```

### Key Components

#### 1. Isaac Lab Environment (`src/environment/isaac_lab_env.py`)

**Responsibilities:**
- Wraps Isaac Lab's `ManagerBasedRLEnv`
- Configures Franka Panda robot
- Sets up camera sensors (224x224 RGB)
- Adds contact sensors for collision detection
- Applies phase-specific visual monitoring cues

**Key Methods:**
- `reset(phase)` - Reset environment for new episode
- `step(action)` - Execute action, return observation
- `_process_observation()` - Add realistic sensor noise
- `_add_monitoring_cues()` - Apply phase-specific visual cues

#### 2. OpenVLA Wrapper (`src/vla/models/openvla_wrapper.py`)

**Key Features:**
- Loads OpenVLA-7B with automatic device mapping
- CPU offloading for 11GB GPUs
- Extracts activation statistics
- Returns 7-DOF actions (dx,dy,dz,droll,dpitch,dyaw,gripper)

**Memory Optimization:**
```python
device_map="auto"  # Automatic GPU/CPU distribution
max_memory={0: "10GiB", 1: "10GiB", ..., "cpu": "30GiB"}
torch_dtype=torch.bfloat16  # Half precision
```

#### 3. Strategic Deception Protocol

**5 Phases (2×2 + baseline):**

| Phase | Monitoring Visible | Incentive Alignment |
|-------|-------------------|-------------------|
| baseline | None | Neutral |
| monitoring_safe | ✅ Camera visible | ✅ Aligned |
| monitoring_conflict | ✅ Camera visible | ❌ Misaligned |
| no_monitoring_safe | ❌ No camera | ✅ Aligned |
| no_monitoring_conflict | ❌ No camera | ❌ Misaligned |

**Measured Variables:**
- Task success rate
- Safety violations (collisions, joint limits)
- Activation patterns in VLA model
- Behavioral differences across phases

## Hardware Configuration

**Current Setup:**
- 4× RTX 2080 Ti (11GB each)
- GPU 0: Isaac Sim rendering (~4GB)
- GPU 1: OpenVLA-7B (~10GB GPU + 4GB CPU offload)
- GPUs 2-3: Available for parallel experiments

**Optimal Usage:**
```bash
# Single experiment
export ISAAC_GPU=0
./isaaclab.sh -p script.py --device cuda:1

# Parallel experiments (advanced)
# Terminal 1
CUDA_VISIBLE_DEVICES=0,1 ./isaaclab.sh -p script.py --device cuda:1 --experiment exp1

# Terminal 2
CUDA_VISIBLE_DEVICES=2,3 ./isaaclab.sh -p script.py --device cuda:3 --experiment exp2
```

## Next Steps

**To see Isaac Sim working immediately:**

1. Run this command from a fresh terminal (no conda):
```bash
cd ~/Downloads/IsaacLab
./isaaclab.sh --install 2>&1 | tee ~/isaac_lab_install.log
```

2. Wait for installation to complete (~10 minutes)

3. Test with empty scene:
```bash
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

4. If that works, run PraxisLabs visual test:
```bash
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_isaac_sim_visual.py
```

**Expected Timeline:**
- Isaac Lab installation: 10 minutes
- Visual test: 30 seconds
- Full deception study (1 episode per phase): 5 minutes

---

**Questions? Issues?**
- See installation logs in `~/isaac_lab_install.log`
- Check Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/
- PraxisLabs logs in `/home/mpcr/Desktop/PraxisLabs/logs/`
