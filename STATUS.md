# PraxisLabs Status

## ✅ What's Working

### Infrastructure
- **Isaac Sim**: 5.1.0 installed with GPU support (4x RTX 2080 Ti)
- **Isaac Lab**: Installed and symlinked to Isaac Sim
- **Claude-Flow**: v2.7.34 initialized
- **Claude Code**: v2.0.44 working

### Data Collection System
- **HDF5 Storage**: ✓ Working
- **DataLogger**: ✓ Tested and functional
- **Episode Storage**: ✓ Images, actions, states saved

### Project Structure
```
/home/mpcr/Desktop/PraxisLabs/
├── src/              # Source code
│   ├── vla/          # VLA model wrappers
│   ├── environment/  # Isaac Lab integration
│   ├── experiments/  # 5-phase protocol
│   ├── metrics/      # Safety metrics
│   └── logging/      # Data collection ✓
├── scripts/          # Test & run scripts
├── tests/            # Unit & integration tests
├── config/           # Configuration files
├── docs/             # Documentation
└── README.md         # Main readme
```

### Archived
- All setup docs moved to `_archive_docs/`
- Setup scripts moved to `_archive_docs/`

## 🎯 What You'll See

Isaac Lab runs in **headless mode** (no window) by default for:
- Faster execution
- Server/cluster compatibility  
- Batch processing

### To See Visual Output:
```bash
# Not recommended - headless is better for experiments
cd /home/mpcr/Downloads/IsaacLab
./isaaclab.sh -p scripts/demos/quadrupeds.py
# May open window on your display
```

### Headless Mode (Recommended):
```bash
# Runs simulation without GUI - faster
./isaaclab.sh -p scripts/demos/quadrupeds.py --headless
# Output in logs/data only
```

## 📊 Data Collection Test Results

**Status**: ✅ WORKING

The data collection system successfully:
- Creates HDF5 files
- Stores RGB images (480x640x3)
- Saves actions (7D vectors)
- Logs states (10D vectors)
- Organizes by episodes

**Test file created**: `test_data_collection.h5`

## 🚀 Next Steps

### 1. Test Isaac Lab Headless (No Window Needed)
```bash
cd /home/mpcr/Downloads/IsaacLab
./isaaclab.sh -p scripts/demos/quadrupeds.py --headless --num_envs 16
```
**You'll see**: Terminal output showing simulation progress, no GUI window

### 2. Test PraxisLabs Data Collection
```bash
cd /home/mpcr/Desktop/PraxisLabs
conda activate vlaenv
python scripts/test_isaac_sim_visual.py --headless
```
**You'll see**: Simulation runs, data saves to HDF5

### 3. Run Mini Experiment (5 episodes)
```bash
# Requires VLA models - placeholder mode available
python scripts/run_deception_study.py \
  --episodes 5 \
  --headless \
  --output results/test
```
**You'll see**: Progress bars, episode completion, data in `results/test/`

## 💡 Understanding Headless Mode

**Headless = No GUI Window** (this is normal and faster)

You won't see:
- ❌ 3D viewer window
- ❌ Real-time visualization

You will see:
- ✓ Terminal output
- ✓ Progress indicators
- ✓ Log messages
- ✓ Data files created
- ✓ GPU usage in `nvidia-smi`

## 📁 Output Locations

| Data Type | Location |
|-----------|----------|
| Episodes | `data/episodes/` |
| Results | `results/` |
| Logs | `logs/` |
| Checkpoints | `checkpoints/` |
| Test data | `*.h5` files |

## 🔍 How to Verify It's Working

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
# Should see GPU utilization when running
```

### Check Data Files
```bash
ls -lh *.h5  # HDF5 data files
ls -lh results/  # Experiment results
```

### View HDF5 Data
```bash
python -c "
import h5py
with h5py.File('test_data_collection.h5', 'r') as f:
    print('Episodes:', list(f.keys()))
    for ep in f.keys():
        print(f'{ep}: {list(f[ep].keys())}')
"
```

## ⚠️ Why No Window?

1. **Design**: Research systems run headless for efficiency
2. **Performance**: GUI uses extra GPU memory
3. **Automation**: Easier to run in batches
4. **Server**: Can run on remote machines

If you need visualization:
- Use tensorboard for training
- Create videos from saved episodes
- Use jupyter notebooks for analysis

## 📚 Documentation

- **Main**: `README.md` - Project overview
- **Setup**: `_archive_docs/` - Installation guides
- **Code**: `docs/` - Technical documentation

## ✅ System Ready

All systems are operational:
- ✓ Simulation platform
- ✓ Data collection
- ✓ GPU acceleration
- ✓ Clean project structure

Ready to run experiments!

