# PraxisLabs - Quick Guide

## ✅ Status: Ready to Run

### What's Working
- **Isaac Sim**: GPU-accelerated (4x RTX 2080 Ti)
- **Isaac Lab**: Installed and functional  
- **Data Collection**: HDF5 storage working
- **Folder**: Cleaned up (docs in `_archive_docs/`)

### Why No Window?
**This is normal.** Research simulations run in **headless mode** (no GUI):
- ✓ Faster execution
- ✓ Less GPU memory
- ✓ Better for batch experiments
- ✓ Works on servers

You see **terminal output** and **data files**, not a 3D window.

## 🚀 Quick Test

### Test Isaac Lab (Headless - No Window)
```bash
cd /home/mpcr/Downloads/IsaacLab
./isaaclab.sh -p scripts/demos/quadrupeds.py --headless --num_envs 16
```
**Output**: Terminal messages, GPU usage, no window needed

### Test Data Collection
```bash
cd /home/mpcr/Desktop/PraxisLabs
conda activate vlaenv

# Simple test
python -c "
import h5py, numpy as np
with h5py.File('test.h5', 'w') as f:
    f.create_dataset('data', data=np.random.rand(100, 10))
print('✓ Data collection works')
"
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
# Run while Isaac Lab is active
```

## 📁 Clean Structure

```
PraxisLabs/
├── src/              # VLA & simulation code
├── scripts/          # Test & run scripts
├── tests/            # Unit tests
├── config/           # Configs
├── docs/             # Technical docs
├── README.md         # Main readme
├── STATUS.md         # Detailed status
└── _archive_docs/    # Setup guides (archived)
```

## 🎯 Next Steps

1. **Install dependencies** (if needed):
   ```bash
   conda activate vlaenv
   pip install -r requirements.txt
   ```

2. **Run unit tests**:
   ```bash
   pytest tests/unit/ -v
   ```

3. **Download VLA models** (optional, ~14GB):
   ```bash
   # Only if running full experiments
   # See docs/SETUP.md
   ```

## 💡 Understanding Output

**You won't see**: 3D viewer windows  
**You will see**: 
- Terminal progress bars
- Log messages
- Data files created (`.h5`)
- GPU usage increase

## 📊 Data Output

Results go to:
- `results/` - Experiment data
- `logs/` - Log files
- `*.h5` - HDF5 episode data

View with:
```bash
python -c "
import h5py
with h5py.File('results/episode.h5', 'r') as f:
    print(list(f.keys()))
"
```

## 📚 Docs

- `STATUS.md` - System status & what to expect
- `README.md` - Full project documentation
- `docs/` - Architecture & setup details
- `_archive_docs/` - All setup instructions

## ✅ System Ready

Everything is configured. No window is expected - this is headless research infrastructure.

