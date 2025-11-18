# VLA Interaction Modes - Quick Reference

## 🎮 4 Ways to Interact with Your VLA Model

```
┌─────────────────────────────────────────────────────────────┐
│              MODE 1: GUI TEST (Visual)                      │
├─────────────────────────────────────────────────────────────┤
│  What: Watch robot move in Isaac Sim window                │
│  When: Debugging, demos, understanding behavior             │
│  Speed: Slow (rendering overhead)                           │
│  Command: ./run_test_gui.sh                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            MODE 2: HEADLESS TEST (Fast)                     │
├─────────────────────────────────────────────────────────────┤
│  What: No window, just logs                                 │
│  When: Production runs, overnight experiments               │
│  Speed: Fast (no rendering)                                 │
│  Command: ./run_test_headless.sh                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│        MODE 3: CUSTOM INSTRUCTION (Scripted)                │
├─────────────────────────────────────────────────────────────┤
│  What: Test specific instructions                           │
│  When: Comparing different commands                         │
│  Speed: Depends on --headless flag                          │
│  Command: test_custom_instruction.py --instruction "..."    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         MODE 4: INTERACTIVE (Real-time)                     │
├─────────────────────────────────────────────────────────────┤
│  What: Type commands, watch results                         │
│  When: Exploring model capabilities                         │
│  Speed: Depends on --headless flag                          │
│  Command: interactive_vla.py                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Mode 1: GUI Test (Watch & Learn)

### Use Case
- First-time setup verification
- Understanding what the model does
- Debugging weird behavior
- Creating demos/videos

### Command
```bash
cd /home/mpcr/Desktop/PraxisLabs
./run_test_gui.sh
```

### What You See
```
┌─────────────────────────────────────────┐
│        Isaac Sim Window                 │
│  ┌───────────────────────────────┐      │
│  │   ┌──────┐       ┌────┐      │      │
│  │   │Robot │  -->  │Cube│      │      │
│  │   │ Arm  │       └────┘      │      │
│  │   └──────┘                    │      │
│  │                               │      │
│  │   📹 (monitoring icon)        │      │
│  └───────────────────────────────┘      │
│                                         │
│  Controls:                              │
│   - Mouse: Rotate view                  │
│   - Scroll: Zoom                        │
│   - WASD: Move camera                   │
└─────────────────────────────────────────┘

Terminal Output:
✓ Isaac Sim launched
✓ Environment created  
✓ VLA model loaded: 7.51B parameters
  
  Step   0/100 | Reward:  0.346
  Step  20/100 | Reward:  0.001
  ...
```

### Pros & Cons
✅ See exactly what's happening  
✅ Debug visually  
✅ Great for learning  

❌ Slower (30-40% overhead)  
❌ Uses more GPU memory  
❌ Requires display  

---

## 🚀 Mode 2: Headless Test (Production)

### Use Case
- Long experiments (overnight)
- Batch processing
- Multiple parallel runs
- When speed matters

### Command
```bash
cd /home/mpcr/Desktop/PraxisLabs
./run_test_headless.sh

# Or for full study:
cd /home/mpcr/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/run_deception_study.py \
  --headless \
  --model openvla-7b \
  --tasks pick_place assembly \
  --episodes-per-phase 50
```

### What You See
```bash
# Terminal only - no window
================================================================================
VLA + Isaac Lab Test - Headless Mode
================================================================================

✓ Isaac Sim launched (headless)
✓ Environment created
✓ VLA model loaded

  Step   0/100 | Reward:  0.346 | Done: False
  Step  20/100 | Reward:  0.001 | Done: False
  Step  40/100 | Reward:  0.000 | Done: False
  ...

✓ Episode complete
  Total reward: 2.45
  Success: False
```

### Monitor Progress
```bash
# Watch logs in real-time
tail -f /tmp/isaaclab_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check if still running
ps aux | grep isaaclab
```

### Pros & Cons
✅ Much faster (30-40% speedup)  
✅ Less GPU memory  
✅ Can run overnight  
✅ Run multiple in parallel  

❌ No visual feedback  
❌ Harder to debug  

---

## 🎯 Mode 3: Custom Instruction (Testing)

### Use Case
- Test specific commands
- Compare different instructions
- Verify instruction understanding
- A/B testing

### Command
```bash
cd /home/mpcr/Downloads/IsaacLab

# Test 1: Safety emphasis
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Pick up the cube very carefully avoiding all collisions" \
  --num-steps 100

# Test 2: Speed emphasis
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Pick up the cube as fast as possible" \
  --num-steps 100 \
  --headless

# Test 3: Complex task
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Move left, then pick up the cube, then place it on the right" \
  --num-steps 200
```

### What You See
```bash
================================================================================
CUSTOM INSTRUCTION TEST
================================================================================

Instruction: 'Pick up the cube very carefully avoiding all collisions'
Steps: 100
VLA Device: cuda:1

Creating Isaac Lab environment...
✓ Environment created
  Action space: Box(-inf, inf, (1, 8), float32)

Loading OpenVLA-7B on cuda:1...
✓ VLA model loaded successfully

================================================================================
RUNNING EPISODE
================================================================================

Step   0/100 | Reward:  0.346 | Total:   0.35 | Done: False
Step  20/100 | Reward:  0.001 | Total:   2.14 | Done: False
Step  40/100 | Reward:  0.000 | Total:   3.89 | Done: False
...

================================================================================
TEST COMPLETE
================================================================================

Instruction: 'Pick up the cube very carefully avoiding all collisions'
Steps executed: 100
Final episode reward: 5.23
```

### Comparison Workflow
```bash
# Run different instructions
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Move slowly and carefully" \
  --num-steps 100 \
  --headless | tee slow.log

./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Move as fast as possible" \
  --num-steps 100 \
  --headless | tee fast.log

# Compare results
grep "Total reward" slow.log fast.log
```

### Pros & Cons
✅ Test specific instructions  
✅ Easy comparison  
✅ Scriptable (for loops)  
✅ Can run headless or GUI  

❌ One instruction per run  
❌ Needs restart for new instruction  

---

## 🎮 Mode 4: Interactive (Exploration)

### Use Case
- Exploring model capabilities
- Quick testing of ideas
- Finding interesting behaviors
- Demos/presentations

### Command
```bash
cd /home/mpcr/Downloads/IsaacLab

# Interactive mode (GUI)
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/interactive_vla.py

# Interactive mode (headless, faster)
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/interactive_vla.py --headless

# Fewer steps per instruction (faster feedback)
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/interactive_vla.py \
  --steps-per-instruction 30
```

### What You See
```bash
================================================================================
INTERACTIVE VLA CONTROL
================================================================================

Type instructions and watch the robot execute them!
Commands:
  - Type any instruction (e.g., 'pick up the cube')
  - Type 'quit' or 'exit' to stop
  - Type 'reset' to reset the environment
  - Type 'help' for example instructions

Setting up environment...
✓ Environment created
Loading OpenVLA-7B on cuda:1...
✓ VLA model loaded successfully

================================================================================
Enter instruction (or 'help', 'reset', 'quit'): pick up the cube

Instruction #1
  Executing: 'pick up the cube'
  Running 50 steps...
    Step  0/50 | Reward:  0.346
    Step 10/50 | Reward:  0.123
    Step 20/50 | Reward:  0.045
    Step 30/50 | Reward:  0.012
    Step 40/50 | Reward:  0.001
  ✓ Completed! Total reward: 5.67

================================================================================
Enter instruction (or 'help', 'reset', 'quit'): move to the left slowly

Instruction #2
  Executing: 'move to the left slowly'
  Running 50 steps...
    Step  0/50 | Reward:  0.234
    Step 10/50 | Reward:  0.089
    ...

================================================================================
Enter instruction (or 'help', 'reset', 'quit'): help

  Example instructions:
    - 'Pick up the red cube'
    - 'Move to the left slowly'
    - 'Place the object on the target'
    - 'Avoid obstacles and reach the goal'
    - 'Move as quickly as possible'
    - 'Be very careful and avoid collisions'

================================================================================
Enter instruction (or 'help', 'reset', 'quit'): quit

✓ Exiting interactive mode
```

### Interactive Commands
```
help   - Show example instructions
reset  - Reset the environment
quit   - Exit interactive mode
[any]  - Execute as instruction
```

### Pros & Cons
✅ Real-time feedback  
✅ Try many instructions quickly  
✅ No restart needed  
✅ Great for exploration  

❌ Manual (not scriptable)  
❌ No automatic logging  
❌ Results not saved  

---

## 🔍 How to Verify OpenVLA is Working

### Check 1: Terminal Output
```bash
./run_test_gui.sh 2>&1 | grep -E "OpenVLA|dummy|Applied|parameters"
```

**✅ Working:**
```
Loading OpenVLA-7B model on cuda:1...
Applied compatibility patch: _supports_sdpa
OpenVLA model loaded successfully
Model parameters: 7.51B
```

**❌ Not working:**
```
Using dummy action - model not loaded
Failed to load OpenVLA model: ...
Falling back to dummy mode
```

### Check 2: GPU Memory
```bash
nvidia-smi
```

**✅ OpenVLA loaded:**
```
| GPU 1  RTX 2080 Ti     8200MiB / 11510MiB |  ← 7-8GB = OpenVLA
```

**❌ Dummy mode:**
```
| GPU 1  RTX 2080 Ti      500MiB / 11510MiB |  ← ~500MB = dummy
```

### Check 3: Action Behavior

**OpenVLA actions:**
- Purposeful movements
- Reaches toward objects
- Varies with instruction
- Magnitude: 0.1-0.5 typical

**Dummy actions:**
- Random jitter
- No task direction
- Ignores instruction
- Magnitude: ~0.01

---

## 📊 Comparison Table

| Feature | GUI | Headless | Custom | Interactive |
|---------|-----|----------|--------|-------------|
| **Visual feedback** | ✅ | ❌ | Optional | Optional |
| **Speed** | Slow | Fast | Depends | Depends |
| **Scriptable** | ✅ | ✅ | ✅ | ❌ |
| **Flexibility** | Low | Low | Medium | High |
| **Debugging** | ✅✅✅ | ✅ | ✅✅ | ✅✅ |
| **Production** | ❌ | ✅✅✅ | ✅✅ | ❌ |
| **Learning** | ✅✅✅ | ✅ | ✅✅ | ✅✅✅ |

---

## 🎯 Recommended Workflow

### Day 1: Setup & Verification
```bash
# 1. Quick GUI test (verify everything works)
./run_test_gui.sh

# 2. Quick headless test (verify headless works)
./run_test_headless.sh
```

### Day 2: Exploration
```bash
# 3. Interactive mode (explore capabilities)
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/interactive_vla.py

# Try different instructions:
> pick up the cube
> move slowly to the left
> complete the task quickly
> be very careful
```

### Day 3: Testing
```bash
# 4. Test specific instructions (compare behaviors)
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Move carefully and avoid collisions" --headless

./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Complete as fast as possible" --headless
```

### Day 4+: Production
```bash
# 5. Full deception study (overnight run)
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/run_deception_study.py \
  --headless \
  --model openvla-7b \
  --tasks pick_place assembly obstacle \
  --episodes-per-phase 50
```

---

## 🚀 Quick Start Commands

```bash
# Quick visual test
cd ~/Desktop/PraxisLabs && ./run_test_gui.sh

# Quick headless test  
cd ~/Desktop/PraxisLabs && ./run_test_headless.sh

# Custom instruction (GUI)
cd ~/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/test_custom_instruction.py \
  --instruction "Pick up the cube carefully"

# Interactive mode (GUI)
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/interactive_vla.py

# Full study (headless)
./isaaclab.sh -p ~/Desktop/PraxisLabs/scripts/run_deception_study.py \
  --headless --model openvla-7b --tasks pick_place --episodes-per-phase 10
```

---

**Ready?** Start with: `cd ~/Desktop/PraxisLabs && ./run_test_gui.sh` 🚀

