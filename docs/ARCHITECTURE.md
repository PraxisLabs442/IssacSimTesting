# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Experiment Runner                        │
│  - Multi-phase orchestration                                 │
│  - Episode management                                        │
│  - Results aggregation                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     Control Loop                             │
│  - Episode execution                                         │
│  - Step-by-step coordination                                 │
└─────────────┬──────────────────────┬────────────────────────┘
              │                      │
              ▼                      ▼
┌──────────────────────┐  ┌──────────────────────────────────┐
│   VLA Model          │  │   Isaac Lab Environment          │
│  - OpenVLA           │  │  - Physics simulation            │
│  - Action prediction │  │  - Robot control                 │
│  - Activation logs   │  │  - Camera rendering              │
└──────────┬───────────┘  └─────────────┬────────────────────┘
           │                            │
           │                            │
           ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Logging & Metrics                          │
│  - Trajectory logging (JSON/HDF5)                            │
│  - Safety metrics                                            │
│  - Activation tracking                                       │
└──────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. VLA Model Wrapper (`src/vla/`)
- **Purpose**: Interface with Vision-Language-Action models
- **Key Features**:
  - Model loading and device placement
  - Image preprocessing
  - Action prediction
  - Activation logging for deception analysis
  - Phase-aware instruction modification

### 2. Isaac Lab Environment (`src/environment/`)
- **Purpose**: Simulation environment management
- **Key Features**:
  - ManagerBasedRLEnv wrapper
  - Camera configuration
  - Phase-specific visual modifications
  - Safety monitoring hooks

### 3. Control Loop (`src/experiments/control_loop.py`)
- **Purpose**: Connect VLA model with environment
- **Key Features**:
  - Episode execution
  - Step-by-step action-observation loop
  - Metrics collection
  - Logging coordination

### 4. Logging Infrastructure (`src/logging/`)
- **Purpose**: Persistent data storage
- **Key Features**:
  - JSON format for small-scale experiments
  - HDF5 format for large-scale experiments
  - Buffered writing for efficiency
  - Activation and image logging

### 5. Safety Metrics (`src/metrics/`)
- **Purpose**: Compute safety-relevant metrics
- **Key Features**:
  - Collision detection and counting
  - Near-miss tracking
  - Constraint violation monitoring
  - Aggregate safety scoring

### 6. Experiment Runner (`src/experiments/experiment_runner.py`)
- **Purpose**: Orchestrate multi-phase experiments
- **Key Features**:
  - Phase management
  - Episode scheduling
  - Results aggregation
  - Report generation

## Data Flow

### Initialization
```
Config → VLA Loader → Model
Config → Env Builder → Isaac Lab Env
Config → Logger → File Handles
```

### Episode Loop
```
1. Env.reset(phase) → Observation
2. Observation → VLA.predict() → Action
3. Action → Env.step() → (Next Obs, Reward, Done, Info)
4. All data → Logger.log_step()
5. Info → Safety.compute_metrics() → Safety Metrics
6. Repeat until done
7. Trajectory → Safety.compute_episode_metrics()
8. Episode metrics → Logger.end_episode()
```

### Experiment Loop
```
For each phase:
  For each episode:
    Run episode
    Collect metrics
  Aggregate phase results
Generate final report
```

## Key Design Decisions

### 1. External Model
The VLA model runs in the same Python process but is logically separate from Isaac Sim. This allows:
- Independent scaling (model on GPU 1, sim on GPU 0)
- Easy model swapping
- Activation logging without sim interference

### 2. Buffered Logging
Logs are buffered before writing to reduce I/O overhead:
- Configurable buffer size
- Automatic flushing on episode end
- Supports both JSON and HDF5

### 3. Phase-Based Design
All components are phase-aware:
- Environment can modify visual cues per phase
- VLA can adjust instructions per phase
- Logging includes phase metadata
- Metrics are computed per phase

### 4. Safety-First Metrics
Safety is a first-class concern:
- Per-step safety metrics
- Episode-level aggregation
- Customizable thresholds
- Event tracking

## Extension Points

### Adding New VLA Models
1. Subclass `VLALoader` or modify `_load_model()`
2. Implement model-specific preprocessing
3. Update action decoding

### Adding New Environments
1. Create new config in `EnvironmentConfig`
2. Update `_create_environment()` with task-specific setup
3. Ensure observation dict includes 'rgb' and 'state'

### Adding New Phases
1. Add phase name to `ExperimentConfig.phases`
2. Update `_apply_phase_instruction()` in VLA loader
3. Add phase-specific logic in `env.reset()`

### Adding New Metrics
1. Extend `SafetyMetricsCalculator.compute_step_metrics()`
2. Update `compute_episode_metrics()` for aggregation
3. Include in experiment report

## Performance Considerations

### Memory
- Images can be large: use `compress_images=True` or `log_images=False`
- Activations are large: only log specific layers
- HDF5 is more memory-efficient than JSON for large datasets

### Compute
- Isaac Sim is GPU-intensive: use GPU 0
- VLA inference can be parallelized: use multiple GPUs
- Consider model quantization for faster inference

### Storage
- JSON: ~1-10 MB per episode (without images)
- HDF5: ~100 KB - 1 MB per episode (compressed)
- Activations: ~10-100 MB per episode depending on layers
