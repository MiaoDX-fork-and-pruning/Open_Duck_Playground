# Newton Physics Backend Integration

This directory contains the Newton/MuJoCo-Warp integration for Open Duck Playground, providing GPU-accelerated physics simulation with potential 30-100x speedups over CPU MuJoCo.

## Overview

The Newton integration provides:
- GPU-accelerated physics stepping via NVIDIA Warp kernels
- JAX-compatible interface maintaining existing training infrastructure
- Automatic backend selection based on hardware availability
- Performance monitoring and comparison tools
- Physics accuracy verification

## Installation

On your Ubuntu server with NVIDIA GPU:

```bash
# Install Newton and MuJoCo-Warp
pip install newton-physics mujoco_warp

# Verify CUDA is available
python -c "import jax; print(jax.devices())"
```

## Usage

### 1. Environment Variable Control

```bash
# Use Newton backend
export OPEN_DUCK_BACKEND=newton

# Use MJX backend
export OPEN_DUCK_BACKEND=mjx

# Auto-select based on GPU availability (default)
export OPEN_DUCK_BACKEND=auto
```

### 2. Programmatic Control

```python
from playground.open_duck_mini_v2.env_factory import create_standing_env

# Explicitly use Newton
env = create_standing_env(xml_path, config, backend="newton")

# Explicitly use MJX
env = create_standing_env(xml_path, config, backend="mjx")

# Auto-select (uses Newton if GPU available)
env = create_standing_env(xml_path, config, backend="auto")
```

### 3. Training with Newton

The Newton backend is fully compatible with the existing training pipeline:

```python
# Training automatically uses the backend specified
python -m playground.open_duck_mini_v2.train --backend newton
```

## Performance Monitoring

The integration includes automatic performance monitoring:

```python
# Performance stats are logged every 1000 steps
# Compare backends after training
from playground.open_duck_mini_v2.performance_monitor import get_performance_logger
logger = get_performance_logger()
logger.compare_backends()
```

## Benchmarking

Run the benchmark script to compare performance:

```python
# On GPU server
python playground/open_duck_mini_v2/benchmark_physics.py
```

Expected speedups:
- Batch size 1: 10-30x
- Batch size 128: 50-70x
- Batch size 1024: 70-100x

## Physics Verification

Verify physics accuracy between backends:

```python
python playground/open_duck_mini_v2/verify_physics.py
```

This will:
- Check determinism of each backend
- Compare trajectories between MJX and Newton
- Generate comparison plots
- Report accuracy metrics (RMSE, MAE, correlation)

## Architecture

### File Structure

```
newton_base.py          # Base Newton environment class
newton_utils.py         # State conversion utilities
newton_standing.py      # Newton-accelerated standing task
env_factory.py          # Environment creation with backend selection
performance_monitor.py  # Performance tracking utilities
benchmark_physics.py    # Benchmarking script
verify_physics.py       # Physics accuracy verification
```

### Integration Pattern

The Newton integration uses JAX custom calls to maintain compatibility:

```
JAX/Flax Training Code
        ↓
    Brax Wrapper
        ↓
  Newton Environment
        ↓
  newton.jax.custom_call
        ↓
  Newton/Warp GPU Kernels
```

## Troubleshooting

### Newton not available
- Ensure you're on a machine with NVIDIA GPU
- Install newton-physics and mujoco_warp packages
- Check CUDA version compatibility

### Performance not improved
- Verify GPU is being used: `jax.devices()`
- Check batch size (larger batches see more speedup)
- Ensure JIT compilation is enabled

### Physics differences
- Small numerical differences are expected
- Check RMSE is < 1e-6 for positions
- Verify reward correlation > 0.99

## Future Optimizations

The following optimizations are planned:
- Batched sensor queries (TODO #18)
- Vectorized observation computation (TODO #19)
- Additional JIT compilation points (TODO #17)
- Multi-GPU support when available in Newton

## Notes

- The Newton backend requires NVIDIA GPU with CUDA 11.0+
- JAX must be installed with GPU support
- Physics accuracy is maintained within numerical precision
- All existing training code remains compatible