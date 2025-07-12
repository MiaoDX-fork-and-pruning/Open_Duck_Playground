"""Benchmark script to compare MJX vs Newton physics performance."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple
import mujoco
import mjx
from dataclasses import dataclass
from pathlib import Path

# Import Newton when available
try:
    import newton
    import mujoco_warp as mjwarp
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False
    print("Newton not available. Install newton-physics and mujoco_warp to run Newton benchmarks.")


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    backend: str
    batch_size: int
    num_steps: int
    total_time: float
    time_per_step: float
    steps_per_second: float
    
    
def load_mjx_model(model_path: str) -> Tuple[mujoco.MjModel, mjx.Model]:
    """Load MuJoCo model and convert to MJX."""
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def load_newton_model(model_path: str):
    """Load MuJoCo model and convert to Newton."""
    if not NEWTON_AVAILABLE:
        return None, None
    
    # Load model using Newton's MJCF loader
    model = newton.load_mjcf(model_path)
    world = newton.World(model, device="cuda" if jax.devices()[0].platform == "gpu" else "cpu")
    return model, world


def benchmark_mjx(model_path: str, batch_size: int, num_steps: int) -> BenchmarkResult:
    """Benchmark MJX physics stepping."""
    mj_model, mjx_model = load_mjx_model(model_path)
    
    # Initialize data
    data = mjx.make_data(mjx_model)
    
    # Create batched data
    def init_fn(key):
        return data
    
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    batched_data = jax.vmap(init_fn)(keys)
    
    # JIT compile step function
    @jax.jit
    def mjx_step_fn(data, ctrl):
        return mjx.step(mjx_model, data, ctrl)
    
    # Warmup
    ctrl = jnp.zeros((batch_size, mj_model.nu))
    for _ in range(10):
        batched_data = jax.vmap(mjx_step_fn)(batched_data, ctrl)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_steps):
        batched_data = jax.vmap(mjx_step_fn)(batched_data, ctrl)
    batched_data.qpos.block_until_ready()  # Wait for computation to complete
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / num_steps
    steps_per_second = num_steps / total_time
    
    return BenchmarkResult(
        backend="MJX",
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
        time_per_step=time_per_step,
        steps_per_second=steps_per_second
    )


def benchmark_newton(model_path: str, batch_size: int, num_steps: int) -> BenchmarkResult:
    """Benchmark Newton physics stepping."""
    if not NEWTON_AVAILABLE:
        return None
        
    model, world = load_newton_model(model_path)
    
    # Initialize state with batch
    state = world.initial_state(batch=batch_size)
    
    # Create control input
    ctrl = jnp.zeros((batch_size, model.nu))
    
    # JIT compile step function if using JAX backend
    if hasattr(newton, 'jax'):
        @jax.jit
        def newton_step_fn(state, ctrl):
            return world.step(state, ctrl)
    else:
        newton_step_fn = world.step
    
    # Warmup
    for _ in range(10):
        state = newton_step_fn(state, ctrl)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_steps):
        state = newton_step_fn(state, ctrl)
    
    # Wait for computation to complete
    if hasattr(state, 'q'):
        state.q.block_until_ready()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / num_steps
    steps_per_second = num_steps / total_time
    
    return BenchmarkResult(
        backend="Newton",
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
        time_per_step=time_per_step,
        steps_per_second=steps_per_second
    )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a nice format."""
    print("\n" + "="*80)
    print("Physics Engine Benchmark Results")
    print("="*80)
    print(f"{'Backend':<10} {'Batch':<8} {'Steps':<8} {'Total(s)':<10} {'ms/step':<10} {'steps/s':<10}")
    print("-"*80)
    
    for result in results:
        if result is not None:
            print(f"{result.backend:<10} {result.batch_size:<8} {result.num_steps:<8} "
                  f"{result.total_time:<10.3f} {result.time_per_step*1000:<10.3f} "
                  f"{result.steps_per_second:<10.1f}")
    
    print("="*80)
    
    # Calculate speedup if both results available
    mjx_result = next((r for r in results if r and r.backend == "MJX"), None)
    newton_result = next((r for r in results if r and r.backend == "Newton"), None)
    
    if mjx_result and newton_result:
        speedup = newton_result.steps_per_second / mjx_result.steps_per_second
        print(f"\nNewton speedup over MJX: {speedup:.2f}x")


def main():
    """Run benchmarks."""
    # Model path
    model_path = str(Path(__file__).parent / "assets" / "open_duck_mini_v2_backlash_zeropos.xml")
    
    # Benchmark configurations
    batch_sizes = [1, 32, 128, 512, 1024]
    num_steps = 1000
    
    results = []
    
    print(f"Running benchmarks on device: {jax.devices()[0]}")
    print(f"Model: {model_path}")
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking with batch size {batch_size}...")
        
        # Benchmark MJX
        try:
            mjx_result = benchmark_mjx(model_path, batch_size, num_steps)
            results.append(mjx_result)
            print(f"  MJX: {mjx_result.steps_per_second:.1f} steps/s")
        except Exception as e:
            print(f"  MJX benchmark failed: {e}")
            
        # Benchmark Newton
        if NEWTON_AVAILABLE:
            try:
                newton_result = benchmark_newton(model_path, batch_size, num_steps)
                results.append(newton_result)
                print(f"  Newton: {newton_result.steps_per_second:.1f} steps/s")
            except Exception as e:
                print(f"  Newton benchmark failed: {e}")
    
    # Print summary
    print_results(results)
    
    # Save results
    with open("benchmark_results.txt", "w") as f:
        f.write("Physics Engine Benchmark Results\n")
        f.write("="*80 + "\n")
        for result in results:
            if result:
                f.write(f"{result.backend}, batch={result.batch_size}, "
                       f"steps/s={result.steps_per_second:.1f}\n")


if __name__ == "__main__":
    main()