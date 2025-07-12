"""Verify physics accuracy between MJX and Newton backends."""

import jax
import jax.numpy as jp
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from .env_factory import create_standing_env
from playground.common.configs import flat_terrain_cfg


def compare_trajectories(
    xml_path: str,
    num_steps: int = 1000,
    seed: int = 42,
    save_plots: bool = True
) -> Dict[str, np.ndarray]:
    """Compare physics trajectories between MJX and Newton.
    
    Args:
        xml_path: Path to robot XML
        num_steps: Number of steps to simulate
        seed: Random seed for reproducibility
        save_plots: Whether to save comparison plots
        
    Returns:
        Dictionary with trajectory differences
    """
    # Create environments with each backend
    config = flat_terrain_cfg.get_config()
    
    print("Creating MJX environment...")
    mjx_env = create_standing_env(xml_path, config, backend="mjx")
    
    print("Creating Newton environment...")
    newton_env = create_standing_env(xml_path, config, backend="newton")
    
    # Initialize with same random seed
    rng = jax.random.PRNGKey(seed)
    
    # Reset both environments
    mjx_state = mjx_env.reset(rng)
    newton_state = newton_env.reset(rng)
    
    # Storage for trajectories
    mjx_trajectory = {
        'qpos': [],
        'qvel': [],
        'reward': [],
        'done': []
    }
    
    newton_trajectory = {
        'qpos': [],
        'qvel': [],
        'reward': [],
        'done': []
    }
    
    # Generate same random actions
    rng, action_rng = jax.random.split(rng)
    actions = jax.random.uniform(
        action_rng, 
        shape=(num_steps, mjx_env.action_size),
        minval=-1.0,
        maxval=1.0
    )
    
    print(f"Running {num_steps} steps...")
    
    # Simulate both environments
    for i in range(num_steps):
        action = actions[i]
        
        # Step MJX
        mjx_state = mjx_env.step(mjx_state, action)
        mjx_trajectory['qpos'].append(mjx_state.data.qpos)
        mjx_trajectory['qvel'].append(mjx_state.data.qvel)
        mjx_trajectory['reward'].append(mjx_state.reward)
        mjx_trajectory['done'].append(mjx_state.done)
        
        # Step Newton
        newton_state = newton_env.step(newton_state, action)
        newton_trajectory['qpos'].append(newton_state.data.qpos)
        newton_trajectory['qvel'].append(newton_state.data.qvel)
        newton_trajectory['reward'].append(newton_state.reward)
        newton_trajectory['done'].append(newton_state.done)
        
        if i % 100 == 0:
            print(f"  Step {i}/{num_steps}")
    
    # Convert to arrays
    for key in mjx_trajectory:
        mjx_trajectory[key] = jp.stack(mjx_trajectory[key])
        newton_trajectory[key] = jp.stack(newton_trajectory[key])
    
    # Calculate differences
    differences = {}
    for key in ['qpos', 'qvel']:
        diff = mjx_trajectory[key] - newton_trajectory[key]
        differences[f'{key}_mae'] = jp.mean(jp.abs(diff))
        differences[f'{key}_rmse'] = jp.sqrt(jp.mean(diff**2))
        differences[f'{key}_max'] = jp.max(jp.abs(diff))
    
    # Reward difference
    reward_diff = mjx_trajectory['reward'] - newton_trajectory['reward']
    differences['reward_mae'] = jp.mean(jp.abs(reward_diff))
    differences['reward_correlation'] = jp.corrcoef(
        mjx_trajectory['reward'], 
        newton_trajectory['reward']
    )[0, 1]
    
    print("\nPhysics Accuracy Comparison:")
    print("="*50)
    for key, value in differences.items():
        print(f"{key}: {value:.6f}")
    
    if save_plots:
        _plot_comparison(mjx_trajectory, newton_trajectory, differences)
    
    return differences


def _plot_comparison(mjx_traj: Dict, newton_traj: Dict, differences: Dict):
    """Plot trajectory comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot position trajectories
    ax = axes[0, 0]
    time_steps = np.arange(len(mjx_traj['qpos']))
    
    # Plot first few joint positions
    for i in range(min(3, mjx_traj['qpos'].shape[1])):
        ax.plot(time_steps, mjx_traj['qpos'][:, i], 
                label=f'MJX q{i}', linestyle='-', alpha=0.7)
        ax.plot(time_steps, newton_traj['qpos'][:, i], 
                label=f'Newton q{i}', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Joint Position')
    ax.set_title('Joint Position Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot velocity trajectories
    ax = axes[0, 1]
    for i in range(min(3, mjx_traj['qvel'].shape[1])):
        ax.plot(time_steps, mjx_traj['qvel'][:, i], 
                label=f'MJX v{i}', linestyle='-', alpha=0.7)
        ax.plot(time_steps, newton_traj['qvel'][:, i], 
                label=f'Newton v{i}', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Joint Velocity')
    ax.set_title('Joint Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot rewards
    ax = axes[1, 0]
    ax.plot(time_steps, mjx_traj['reward'], label='MJX', alpha=0.7)
    ax.plot(time_steps, newton_traj['reward'], label='Newton', alpha=0.7)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot error metrics
    ax = axes[1, 1]
    qpos_error = jp.mean(jp.abs(mjx_traj['qpos'] - newton_traj['qpos']), axis=1)
    qvel_error = jp.mean(jp.abs(mjx_traj['qvel'] - newton_traj['qvel']), axis=1)
    
    ax.plot(time_steps, qpos_error, label='Position Error', alpha=0.7)
    ax.plot(time_steps, qvel_error, label='Velocity Error', alpha=0.7)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Trajectory Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Add summary text
    fig.text(0.02, 0.02, 
             f"RMSE qpos: {differences['qpos_rmse']:.6f}, "
             f"RMSE qvel: {differences['qvel_rmse']:.6f}, "
             f"Reward correlation: {differences['reward_correlation']:.4f}",
             fontsize=10, ha='left')
    
    # Save plot
    output_path = Path(__file__).parent / "physics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()


def verify_determinism(xml_path: str, backend: str, num_runs: int = 5):
    """Verify that physics is deterministic for a given backend."""
    config = flat_terrain_cfg.get_config()
    env = create_standing_env(xml_path, config, backend=backend)
    
    # Fixed seed and actions
    seed = 12345
    rng = jax.random.PRNGKey(seed)
    
    # Generate fixed actions
    action_rng = jax.random.PRNGKey(67890)
    actions = jax.random.uniform(
        action_rng,
        shape=(100, env.action_size),
        minval=-1.0,
        maxval=1.0
    )
    
    print(f"\nVerifying determinism for {backend} backend...")
    
    trajectories = []
    for run in range(num_runs):
        # Reset with same seed
        state = env.reset(rng)
        qpos_traj = [state.data.qpos]
        
        # Run same actions
        for action in actions:
            state = env.step(state, action)
            qpos_traj.append(state.data.qpos)
            
        trajectories.append(jp.stack(qpos_traj))
    
    # Check if all trajectories are identical
    all_identical = True
    for i in range(1, num_runs):
        diff = jp.max(jp.abs(trajectories[0] - trajectories[i]))
        if diff > 1e-10:
            all_identical = False
            print(f"  Run {i} differs by max {diff:.2e}")
    
    if all_identical:
        print(f"  ✓ {backend} is deterministic")
    else:
        print(f"  ✗ {backend} is NOT deterministic")
        
    return all_identical


if __name__ == "__main__":
    # Run accuracy comparison
    xml_path = str(Path(__file__).parent / "assets" / "open_duck_mini_v2_backlash_zeropos.xml")
    
    # Verify determinism
    verify_determinism(xml_path, "mjx")
    # verify_determinism(xml_path, "newton")  # Uncomment when Newton is available
    
    # Compare trajectories
    # differences = compare_trajectories(xml_path)  # Uncomment when Newton is available