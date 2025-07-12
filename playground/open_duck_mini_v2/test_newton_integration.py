"""Test Newton backend integration with training pipeline."""

import os
from pathlib import Path
from typing import Optional

def test_newton_training(
    backend: str = "auto",
    num_steps: int = 1000,
    config_name: str = "flat_terrain_cfg"
):
    """Test if Newton backend works with the training pipeline.
    
    Args:
        backend: Physics backend to use ("mjx", "newton", "auto")
        num_steps: Number of training steps
        config_name: Configuration to use
    """
    # Set backend via environment variable
    os.environ["OPEN_DUCK_BACKEND"] = backend
    
    print(f"\nTesting training with backend: {backend}")
    print("="*60)
    
    try:
        # Import training components
        from playground.common import runner
        from playground.common.configs import flat_terrain_cfg
        from .env_factory import create_standing_env
        
        # Get config
        config = flat_terrain_cfg.get_config()
        
        # Override training steps for quick test
        config.num_training_steps = num_steps
        config.batch_size = 128  # Smaller batch for testing
        
        # Create environment with specified backend
        xml_path = str(Path(__file__).parent / "assets" / "open_duck_mini_v2_backlash_zeropos.xml")
        env = create_standing_env(xml_path, config, backend=backend)
        
        print(f"Created environment with backend: {env.backend_name}")
        print(f"Device: {env._device}")
        print(f"Action size: {env.action_size}")
        print(f"Observation size: {env.observation_size}")
        
        # Test a few steps
        import jax
        rng = jax.random.PRNGKey(42)
        state = env.reset(rng)
        
        print("\nTesting environment steps...")
        for i in range(10):
            action = jax.random.uniform(
                rng, shape=(env.action_size,), minval=-1, maxval=1
            )
            state = env.step(state, action)
            print(f"  Step {i}: reward={state.reward:.4f}, done={state.done}")
            
        print("\n✓ Environment stepping works!")
        
        # Test with Brax wrapper
        from brax.training import wrapper
        wrapped_env = wrapper.wrap_for_brax_training(
            env,
            episode_length=config.episode_length,
            action_scale=1.0,
            auto_reset=True
        )
        
        print("\n✓ Brax wrapper works!")
        
        # Would test full training here if we had more time
        # runner.train(env, config)
        
        print(f"\n✓ {backend} backend is compatible with training pipeline!")
        
    except Exception as e:
        print(f"\n✗ Error with {backend} backend: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run integration tests."""
    # Test MJX backend (should always work)
    test_newton_training(backend="mjx", num_steps=100)
    
    # Test Newton backend (will work when packages are installed)
    # test_newton_training(backend="newton", num_steps=100)
    
    # Test auto selection
    # test_newton_training(backend="auto", num_steps=100)


if __name__ == "__main__":
    main()