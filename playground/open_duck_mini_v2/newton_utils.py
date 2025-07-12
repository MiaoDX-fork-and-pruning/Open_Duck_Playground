"""Newton state conversion utilities for Open Duck Mini V2."""

import jax
import jax.numpy as jp
from typing import Dict, Any, Optional
import mujoco
from mujoco import mjx
from dataclasses import dataclass

# Newton state structure based on expected API
@dataclass
class NewtonState:
    """Newton state representation."""
    q: jp.ndarray  # Joint positions
    dq: jp.ndarray  # Joint velocities  
    act: Optional[jp.ndarray] = None  # Actuator states
    time: float = 0.0
    
    @property
    def qpos(self):
        """Alias for MJX compatibility."""
        return self.q
    
    @property
    def qvel(self):
        """Alias for MJX compatibility."""
        return self.dq


def mjx_to_newton_state(mjx_data: mjx.Data) -> Dict[str, jp.ndarray]:
    """Convert MJX data to Newton state dictionary.
    
    Args:
        mjx_data: MJX data structure
        
    Returns:
        Dictionary with Newton state format
    """
    state_dict = {
        'q': mjx_data.qpos,
        'dq': mjx_data.qvel,
        'time': mjx_data.time
    }
    
    # Add actuator states if present
    if hasattr(mjx_data, 'act') and mjx_data.act is not None:
        state_dict['act'] = mjx_data.act
        
    return state_dict


def newton_to_mjx_data(
    newton_state: Dict[str, jp.ndarray],
    mjx_model: mjx.Model,
    original_data: mjx.Data
) -> mjx.Data:
    """Convert Newton state back to MJX data.
    
    Args:
        newton_state: Newton state dictionary
        mjx_model: MJX model for data structure
        original_data: Original MJX data for fields not in Newton state
        
    Returns:
        Updated MJX data
    """
    # Start with original data
    updated_data = original_data
    
    # Update core physics state
    if 'q' in newton_state:
        updated_data = updated_data.replace(qpos=newton_state['q'])
    if 'dq' in newton_state:
        updated_data = updated_data.replace(qvel=newton_state['dq'])
    if 'act' in newton_state:
        updated_data = updated_data.replace(act=newton_state['act'])
    if 'time' in newton_state:
        updated_data = updated_data.replace(time=newton_state['time'])
        
    # Newton should compute these during step:
    # - xpos, xquat, xmat (body transforms)
    # - sensordata (sensor readings)
    # - contact information
    
    return updated_data


def create_newton_step_fn(world, n_substeps: int = 1):
    """Create a Newton step function compatible with JAX.
    
    Args:
        world: Newton world instance
        n_substeps: Number of substeps per environment step
        
    Returns:
        JAX-compatible step function
    """
    
    def newton_step(state_dict: Dict[str, jp.ndarray], ctrl: jp.ndarray) -> Dict[str, jp.ndarray]:
        """Step Newton physics.
        
        Args:
            state_dict: State dictionary with q, dq, etc.
            ctrl: Control inputs
            
        Returns:
            Updated state dictionary
        """
        # Create Newton state from dictionary
        state = NewtonState(
            q=state_dict['q'],
            dq=state_dict['dq'],
            act=state_dict.get('act'),
            time=state_dict.get('time', 0.0)
        )
        
        # Step physics with Newton
        # This assumes Newton provides a JAX-compatible step
        for _ in range(n_substeps):
            state = world.step(state, ctrl)
        
        # Convert back to dictionary
        next_state_dict = {
            'q': state.q,
            'dq': state.dq,
            'time': state.time + world.dt * n_substeps
        }
        
        if state.act is not None:
            next_state_dict['act'] = state.act
            
        return next_state_dict
    
    # Return JIT-compiled function
    return jax.jit(newton_step)


def extract_sensor_data_newton(
    world,
    state: NewtonState,
    sensor_ids: Dict[str, int]
) -> Dict[str, jp.ndarray]:
    """Extract sensor data from Newton state.
    
    Args:
        world: Newton world instance
        state: Newton state
        sensor_ids: Dictionary mapping sensor names to IDs
        
    Returns:
        Dictionary of sensor readings
    """
    sensors = {}
    
    # Newton should provide sensor APIs similar to:
    # - world.get_imu_data(state, sensor_id)
    # - world.get_accelerometer_data(state, sensor_id)
    # - world.get_force_sensor_data(state, sensor_id)
    
    # For now, return placeholder
    # This will be implemented when Newton API is available
    
    return sensors


def get_contact_forces_newton(world, state: NewtonState) -> jp.ndarray:
    """Get contact forces from Newton.
    
    Args:
        world: Newton world instance
        state: Newton state
        
    Returns:
        Contact force array
    """
    # Newton should provide contact API
    # For example: world.get_contact_forces(state)
    
    # Placeholder implementation
    return jp.zeros((0,))