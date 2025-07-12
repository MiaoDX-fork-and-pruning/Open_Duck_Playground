# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Newton-based environment for Open Duck Mini V2 with JAX integration."""

from typing import Any, Dict, Optional, Union, Tuple
import functools

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from . import constants
from .base import get_assets, OpenDuckMiniV2Env
from . import newton_utils
from . import performance_monitor

# Newton imports (will be available when packages are installed)
try:
    import newton
    import mujoco_warp as mjwarp
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False
    print("Newton not available. Using MJX backend.")


class NewtonOpenDuckMiniV2Env(OpenDuckMiniV2Env):
    """Newton-accelerated version of Open Duck Mini V2 environment."""
    
    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        use_newton: bool = True,
    ) -> None:
        """Initialize Newton environment.
        
        Args:
            xml_path: Path to MuJoCo XML model
            config: Environment configuration
            config_overrides: Optional config overrides
            use_newton: Whether to use Newton backend (falls back to MJX if False or unavailable)
        """
        super().__init__(xml_path, config, config_overrides)
        
        self.use_newton = use_newton and NEWTON_AVAILABLE
        
        # Initialize performance monitoring
        self._perf_logger = performance_monitor.get_performance_logger()
        
        if self.use_newton:
            self._init_newton()
            performance_monitor.log_backend_info("Newton", 
                                               "cuda" if jax.devices()[0].platform == "gpu" else "cpu")
        else:
            print("Using MJX backend (Newton not available or disabled)")
            performance_monitor.log_backend_info("MJX", 
                                               "gpu" if jax.devices()[0].platform == "gpu" else "cpu")
    
    def _init_newton(self):
        """Initialize Newton physics backend."""
        # Load model with Newton
        xml_content = epath.Path(self._xml_path).read_text()
        self._newton_model = mjwarp.load_mjcf_from_string(
            xml_content, 
            assets=get_assets()
        )
        
        # Create Newton world
        device = "cuda" if jax.devices()[0].platform == "gpu" else "cpu"
        self._newton_world = mjwarp.World(
            self._newton_model, 
            device=device,
            batch_size=1  # Will be set properly during reset
        )
        
        # Create JAX-compatible step function
        self._newton_step_jax = self._create_newton_jax_step()
    
    def _create_newton_jax_step(self):
        """Create JAX-compatible Newton step function using custom call."""
        # Assuming newton_utils is properly defined
        newton_step_fn = newton_utils.create_newton_step_fn(
            self._newton_world, 
            self.n_substeps
        )
        return newton_step_fn
    
    def _newton_step_manual(self, state_dict: Dict, ctrl: jp.ndarray) -> Dict:
        """Manual Newton step for when JAX integration isn't available."""
        # This would convert JAX arrays to Newton format, step, and convert back
        # For now, fall back to MJX
        return None
    
    def _dict_to_newton_state(self, state_dict: Dict):
        """Convert state dictionary to Newton state format."""
        # Newton state format based on expected API
        # This will be implemented when Newton is available
        pass
    
    def _newton_state_to_dict(self, newton_state) -> Dict:
        """Convert Newton state to dictionary format."""
        # Convert back to dictionary format matching MJX
        pass
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset environment, handling Newton backend if enabled."""
        if self.use_newton:
            # Get batch size from rng shape
            batch_size = rng.shape[0] if len(rng.shape) > 1 else 1
            
            # Recreate world with correct batch size if needed
            if self._newton_world.batch_size != batch_size:
                device = "cuda" if jax.devices()[0].platform == "gpu" else "cpu"
                self._newton_world = mjwarp.World(
                    self._newton_model,
                    device=device,
                    batch_size=batch_size
                )
        
        # Use parent class reset
        return super().reset(rng)
    
    def step(
        self, 
        state: mjx_env.State, 
        action: jp.ndarray
    ) -> mjx_env.State:
        """Step the environment using Newton or MJX backend."""
        
        # Start performance timing
        backend_name = self.backend_name
        self._perf_logger.start_step(backend_name)
        
        try:
            if self.use_newton and self._newton_step_jax is not None:
                # Get motor targets from action (matching parent class logic)
                motor_targets = self._get_motor_targets(state, action)
                
                # Convert MJX state to Newton format
                state_dict = newton_utils.mjx_to_newton_state(state.data)
                
                # Step with Newton
                next_state_dict = self._newton_step_jax(state_dict, motor_targets)
                
                if next_state_dict is None:
                    # Fallback to MJX
                    return super().step(state, action)
                
                # Convert back to MJX data format
                next_data = newton_utils.newton_to_mjx_data(
                    next_state_dict,
                    self._mjx_model,
                    state.data
                )
                
                # Physics step is done, now handle the rest like parent class
                # This includes observations, rewards, termination, etc.
                # For now, we'll need to compute these using the updated data
                
                # Update state with new data
                next_state = state.replace(
                    data=next_data,
                    time=state.time + self.sim_dt * self.n_substeps
                )
                
                return next_state
            else:
                # Use MJX backend
                return super().step(state, action)
        finally:
            # End performance timing
            self._perf_logger.end_step(backend_name)
    
    def _get_motor_targets(self, state: mjx_env.State, action: jp.ndarray) -> jp.ndarray:
        """Extract motor targets from action."""
        # This should match the parent class implementation
        # Will be properly implemented based on the parent class logic
        return action
    
    @property
    def backend_name(self) -> str:
        """Return the name of the physics backend being used."""
        return "Newton" if self.use_newton else "MJX"