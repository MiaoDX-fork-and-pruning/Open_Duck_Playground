# Newton-enabled Joystick environment

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco import mjx
from mujoco.mjx._src import math

from . import joystick
from .newton_base import NewtonOpenDuckMiniV2Env
from . import constants
from playground.common.poly_reference_motion import PolyReferenceMotion
from .custom_rewards import reward_imitation
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
    reward_alive,
)

USE_IMITATION_REWARD = True  # From joystick.py


class NewtonJoystick(joystick.Joystick, NewtonOpenDuckMiniV2Env):
    """Newton-accelerated version of Joystick environment."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = joystick.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        backend: str = "mjx",
    ):
        self.use_newton = (backend == "newton")
        NewtonOpenDuckMiniV2Env.__init__(
            self,
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
            use_newton=self.use_newton,
        )
        joystick.Joystick._post_init(self)

    @property
    def backend_name(self) -> str:
        return "NEWTON" if self.use_newton else "MJX"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self.use_newton:
            self._newton_world.reset()
            data = mjx.get_data(self.mjx_model, self._newton_world.get_state())
            # Full reset logic adapted from joystick.py
            qpos = self._init_q
            qvel = jp.zeros(self.mjx_model.nv)
            rng, key = jax.random.split(rng)
            dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)
            base_qpos = self.get_floating_base_qpos(qpos)
            base_qpos = base_qpos.at[0:2].set(qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2] + dxy)
            rng, key = jax.random.split(rng)
            yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
            quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
            new_quat = math.quat_mul(qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7], quat)
            base_qpos = base_qpos.at[3:7].set(new_quat)
            qpos = self.set_floating_base_qpos(base_qpos, qpos)
            rng, key = jax.random.split(rng)
            qpos_noise = jax.random.uniform(key, (self._actuators,), minval=0.0, maxval=0.1) * self._qpos_noise_scale
            actuator_qpos = qpos[self.actuator_joint_qpos_addr] + qpos_noise
            qpos = qpos.at[self.actuator_joint_qpos_addr].set(actuator_qpos)
            rng, key = jax.random.split(rng)
            command = jax.random.uniform(key, (3,), minval=[self._config.lin_vel_x[0], self._config.lin_vel_y[0], self._config.ang_vel_yaw[0]], maxval=[self._config.lin_vel_x[1], self._config.lin_vel_y[1], self._config.ang_vel_yaw[1]])
            rng, key = jax.random.split(rng)
            neck_pitch = jax.random.uniform(key, (1,), minval=self._config.neck_pitch_range[0], maxval=self._config.neck_pitch_range[1]) * self._config.head_range_factor
            rng, key = jax.random.split(rng)
            head_pitch = jax.random.uniform(key, (1,), minval=self._config.head_pitch_range[0], maxval=self._config.head_pitch_range[1]) * self._config.head_range_factor
            rng, key = jax.random.split(rng)
            head_yaw = jax.random.uniform(key, (1,), minval=self._config.head_yaw_range[0], maxval=self._config.head_yaw_range[1]) * self._config.head_range_factor
            rng, key = jax.random.split(rng)
            head_roll = jax.random.uniform(key, (1,), minval=self._config.head_roll_range[0], maxval=self._config.head_roll_range[1]) * self._config.head_range_factor
            command = jp.concatenate([command, neck_pitch, head_pitch, head_yaw, head_roll])
            rng, key = jax.random.split(rng)
            push_interval = jax.random.uniform(key, minval=self._config.push_config.interval_range[0], maxval=self._config.push_config.interval_range[1])
            push_interval_steps = int(push_interval / self.ctrl_dt)
            current_reference_motion = self.PRM.get_reference_motion(command[0], command[1], command[2], 0) if USE_IMITATION_REWARD else jp.zeros(0)
            info = {
                "rng": rng,
                "command": command,
                "push": jp.array([0.0, 0.0]),
                "push_step": 0,
                "push_interval_steps": push_interval_steps,
                "action_history": jp.zeros(self._config.noise_config.action_max_delay * self._actuators),
                "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
                "imitation_i": 0,
                "current_reference_motion": current_reference_motion,
            }
            metrics = {k: jp.zeros(()) for k, v in self._config.reward_config.scales.items() if v != 0}
            metrics["swing_peak"] = jp.zeros(())
            contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
            obs = self._get_obs(data, info, contact)
            return mjx_env.State(data, obs, 0.0, 0.0, metrics, info)
        else:
            return joystick.Joystick.reset(self, rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if USE_IMITATION_REWARD:
            state.info["imitation_i"] += 1
            state.info["imitation_i"] = state.info["imitation_i"] % self.PRM.nb_steps_in_period
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                state.info["command"][0], state.info["command"][1], state.info["command"][2], state.info["imitation_i"]
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)
        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(state.info["rng"], 4)
        action_history = jp.roll(state.info["action_history"], self._actuators).at[:self._actuators].set(action)
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(action_delay_rng, (1,), minval=self._config.noise_config.action_min_delay, maxval=self._config.noise_config.action_max_delay)
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(push2_rng, minval=self._config.push_config.magnitude_range[0], maxval=self._config.push_config.magnitude_range[1])
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)]) * (jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0) * self._config.push_config.enable
        qvel = state.data.qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2].set(push * push_magnitude + state.data.qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)
        motor_targets = self._default_actuator + action_w_delay * self._config.action_scale
        state.info["motor_targets"] = motor_targets
        if self.use_newton:
            data = self._newton_step_jax(data, motor_targets, self.n_substeps)  # Assuming base has this method
        else:
            data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
        state.info["push_step"] += 1
        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, state.info, contact)
        reward = sum(v * getattr(self, k)(data, state.info, contact) for k, v in self._config.reward_config.scales.items() if v != 0)
        done = jp.where(data.qpos[2] < 0.2, 1.0, 0.0)  # Example termination
        metrics = {f"{'reward' if v > 0 else 'cost'}/{k}": getattr(self, k)(data, state.info, contact) for k, v in self._config.reward_config.scales.items() if v != 0}
        return mjx_env.State(data, obs, reward, done, metrics, state.info) 