"""JAX implementation of Acrobot with a continuous torque.

Same dynamics as Acrobot-v1, with the three discrete torques replaced by a
continuous torque, which makes the transition differentiable in the action.
The joint angles are not wrapped, so that the state changes continuously.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import spaces
from gymnax.environments.classic_control import acrobot
from gymnax.environments.classic_control.acrobot import EnvState


@struct.dataclass
class EnvParams(acrobot.EnvParams):
    max_torque: float = 1.0  # Torque of the action 1, as in Acrobot-v1
    dense_reward: bool = False  # Height of the tip minus the goal height, at each step
    goal_bonus: float = 100.0  # Added to the dense reward when the tip crosses the line


class ContinuousAcrobot(acrobot.Acrobot):
    """Acrobot with a torque in [-1, 1], scaled by ``max_torque``."""

    _supports_transition_gradients = True

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for AcrobotContinuous-v0
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        torque = params.max_torque * jnp.clip(jnp.squeeze(action), -1.0, 1.0)
        torque = torque + jax.random.uniform(
            key,
            shape=(),
            minval=-params.torque_noise_max,
            maxval=params.torque_noise_max,
        )

        s_augmented = jnp.array(
            [
                state.joint_angle1,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
                torque,
            ]
        )
        ns = acrobot.rk4(s_augmented, params)

        state = EnvState(
            joint_angle1=ns[0],
            joint_angle2=ns[1],
            velocity_1=jnp.clip(ns[2], -params.max_vel_1, params.max_vel_1),
            velocity_2=jnp.clip(ns[3], -params.max_vel_2, params.max_vel_2),
            time=jnp.int32(state.time + 1),
        )

        # -1 per step until the tip is above the line, as in Acrobot-v1, or
        # the height of the tip below the line, which is negative until then,
        # plus a bonus for crossing it, so that a planner does not settle for
        # swinging just below the line
        done = self.is_terminal(state, params)
        reached = self.is_terminated(state, params)
        sparse = -1.0 * (1 - reached)
        height = -jnp.cos(state.joint_angle1) - jnp.cos(
            state.joint_angle1 + state.joint_angle2
        )
        dense = height - 1.0 + params.goal_bonus * reached
        reward = jax.lax.select(params.dense_reward, dense, sparse)
        observation, state = self._apply_transition_gradient_policy(
            self.get_obs(state), state
        )
        return (
            observation,
            state,
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "AcrobotContinuous-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)
