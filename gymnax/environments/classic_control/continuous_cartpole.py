"""JAX implementation of CartPole with a continuous force.

Same dynamics as CartPole, with the two discrete pushes replaced by a
continuous force, which makes the transition differentiable in the action.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import spaces
from gymnax.environments.classic_control import cartpole
from gymnax.environments.classic_control.cartpole import EnvState


@struct.dataclass
class EnvParams(cartpole.EnvParams):
    max_steps_in_episode: int = 200  # As in CartPole-v0


class ContinuousCartPole(cartpole.CartPole):
    """CartPole with a force in [-1, 1], scaled by ``force_mag``."""

    _supports_transition_gradients = True

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPoleContinuous-v0
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        force = params.force_mag * jnp.clip(jnp.squeeze(action), -1.0, 1.0)
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + params.polemass_length * state.theta_dot**2 * sintheta
        ) / params.total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length
            * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        # Only default Euler integration option available here!
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - prev_terminal

        # Update state dict and evaluate termination conditions
        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        observation, state = self._apply_transition_gradient_policy(
            self.get_obs(state), state
        )
        return (
            observation,
            state,
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPoleContinuous-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)


class CartPoleSwingUp(ContinuousCartPole):
    """Continuous CartPole starting with the pole hanging down, without termination.

    The reward follows the cartpole task of PETS (Chua et al., 2018): it is the
    closeness of the tip of the pole to its upright position above the origin,
    minus a small penalty on the action.
    """

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        pole_length = 2 * params.length
        tip_x = state.x + pole_length * jnp.sin(state.theta)
        tip_y = pole_length * jnp.cos(state.theta)
        distance = tip_x**2 + (tip_y - pole_length) ** 2
        force = jnp.clip(jnp.squeeze(action), -1.0, 1.0)
        reward = jnp.exp(-distance / pole_length**2) - 0.01 * force**2

        observation, state, _, done, info = super().step_env(key, state, action, params)
        return observation, state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        state = EnvState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=jnp.pi + init_state[2],
            theta_dot=init_state[3],
            time=0,
        )
        return self.get_obs(state), state

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPoleSwingUp-v0"
