"""JAX implementation of a planar quadrotor.

A rigid body in the vertical (x, z) plane, pushed by two rotors at a distance
``arm_length`` from its center, as in Tedrake, Underactuated Robotics, ch. 3:

    m x'' = -(u1 + u2) sin(theta)
    m z'' = (u1 + u2) cos(theta) - m g
    I theta'' = arm_length (u1 - u2)

The task is to hover at the origin, starting from a random nearby state.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    x: jax.Array
    z: jax.Array
    theta: jax.Array
    x_dot: jax.Array
    z_dot: jax.Array
    theta_dot: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.81
    mass: float = 1.0
    inertia: float = 0.05
    arm_length: float = 0.25
    dt: float = 0.02
    max_steps_in_episode: int = 200


class PlanarQuadrotor(environment.Environment[EnvState, EnvParams]):
    """Planar quadrotor, the actions in [-1, 1] map each rotor thrust to [0, m g].

    The action 0 is hovering. The reward is the closeness to the origin, as in the
    cartpole task of PETS (Chua et al., 2018), minus a small penalty on the action.
    """

    _supports_transition_gradients = True

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        action = jnp.clip(jnp.reshape(action, (2,)), -1.0, 1.0)
        thrusts = params.mass * params.gravity / 2 * (1 + action)

        x_acc = -jnp.sum(thrusts) * jnp.sin(state.theta) / params.mass
        z_acc = jnp.sum(thrusts) * jnp.cos(state.theta) / params.mass - params.gravity
        theta_acc = params.arm_length * (thrusts[0] - thrusts[1]) / params.inertia

        # Semi-implicit Euler integration
        x_dot = state.x_dot + params.dt * x_acc
        z_dot = state.z_dot + params.dt * z_acc
        theta_dot = state.theta_dot + params.dt * theta_acc

        distance = state.x**2 + state.z**2
        reward = jnp.exp(-distance) - 0.01 * jnp.sum(action**2)

        state = EnvState(
            x=state.x + params.dt * x_dot,
            z=state.z + params.dt * z_dot,
            theta=state.theta + params.dt * theta_dot,
            x_dot=x_dot,
            z_dot=z_dot,
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
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Performs resetting of environment."""
        key_pos, key_rest = jax.random.split(key)
        pos = jax.random.uniform(key_pos, minval=-1.0, maxval=1.0, shape=(2,))
        rest = jax.random.uniform(key_rest, minval=-0.1, maxval=0.1, shape=(4,))
        state = EnvState(
            x=pos[0],
            z=pos[1],
            theta=rest[0],
            x_dot=rest[1],
            z_dot=rest[2],
            theta_dot=rest[3],
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        return jnp.array(
            [state.x, state.z, state.theta, state.x_dot, state.z_dot, state.theta_dot]
        )

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "PlanarQuadrotor-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.full((6,), jnp.finfo(jnp.float32).max)
        return spaces.Box(-high, high, (6,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.finfo(jnp.float32).max
        return spaces.Dict(
            {
                "x": spaces.Box(-high, high, (), jnp.float32),
                "z": spaces.Box(-high, high, (), jnp.float32),
                "theta": spaces.Box(-high, high, (), jnp.float32),
                "x_dot": spaces.Box(-high, high, (), jnp.float32),
                "z_dot": spaces.Box(-high, high, (), jnp.float32),
                "theta_dot": spaces.Box(-high, high, (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
