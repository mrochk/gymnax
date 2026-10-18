"""Tests for the continuous torque Acrobot."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

import gymnax
from gymnax.environments.classic_control.acrobot import EnvState, wrap


def _state(s):
    return EnvState(
        joint_angle1=jnp.array(s[0]),
        joint_angle2=jnp.array(s[1]),
        velocity_1=jnp.array(s[2]),
        velocity_2=jnp.array(s[3]),
        time=0,
    )


def _wrapped(state):
    return np.array(
        [
            wrap(state.joint_angle1, -jnp.pi, jnp.pi),
            wrap(state.joint_angle2, -jnp.pi, jnp.pi),
            state.velocity_1,
            state.velocity_2,
        ]
    )


def test_matches_gymnasium_acrobot_on_its_discrete_torques():
    env, params = gymnax.make("AcrobotContinuous-v0")
    env_gym = gym.make("Acrobot-v1").unwrapped
    env_gym.reset(seed=0)
    rng = np.random.default_rng(0)

    for _ in range(200):
        # Random states, including ones that leave [-pi, pi) after the step
        s = rng.uniform([-np.pi, -np.pi, -6, -12], [np.pi, np.pi, 6, 12])
        a = rng.integers(3)
        env_gym.state = s.copy()
        _, reward_gym, terminated_gym, _, _ = env_gym.step(a)

        _, next_state, reward, done, _ = env.step_env(
            jax.random.key(0), _state(s), jnp.array([a - 1.0]), params
        )

        np.testing.assert_allclose(_wrapped(next_state), env_gym.state, atol=1e-4)
        assert reward == reward_gym
        assert bool(done) == terminated_gym


def test_matches_gymnax_acrobot_on_its_discrete_torques():
    env, params = gymnax.make("AcrobotContinuous-v0")
    env_discrete, params_discrete = gymnax.make("Acrobot-v1")
    _, state = env.reset(jax.random.key(0), params)

    for a in range(3):
        obs, *_ = env.step_env(jax.random.key(1), state, jnp.array([a - 1.0]), params)
        obs_discrete, *_ = env_discrete.step_env(
            jax.random.key(1), state, a, params_discrete
        )
        np.testing.assert_allclose(obs, obs_discrete, atol=1e-6)


def test_torque_is_clipped_and_scaled():
    env, params = gymnax.make("AcrobotContinuous-v0")
    _, state = env.reset(jax.random.key(0), params)
    step = lambda a, p: env.step_env(jax.random.key(1), state, jnp.array([a]), p)[1]

    clipped, saturated = step(5.0, params), step(1.0, params)
    doubled = step(0.5, params.replace(max_torque=2.0))
    np.testing.assert_allclose(_wrapped(clipped), _wrapped(saturated))
    np.testing.assert_allclose(_wrapped(doubled), _wrapped(saturated), atol=1e-6)


def test_dense_reward_is_the_height_of_the_tip_below_the_line():
    env, params = gymnax.make("AcrobotContinuous-v0")
    params = params.replace(dense_reward=True)
    _, state = env.reset(jax.random.key(0), params)
    _, next_state, reward, _, _ = env.step_env(
        jax.random.key(1), state, jnp.array([0.3]), params
    )
    t1, t2 = next_state.joint_angle1, next_state.joint_angle2
    np.testing.assert_allclose(reward, -np.cos(t1) - np.cos(t1 + t2) - 1.0, atol=1e-6)
    # Hanging down, the tip is 2 below the pivot and 3 below the line
    np.testing.assert_allclose(reward, -3.0, atol=0.1)
