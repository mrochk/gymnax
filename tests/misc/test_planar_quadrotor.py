"""Tests for the planar quadrotor, against its equations of motion."""

import jax
import jax.numpy as jnp
import numpy as np

import gymnax
from gymnax.environments.misc.planar_quadrotor import EnvState


def _state(x=0.0, z=0.0, theta=0.0, x_dot=0.0, z_dot=0.0, theta_dot=0.0):
    f = lambda v: jnp.array(v, dtype=jnp.float32)
    return EnvState(
        x=f(x), z=f(z), theta=f(theta),
        x_dot=f(x_dot), z_dot=f(z_dot), theta_dot=f(theta_dot), time=0,
    )


def _accelerations(state, action, params):
    """Accelerations of the step, recovered from the change of velocity."""
    _, next_state, *_ = gymnax.make("PlanarQuadrotor-misc")[0].step_env(
        jax.random.key(0), state, jnp.array(action), params
    )
    return np.array(
        [
            next_state.x_dot - state.x_dot,
            next_state.z_dot - state.z_dot,
            next_state.theta_dot - state.theta_dot,
        ]
    ) / params.dt, next_state


def test_hovers_at_rest_with_zero_action():
    _, params = gymnax.make("PlanarQuadrotor-misc")
    acc, next_state = _accelerations(_state(x=0.3, z=-0.2), [0.0, 0.0], params)
    np.testing.assert_allclose(acc, 0.0, atol=1e-5)
    np.testing.assert_allclose([next_state.x, next_state.z], [0.3, -0.2], atol=1e-6)


def test_falls_freely_without_thrust():
    _, params = gymnax.make("PlanarQuadrotor-misc")
    acc, _ = _accelerations(_state(theta=0.7), [-1.0, -1.0], params)
    np.testing.assert_allclose(acc, [0.0, -params.gravity, 0.0], atol=1e-5)


def test_matches_equations_of_motion():
    _, params = gymnax.make("PlanarQuadrotor-misc")
    rng = np.random.default_rng(0)
    m, g = params.mass, params.gravity

    for _ in range(20):
        theta = rng.uniform(-np.pi, np.pi)
        action = rng.uniform(-1, 1, 2)
        u1, u2 = m * g / 2 * (1 + action)

        acc, _ = _accelerations(_state(theta=theta), action, params)
        expected = [
            -(u1 + u2) * np.sin(theta) / m,
            (u1 + u2) * np.cos(theta) / m - g,
            params.arm_length * (u1 - u2) / params.inertia,
        ]
        np.testing.assert_allclose(acc, expected, rtol=1e-4, atol=1e-4)


def test_integrates_with_semi_implicit_euler():
    _, params = gymnax.make("PlanarQuadrotor-misc")
    state = _state(x=0.1, z=0.2, theta=0.3, x_dot=0.4, z_dot=-0.5, theta_dot=0.6)
    _, next_state = _accelerations(state, [0.2, -0.3], params)

    # The positions move with the updated velocities
    for p, v in (("x", "x_dot"), ("z", "z_dot"), ("theta", "theta_dot")):
        expected = getattr(state, p) + params.dt * getattr(next_state, v)
        np.testing.assert_allclose(getattr(next_state, p), expected, atol=1e-6)


def test_reward_is_highest_at_the_origin():
    env, params = gymnax.make("PlanarQuadrotor-misc")
    reward = lambda s, a: env.step_env(jax.random.key(0), s, jnp.array(a), params)[2]
    np.testing.assert_allclose(reward(_state(), [0.0, 0.0]), 1.0)
    assert reward(_state(x=1.0), [0.0, 0.0]) < reward(_state(), [0.0, 0.0])
    assert reward(_state(), [1.0, 1.0]) < reward(_state(), [0.0, 0.0])
