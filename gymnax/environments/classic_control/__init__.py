"""Classic control environments in JAX.

This module provides JAX implementations of classic control environments
from OpenAI Gym, including:
- Acrobot
- ContinuousAcrobot
- CartPole
- ContinuousCartPole
- CartPoleSwingUp
- MountainCar
- ContinuousMountainCar
- Pendulum
"""

from .acrobot import Acrobot
from .cartpole import CartPole
from .continuous_acrobot import ContinuousAcrobot
from .continuous_cartpole import CartPoleSwingUp, ContinuousCartPole
from .continuous_mountain_car import ContinuousMountainCar
from .mountain_car import MountainCar
from .pendulum import Pendulum

__all__ = [
    "Acrobot",
    "ContinuousAcrobot",
    "CartPole",
    "ContinuousCartPole",
    "CartPoleSwingUp",
    "MountainCar",
    "ContinuousMountainCar",
    "Pendulum",
]
