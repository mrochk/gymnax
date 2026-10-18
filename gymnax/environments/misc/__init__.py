"""Miscellaneous environments in JAX.

This module provides JAX implementations of miscellaneous environments, including:
- BernoulliBandit
- GaussianBandit
- FourRooms
- FrozenLake
- MetaMaze
- PointRobot
- PlanarQuadrotor
- Reacher
- Swimmer
- Pong
"""

from .bernoulli_bandit import BernoulliBandit
from .frozen_lake import FrozenLake
from .gaussian_bandit import GaussianBandit
from .meta_maze import MetaMaze
from .planar_quadrotor import PlanarQuadrotor
from .point_robot import PointRobot
from .pong import Pong
from .reacher import Reacher
from .rooms import FourRooms
from .swimmer import Swimmer

__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "FourRooms",
    "FrozenLake",
    "MetaMaze",
    "PointRobot",
    "PlanarQuadrotor",
    "Reacher",
    "Swimmer",
    "Pong",
]
