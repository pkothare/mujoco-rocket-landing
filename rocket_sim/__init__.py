"""
Rocket Landing Simulation using MuJoCo.

This package provides a MuJoCo-based simulation of
a rocket first stage for reinforcement learning
landing experiments.

Sub-packages:
    core/   - environment, state, config, utilities
    rl/     - PPO, networks, policies
    tools/  - CLI scripts (train, evaluate, render, dump_episode, plot)
"""

from .core import Config, RocketEnv, RocketState, axis_angle_to_quat, quat_to_euler
from .rl import POLICIES, PPO, SquashedActorCritic, autopilot_policy, random_policy

__all__ = [
    "Config",
    "POLICIES",
    "PPO",
    "RocketEnv",
    "RocketState",
    "SquashedActorCritic",
    "autopilot_policy",
    "axis_angle_to_quat",
    "quat_to_euler",
    "random_policy",
]
