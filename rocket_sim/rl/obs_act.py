"""Observation / action helpers for ablations."""

import numpy as np

from rocket_sim.core.config import Config

# Agent-level timestep: 8 substeps x 0.002s = 0.016s per action.
AGENT_DT = 0.016


def get_action_indices(action_mode: str) -> list[int]:
    """Return actuator indices active under *action_mode*.

    Canonical actuator order (from rocket.xml):
      0: main_thrust   1: tvc_pitch   2: tvc_yaw
      3-6: grid_fins    7-10: legs     11-14: RCS
    """
    if action_mode == "full":
        return list(range(15))
    if action_mode == "no_gridfins":
        return [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14]
    if action_mode == "no_rcs":
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if action_mode == "no_gridfins_rcs":
        return [0, 1, 2, 7, 8, 9, 10]
    raise ValueError(f"Unknown action_mode: {action_mode}")


def get_obs_dim(cfg: Config) -> int:
    base = 10 if cfg.obs_mode == "no_angvel" else 13
    if cfg.obs_augment:
        base += 2  # vz_error + tilt
    return base * cfg.obs_history_len


def get_act_dim(cfg: Config) -> int:
    if cfg.action_mode == "no_gridfins":
        return 11
    if cfg.action_mode == "no_rcs":
        return 11
    if cfg.action_mode == "no_gridfins_rcs":
        return 7
    return 15


def process_obs(
    obs: np.ndarray, cfg: Config, prev_obs: np.ndarray | None = None
) -> np.ndarray:
    """Apply observation ablation."""
    if cfg.obs_mode == "no_angvel":
        return obs[:10].copy()
    if cfg.obs_mode == "noisy":
        return obs + np.random.normal(
            0, cfg.obs_noise_std, size=obs.shape
        ).astype(np.float32)
    if cfg.obs_mode == "delayed":
        return prev_obs.copy() if prev_obs is not None else obs.copy()
    return obs.copy()


def expand_action(policy_action: np.ndarray, cfg: Config) -> np.ndarray:
    """Map reduced-dim policy output to full 15-dim env action.

    Uses get_action_indices to scatter policy outputs into the
    correct actuator slots.
    """
    indices = get_action_indices(cfg.action_mode)
    full = np.zeros(15, dtype=np.float32)
    for i, idx in enumerate(indices):
        full[idx] = policy_action[i]
    # Legs are a single binary actuator: average the 4 leg outputs,
    leg_mean = np.mean(full[7:11])
    leg_val = 2.4 if leg_mean > 1.2 else 0.0
    full[7:11] = leg_val
    return full
