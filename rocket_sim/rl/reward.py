"""Reward computation for PPO training and evaluation."""

import numpy as np

from rocket_sim.core.config import Config
from rocket_sim.core.env import RocketEnv
from .obs_act import AGENT_DT


def compute_reward(
    state,
    action: np.ndarray,
    info: dict,
    env: RocketEnv,
    cfg: Config,
    prev_state=None,
    ep_flags: dict | None = None,
) -> float:
    """Dense shaped reward (v35 and later).

    Terms:
      1. Potential-based altitude descent shaping.
      2. State quality: tilt, angular velocity, horizontal speed, time.
      2b. Slow-descent penalty (anti-hovering).
      3. Leg deployment latch.
      4. Terminal landing bonus / crash penalty.
    """
    if cfg.debug_log:
        pos = state.position
        vel = state.linear_velocity
        ang = state.angular_velocity
        tilt = RocketEnv._body_tilt(state.orientation)
        print(
            f"DBG step={info.get('step', 0):5d}  "
            f"x={pos[0]:8.2f} y={pos[1]:8.2f} z={pos[2]:8.2f}  "
            f"vx={vel[0]:8.2f} vy={vel[1]:8.2f} vz={vel[2]:8.2f}  "
            f"wx={ang[0]:7.3f} wy={ang[1]:7.3f} wz={ang[2]:7.3f}  "
            f"tilt={tilt:6.3f}"
        )

    if cfg.reward_mode == "sparse":
        r = 0.0
        if info.get("landed", False):
            r += cfg.landing_bonus * env._compute_landing_quality(state)
        if info.get("crashed", False):
            r -= cfg.crash_penalty
        return r

    r = 0.0
    dt = AGENT_DT

    # 1) Potential-based altitude shaping
    if cfg.reward_use_altitude and prev_state is not None:
        alt_delta = float(prev_state.altitude) - float(state.altitude)
        alt_delta = min(alt_delta, 10.0 * dt)  # cap at 10 m/s descent
        if alt_delta < 0:
            # Soft proportional penalty for ascending
            r -= 0.9 * abs(alt_delta)
        else:
            coeff = 0.5 if state.altitude > 25.0 else 0.25
            r += coeff * alt_delta

    # 2) State quality penalty
    tilt = RocketEnv._body_tilt(state.orientation)
    angvel_sum = sum(abs(float(state.angular_velocity[i])) for i in range(3))
    vx, vy = state.linear_velocity[0], state.linear_velocity[1]
    h_speed = np.sqrt(vx**2 + vy**2)
    if cfg.reward_use_tilt:
        r -= 5.0 * tilt * dt
    if cfg.reward_use_angvel:
        r -= cfg.reward_angvel_coeff * min(angvel_sum, 50.0) * dt
    if cfg.reward_use_horizontal:
        r -= 0.5 * h_speed * dt
    r -= cfg.time_penalty * dt

    # 2b) Slow-descent penalty for anti-hovering
    if cfg.reward_use_slow_descent:
        vz_down = -float(state.linear_velocity[2])  # positive when descending
        if state.altitude > 30.0 and vz_down < 3.0:
            r -= 5.0 * (3.0 - vz_down) * dt
        elif state.altitude > 10.0 and state.altitude <= 30.0 and vz_down < 1.0:
            r -= 5.0 * (1.0 - vz_down) * dt

    # 3) Leg deployment one-time bonus
    if ep_flags is None:
        ep_flags = {}
    if cfg.reward_use_legs and state.altitude < cfg.leg_deploy_alt:
        legs_deployed = all(action[i] > 1.2 for i in range(7, 11))
        vz = float(state.linear_velocity[2])
        if legs_deployed and not ep_flags.get("legs_deployed", False) and vz < -0.5:
            r += 10.0
            ep_flags["legs_deployed"] = True
        elif not legs_deployed and ep_flags.get("legs_deployed", False):
            r -= 15.0
            ep_flags["legs_deployed"] = False

    # 4) Terminal rewards
    if info.get("landed", False):
        r += cfg.landing_bonus * env._compute_landing_quality(state)
    if info.get("crashed", False):
        r -= cfg.crash_penalty + 1.0 * state.speed
    return r
