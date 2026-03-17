"""Built-in control policies for the rocket landing environment."""

import numpy as np

from rocket_sim.core.env import RocketEnv

# Physics constants for the autopilot
_GRAVITY = 9.81  # m/s^2
_ROCKET_MASS = 540000.0  # kg (from MJCF)
_MAX_THRUST = 10800000.0  # N  (gear on main_thrust actuator)


def autopilot_policy(env: RocketEnv) -> np.ndarray:
    """Physics-based autopilot with body-frame attitude control.

    Key insight: TVC actuators apply torque in the rocket's BODY frame
    (they rotate with the rocket).  So attitude errors and angular velocities
    must be computed in body frame, not world frame.

    Phases:
      1. COAST/STABILISE - low thrust; TVC keeps rocket upright.
      2. SUICIDE BURN - full thrust to arrest descent at the last moment.
      3. TERMINAL - gentle PD hover/touchdown.
    """
    state = env._get_state()
    action = np.zeros(env.action_dim)

    # Effective altitude above pad contact (body center is ~20.6m above pads)
    pad_offset = 20.6  # body-center z when pads touch ground
    alt = max(0.0, state.altitude - pad_offset)
    raw_alt = state.altitude  # for leg deploy check
    vz = state.vertical_speed  # negative = falling
    vx, vy = state.linear_velocity[0], state.linear_velocity[1]
    px, py = state.position[0], state.position[1]

    # Body-frame tilt error
    qw, qx, qy, qz = state.orientation
    up_body_x = 2.0 * (qx * qz + qw * qy)  # R[0,2]
    up_body_y = 2.0 * (qy * qz - qw * qx)  # R[1,2]
    up_body_z = 1.0 - 2.0 * (qx**2 + qy**2)  # R[2,2] = cos(tilt)

    # Tilt angle from vertical
    tilt = np.arccos(np.clip(up_body_z, -1.0, 1.0))

    # Error: how much to rotate about body X/Y to align with vertical
    err_bx = (
        -up_body_y
    )  # ~ sin(roll_body); positive means need negative body-X torque
    # ~ sin(pitch_body); positive -> need negative body-Y torque
    err_by = up_body_x

    # Body-frame angular velocity (qvel[3:6] is body frame for free joints)
    joint_id = env.model.joint("rocket_joint").id
    dof_adr = env.model.jnt_dofadr[joint_id]
    wb_x = env.data.qvel[dof_adr + 3]  # body-frame angular velocity about X
    wb_y = env.data.qvel[dof_adr + 4]  # body-frame angular velocity about Y
    wb_z = env.data.qvel[dof_adr + 5]  # body-frame angular velocity about Z

    # Vertical velocity profile
    a_ref = (
        7.0  # reference deceleration (m/s^2); higher = more conservative margin
    )
    margin = 20.0  # start decelerating to gentle speed at this altitude
    if alt > margin:
        vz_target = -np.sqrt(2.0 * a_ref * (alt - margin)) - 2.0
        vz_target = max(vz_target, -300.0)  # cap at initial velocities
    else:
        # Final 20 m: glide down gently
        vz_target = max(-2.0, -0.3 - alt * 0.08)

    hover_thrust = _ROCKET_MASS * _GRAVITY / _MAX_THRUST  # ~ 0.498
    vz_err = vz - vz_target  # negative -> too fast, need more thrust
    thrust = hover_thrust - 0.10 * vz_err  # stronger gain for tighter tracking

    # SAFE MODE: if tilt is large, stop descent and focus on attitude recovery.
    if tilt > 0.25:  # ~14 deg
        thrust = 1.0
    elif tilt > 0.15 and thrust < 0.35:
        thrust = 0.35

    action[0] = np.clip(thrust, 0.05, 1.0)

    # TVC for attitude control in body frame
    kp = 8.0
    kd = 3.0

    # Lateral velocity braking: tilt intentionally to cancel horizontal speed
    if tilt < 0.5:
        kv = 0.01  # rad per m/s of desired tilt
        # Also include position correction toward pad
        target_vx = (
            np.clip(-px * 0.005, -5, 5) if raw_alt < 500 + pad_offset else 0.0
        )
        target_vy = (
            np.clip(-py * 0.005, -5, 5) if raw_alt < 500 + pad_offset else 0.0
        )
        err_bx -= kv * (vy - target_vy)
        err_by += kv * (vx - target_vx)

    tvc_bx = -kp * err_bx - kd * wb_x  # body-X torque
    tvc_by = -kp * err_by - kd * wb_y  # body-Y torque

    action[1] = np.clip(tvc_bx, -1.0, 1.0)  # tvc_pitch -> body X torque
    action[2] = np.clip(tvc_by, -1.0, 1.0)  # tvc_yaw  -> body Y torque

    # RCS thrusters for body-frame attitude control
    rcs_bx = np.clip(-3.0 * err_bx - 1.0 * wb_x, -1, 1)
    rcs_by = np.clip(-3.0 * err_by - 1.0 * wb_y, -1, 1)
    rcs_yaw = np.clip(-5.0 * wb_z, -1, 1)  # pure yaw damping
    action[11] = rcs_by  # rcs_x_pos -> body Y torque (pitch)
    action[12] = rcs_bx  # rcs_y_pos -> body X torque (roll)
    action[13] = rcs_yaw  # rcs_x_neg -> body Z torque (yaw)
    action[14] = rcs_yaw  # rcs_y_neg -> body Z torque (yaw)

    # Grid fins are not needed for yaw because RCS handles it

    # Deploy legs when low using full 2.4 rad top-down deployment
    if raw_alt < 200 + pad_offset:
        autopilot_policy._legs_deployed = True
    if getattr(autopilot_policy, "_legs_deployed", False):
        action[7:11] = 2.4

    return action


def random_policy(env: RocketEnv) -> np.ndarray:
    """Random actions with moderate thrust - useful as a baseline."""
    action = np.random.uniform(-1, 1, size=env.action_dim)
    action[0] = np.random.uniform(0.3, 0.8)
    return action


POLICIES = {
    "autopilot": autopilot_policy,
    "random": random_policy,
}
