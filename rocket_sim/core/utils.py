"""Utility functions for the rocket landing simulation."""

import numpy as np


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle to quaternion (w,x,y,z)."""
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def autopilot_vz_target(alt: float) -> float:
    """Compute the autopilot's ideal vertical speed for a given altitude.

    Mirrors the suicide-burn profile from the PD autopilot:
      alt > 20 m:  vz_target = -sqrt(2 * a_ref * (alt - margin)) - 2
      alt <= 20 m:  vz_target = max(-2, -0.3 - alt * 0.08)
    """
    a_ref = 7.0
    margin = 20.0
    pad_offset = 20.6
    a = max(0.0, alt - pad_offset)
    if a > margin:
        vz = -np.sqrt(2.0 * a_ref * (a - margin)) - 2.0
        return max(vz, -300.0)
    return max(-2.0, -0.3 - a * 0.08)

