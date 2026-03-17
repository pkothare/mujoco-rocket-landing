"""State dataclass for the rocket landing simulation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RocketState:
    """State representation for the rocket."""

    position: np.ndarray  # x, y, z position (m)
    orientation: np.ndarray  # quaternion (w, x, y, z)
    linear_velocity: np.ndarray  # vx, vy, vz (m/s)
    angular_velocity: np.ndarray  # wx, wy, wz (rad/s)
    euler_angles: np.ndarray  # roll, pitch, yaw (rad)

    @property
    def altitude(self) -> float:
        """Height above ground in meters."""
        return self.position[2]

    @property
    def horizontal_distance(self) -> float:
        """Horizontal distance from landing pad center."""
        return np.sqrt(self.position[0] ** 2 + self.position[1] ** 2)

    @property
    def speed(self) -> float:
        """Total velocity magnitude."""
        return float(np.linalg.norm(self.linear_velocity))

    @property
    def vertical_speed(self) -> float:
        """Vertical velocity (negative = descending)."""
        return self.linear_velocity[2]

    def to_array(self) -> np.ndarray:
        """Flatten state to numpy array for RL."""
        return np.concatenate(
            [
                self.position,
                self.orientation,
                self.linear_velocity,
                self.angular_velocity,
            ]
        )

