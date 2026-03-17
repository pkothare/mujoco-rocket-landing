"""Curriculum schedule for progressive difficulty during training."""

import numpy as np

from rocket_sim.core.utils import axis_angle_to_quat


class CurriculumSchedule:
    """Gradually increase initial-condition difficulty."""

    def __init__(
        self,
        num_stages: int = 5,
        advance_threshold: float = 0.3,
        window: int = 50,
        retreat_threshold: float = 0.05,
    ):
        self.num_stages = num_stages
        self.advance_threshold = advance_threshold
        self.retreat_threshold = retreat_threshold
        self.stage = 0
        self._window = window
        self._history: list[float] = []

    @property
    def difficulty(self) -> float:
        return self.stage / max(self.num_stages - 1, 1)

    @property
    def rolling_success(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean(self._history[-self._window :]))

    def update(self, successes: list[float]) -> None:
        """Append per-episode success values and maybe advance/retreat."""
        self._history.extend(successes)
        rate = self.rolling_success
        if rate >= self.advance_threshold and self.stage < self.num_stages - 1:
            self.stage += 1
            self._history.clear()  # reset window after promotion
        elif (
            rate < self.retreat_threshold
            and self.stage > 0
            and len(self._history) >= self._window
        ):
            self.stage -= 1
            self._history.clear()


def curriculum_initial_state(difficulty: float) -> dict[str, np.ndarray]:
    """Sample initial conditions scaled by difficulty in [0, 1].

    40 stages, all parameters linear in difficulty:
      Stage  0 (d=0.000):    25m,   2 deg tilt, 0.02 rad/s, vz~  -2 m/s
      Stage 10 (d=0.256):  1298m,  48 deg tilt, 0.66 rad/s, vz~ -78 m/s
      Stage 20 (d=0.513):  2577m,  93 deg tilt, 1.29 rad/s, vz~-155 m/s
      Stage 30 (d=0.769):  3853m, 139 deg tilt, 1.93 rad/s, vz~-231 m/s
      Stage 39 (d=1.000):  5000m, 180 deg tilt, 2.50 rad/s, vz~-300 m/s

    Per-stage increment: ~128m altitude, ~4.6 deg tilt, ~0.064 rad/s angvel.
    """
    alt = 25.0 + difficulty * 4975.0
    max_tilt = np.radians(2.0 + difficulty * 178.0)  # 2 deg - 180 deg
    tilt = np.random.uniform(0, max_tilt)
    vz = np.random.uniform(
        -2.0 - difficulty * 298.0, -2.0 - difficulty * 98.0
    )
    hdrift = difficulty * 20.0
    vx = np.random.uniform(-hdrift, hdrift)
    vy = np.random.uniform(-hdrift, hdrift)
    angvel_max = 0.02 + difficulty * 2.48  # 0.02 - 2.5 rad/s
    angvel = np.random.uniform(-angvel_max, angvel_max, size=3)

    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    quat = axis_angle_to_quat(axis, tilt)
    return {
        "position": np.array([0.0, 0.0, alt]),
        "orientation": quat,
        "linear_velocity": np.array([vx, vy, vz]),
        "angular_velocity": angvel,
    }
