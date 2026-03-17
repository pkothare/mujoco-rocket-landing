"""
Rocket MuJoCo Environment for Reinforcement Learning.

This module provides a Gymnasium-style environment for a rocket
first stage propulsive landing task.
"""

# pyright: reportAttributeAccessIssue=false

from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from .state import RocketState
from .utils import axis_angle_to_quat, quat_to_euler


class RocketEnv:
    """
    MuJoCo-based rocket landing environment.

    This environment simulates the propulsive landing of a
    rocket first stage. The goal is to land the rocket
    vertically on the landing pad with minimal velocity
    and tilt.

    Observation Space:
        - Position (x, y, z): 3D position in meters
        - Orientation (qw, qx, qy, qz): quaternion
        - Linear velocity (vx, vy, vz): m/s
        - Angular velocity (wx, wy, wz): rad/s

    Action Space (15-dim continuous):
        - Main thrust: 0-1 (throttle)
        - TVC pitch: -1 to 1
        - TVC yaw: -1 to 1
        - Grid fins (4): -1 to 1 each
        - Leg deployment (4): 0-2.4 each
        - RCS thrusters (4): -1 to 1 each

    Reward:
        Shaped reward for landing safely with:
        - Minimal horizontal offset
        - Minimal velocity at touchdown
        - Upright orientation
        - Fuel efficiency
    """

    # Physical constants
    GRAVITY = 9.81  # m/s^2

    # Landing success criteria
    MAX_LANDING_SPEED = 4.5  # m/s
    MAX_TILT_ANGLE = 0.2618  # radians (~15 degrees)
    MAX_HORIZONTAL_OFFSET = 2.0  # meters

    # Rocket geometry: body center is ~20.6m above leg pads when upright
    _TOUCHDOWN_BODY_Z = 20.6  # expected body-center z at touchdown
    _LANDING_CHECK_ALT = _TOUCHDOWN_BODY_Z + 1.0  # trigger zone

    def __init__(
        self,
        xml_path: str | None = None,
        render_mode: str = "human",
        max_steps: int = 3000,
        random_init: bool = True,
    ):
        """
        Initialize the Rocket landing environment.

        Args:
            xml_path: Path to MJCF XML file. Defaults to
                scene.xml in same directory.
            render_mode: "human" for visualization,
                "rgb_array" for pixel observations
            max_steps: Maximum simulation steps per episode
            random_init: Randomize initial conditions
        """
        if xml_path is None:
            xml_path = str(Path(__file__).parent.parent / "scene.xml")

        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {self.xml_path}")

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.random_init = random_init

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Viewer for rendering
        self.viewer = None
        self._renderer = None
        self._step_count = 0

        # Cache body and joint indices
        self._rocket_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "rocket"
        )

        # Cache geom IDs for contact detection
        self._ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        self._leg_pad_geom_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"leg_{i}_pad"
            )
            for i in range(1, 5)
        ]

        # Cache freejoint qpos/qvel addresses for fast ground-clamping
        _jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "rocket_joint"
        )
        self._qpos_addr = self.model.jnt_qposadr[_jnt_id]  # x,y,z,qw,qx,qy,qz
        self._qvel_addr = self.model.jnt_dofadr[_jnt_id]  # vx,vy,vz,wx,wy,wz

        # Control indices
        self.n_actuators = self.model.nu

        # Observation and action dimensions
        self.observation_dim = 13  # pos(3) + quat(4) + linvel(3) + angvel(3)
        self.action_dim = self.n_actuators

    def reset(
        self,
        seed: int | None = None,
        initial_state: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to initial conditions.

        Args:
            seed: Random seed for reproducibility
            initial_state: Optional dict with 'position',
                'velocity', 'orientation'

        Returns:
            observation: Initial state observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Set initial conditions
        if initial_state is not None:
            self._set_state(initial_state)
        elif self.random_init:
            self._randomize_initial_state()
        else:
            self._set_default_initial_state()

        self._step_count = 0

        # Forward dynamics to initialize
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one simulation step.

        Args:
            action: Control inputs (thrust, TVC, grid fins, legs)

        Returns:
            observation: New state
            reward: Step reward
            terminated: Episode ended (landed or crashed)
            truncated: Episode truncated (max steps)
            info: Additional information
        """
        # Clip actions to each actuator's own control range
        for i in range(self.n_actuators):
            lo, hi = self.model.actuator_ctrlrange[i]
            action[i] = np.clip(action[i], lo, hi)
        self.data.ctrl[:] = action[: self.n_actuators]

        # Step simulation (8 substeps x 0.002s = 0.016s ~ 1/60s per action)
        n_substeps = 8
        z_idx = self._qpos_addr + 2  # z-position index in qpos
        vel_start = self._qvel_addr  # start of 6-dof velocity block
        ground_contact_during_substep = False
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
            # Clamp rocket to ground - prevent penetration and bouncing.
            if self.data.qpos[z_idx] < 0.0:
                self.data.qpos[z_idx] = 0.0
                self.data.qvel[vel_start : vel_start + 6] = 0.0
                ground_contact_during_substep = True
                break  # stop substeps - rocket is on the ground
        self._step_count += 1

        # Get new state
        obs = self._get_observation()
        state = self._get_state()

        # Reward is always overridden by ppo.compute_reward() in
        reward = 0.0

        # Check termination - also flag crash if ground penetration was
        terminated = self._check_terminated(state)
        truncated = self._step_count >= self.max_steps

        info = self._get_info()
        info["landed"] = self._check_landed(state)
        if ground_contact_during_substep and not info["landed"]:
            info["crashed"] = True
            terminated = True
        else:
            info["crashed"] = self._check_crashed(state)

        return obs, reward, terminated, truncated, info

    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera_elevation: float = -20.0,
        camera_azimuth: float = 135.0,
    ) -> np.ndarray | None:
        """Render the current state.

        Args:
            width: Image width in pixels (rgb_array mode only).
            height: Image height in pixels (rgb_array mode only).
            camera_elevation: Camera elevation angle (degrees).
            camera_azimuth: Camera azimuth angle (degrees).

        Returns:
            RGB numpy array of shape (height, width, 3) when render_mode is
            "rgb_array", otherwise None.
        """
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()
            return None
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=height, width=width
                )

            # Adaptive camera distance: zoom in as rocket approaches ground
            state = self._get_state()
            alt = state.altitude
            # Rocket is 42m tall; at ground: 80m away, at altitude: up to 200m
            camera_distance = np.clip(
                80.0 + 20.0 * np.log1p(alt / 100.0), 80.0, 200.0
            )

            # Adaptive extent: shrink when near ground so shadow map
            if alt < 100.0:
                adaptive_extent = float(
                    np.clip(5.0 + (alt / 100.0) * 195.0, 5.0, 200.0)
                )
            else:
                adaptive_extent = 200.0
            self.model.stat.extent = adaptive_extent

            # Set up a tracking camera centred on the rocket
            scene_option = mujoco.MjvOption()
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = self._rocket_body_id
            camera.distance = camera_distance
            camera.elevation = camera_elevation
            camera.azimuth = camera_azimuth
            camera.lookat[:] = [
                0,
                0,
                0,
            ]  # offset from tracked body (0 = centred on it)

            self._renderer.update_scene(
                self.data, camera=camera, scene_option=scene_option
            )
            # Shadows on, haze off (desert skybox provides depth cues)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
            return self._renderer.render()
        return None

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # Internal helpers

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        state = self._get_state()
        return state.to_array()

    def _get_state(self) -> RocketState:
        """Get current rocket state."""
        # Get body state from MuJoCo
        pos = self.data.body(self._rocket_body_id).xpos.copy()
        quat = self.data.body(self._rocket_body_id).xquat.copy()

        # Velocities from qvel (free joint DOFs).
        qvel = self.data.qvel[self._qvel_addr : self._qvel_addr + 6]
        linvel = qvel[:3].copy()
        angvel = qvel[3:6].copy()

        # Convert quaternion to Euler angles
        euler = quat_to_euler(quat)

        return RocketState(
            position=pos,
            orientation=quat,
            linear_velocity=linvel,
            angular_velocity=angvel,
            euler_angles=euler,
        )

    def _set_state(self, state: dict[str, np.ndarray]):
        """Set rocket state from dictionary."""
        # Find the freejoint qpos index
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "rocket_joint"
        )
        qpos_addr = self.model.jnt_qposadr[joint_id]
        qvel_addr = self.model.jnt_dofadr[joint_id]

        if "position" in state:
            self.data.qpos[qpos_addr : qpos_addr + 3] = state["position"]
        if "orientation" in state:
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = state[
                "orientation"
            ]
        if "linear_velocity" in state:
            self.data.qvel[qvel_addr : qvel_addr + 3] = state[
                "linear_velocity"
            ]
        if "angular_velocity" in state:
            self.data.qvel[qvel_addr + 3 : qvel_addr + 6] = state[
                "angular_velocity"
            ]

    def _set_default_initial_state(self):
        """Set default initial state for landing scenario.

        Starts at 5 km altitude with a high downward velocity to emulate
        the entry/descent phase where guidance must kick in.
        """
        self._set_state(
            {
                "position": np.array([0.0, 0.0, 5000.0]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
                "linear_velocity": np.array(
                    [0.0, 0.0, -200.0]
                ),  # 200 m/s downward
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            }
        )

    def _randomize_initial_state(self):
        """Randomize initial conditions for diverse training."""
        # Position directly above landing pad at 5km
        pos = np.array(
            [
                0.0,  # x - centered above pad
                0.0,  # y - centered above pad
                5000.0,  # z (altitude)
            ]
        )

        # Random tilt (quaternion) - fully random orientation
        tilt_angle = np.random.uniform(
            0, np.pi
        )  # 0-180 degrees, any orientation
        tilt_axis = np.random.randn(3)
        tilt_axis /= np.linalg.norm(tilt_axis)
        quat = axis_angle_to_quat(tilt_axis, tilt_angle)

        # High downward velocity with small horizontal drift
        vz = np.random.uniform(-300.0, -100.0)  # 100-300 m/s downward
        vx = np.random.uniform(-20.0, 20.0)  # horizontal drift
        vy = np.random.uniform(-20.0, 20.0)
        linvel = np.array([vx, vy, vz])

        # Small random angular velocity
        angvel = np.random.uniform(-0.1, 0.1, size=3)

        self._set_state(
            {
                "position": pos,
                "orientation": quat,
                "linear_velocity": linvel,
                "angular_velocity": angvel,
            }
        )

    def _compute_landing_quality(self, state: RocketState) -> float:
        """Compute landing quality score (0-1)."""
        # Speed factor (0 = perfect, 1 = max acceptable)
        speed_factor = min(1.0, state.speed / self.MAX_LANDING_SPEED)

        # Tilt factor
        tilt = self._body_tilt(state.orientation)
        tilt_factor = min(1.0, tilt / self.MAX_TILT_ANGLE)

        # Position factor
        pos_factor = min(
            1.0, state.horizontal_distance / self.MAX_HORIZONTAL_OFFSET
        )

        # Quality is 1 minus weighted average of penalty factors
        quality = (
            1.0 - 0.4 * speed_factor - 0.3 * tilt_factor - 0.3 * pos_factor
        )
        return max(0.0, quality)

    def _check_terminated(self, state: RocketState) -> bool:
        """Check if episode should terminate."""
        return self._check_landed(state) or self._check_crashed(state)

    @staticmethod
    def _body_tilt(orientation: np.ndarray) -> float:
        """Tilt angle (rad) between body Z and world Z."""
        qw, qx, qy, qz = orientation
        cos_tilt = 1.0 - 2.0 * (qx**2 + qy**2)
        return np.arccos(np.clip(cos_tilt, -1.0, 1.0))

    def _has_ground_contact(self) -> bool:
        """Check if any leg pad is in contact with ground."""
        contact_geoms = set(self._leg_pad_geom_ids)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._ground_geom_id and g2 in contact_geoms:
                return True
            if g2 == self._ground_geom_id and g1 in contact_geoms:
                return True
        return False

    def _has_body_ground_contact(self) -> bool:
        """Check if the rocket body (not legs) is in contact with ground."""
        body_geoms = set()
        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            if (
                body_id == self._rocket_body_id
                and i not in self._leg_pad_geom_ids
            ):
                if self.model.geom_contype[i] > 0:
                    body_geoms.add(i)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._ground_geom_id and g2 in body_geoms:
                return True
            if g2 == self._ground_geom_id and g1 in body_geoms:
                return True
        return False

    def _check_landed(self, state: RocketState) -> bool:
        """Check for successful landing via actual ground contact."""
        if self._has_ground_contact():
            if state.speed < self.MAX_LANDING_SPEED:
                tilt = self._body_tilt(state.orientation)
                if tilt < self.MAX_TILT_ANGLE:
                    return True
        return False

    def _check_crashed(self, state: RocketState) -> bool:
        """Check for crash conditions.

        Any ground contact that is not a successful landing is a crash.
        This closes the gap where speed/tilt combos between the landing
        and old crash thresholds would leave the episode hanging.
        """
        # Body hit the ground (not on legs)
        if self._has_body_ground_contact():
            return True
        # Any leg contact that doesn't qualify as landed is a crash
        if self._has_ground_contact() and not self._check_landed(state):
            return True
        # Below ground
        if state.altitude < 0:
            return True
        return False

    def _get_info(self) -> dict[str, Any]:
        """Get additional info dictionary."""
        state = self._get_state()
        return {
            "step": self._step_count,
            "altitude": state.altitude,
            "horizontal_distance": state.horizontal_distance,
            "speed": state.speed,
            "vertical_speed": state.vertical_speed,
        }
