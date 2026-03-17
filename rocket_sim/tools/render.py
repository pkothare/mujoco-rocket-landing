"""
Render a full rocket landing episode to an MP4 video.

Uses the autopilot PD controller (or a user-supplied policy) to fly an episode
while capturing RGB frames via MuJoCo offscreen rendering, then stitches them
into a video with moviepy.

Usage:
    uv run python render.py                        # autopilot, default output
    uv run python render.py -o my_landing.mp4      # custom output path
    uv run python render.py --seed 42 --fps 30     # reproducible, 30 fps
    uv run python render.py --num_episodes 5       # render 5 episodes
"""

import argparse
import os
import pathlib

# Use EGL for headless GPU-accelerated offscreen rendering
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from moviepy import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

from rocket_sim.core.config import Config
from rocket_sim.core.env import RocketEnv
from rocket_sim.rl.networks import SquashedActorCritic
from rocket_sim.rl.policies import POLICIES
from rocket_sim.core.utils import autopilot_vz_target
from rocket_sim.rl.curriculum import curriculum_initial_state
from rocket_sim.rl.normalization import RunningMeanStd
from rocket_sim.rl.obs_act import (
    expand_action,
    get_action_indices,
    get_obs_dim,
    process_obs,
)

# Telemetry overlay


def _stamp_telemetry(frame: np.ndarray, env: RocketEnv) -> np.ndarray:
    """Burn telemetry text onto the top-right corner."""
    state = env._get_state()
    tilt = np.degrees(RocketEnv._body_tilt(state.orientation))
    # Rocket long axis is Z: euler[0]=X rot=pitch, euler[1]=Y rot=yaw, euler[2]=Z rot=roll
    lines = [
        f"ALT  {state.altitude:7.1f} m",
        f"VEL  {state.speed:7.1f} m/s",
        f"Vz   {state.vertical_speed:7.1f} m/s",
        f"TILT {tilt:+6.1f} deg",
        f"PTCH {np.degrees(state.euler_angles[0]):+6.1f} deg",
        f"YAW  {np.degrees(state.euler_angles[1]):+6.1f} deg",
        f"ROLL {np.degrees(state.euler_angles[2]):+6.1f} deg",
    ]

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default(size=11)

    # Measure text block width for right-alignment
    line_height = 13
    text_block = "\n".join(lines)
    bbox = draw.textbbox((0, 0), text_block, font=font)
    text_w = bbox[2] - bbox[0]

    margin = 6
    x = img.width - text_w - margin
    y = margin

    # Semi-transparent background
    bg_h = line_height * len(lines) + margin
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [x - margin, y - 2, x + text_w + margin, y + bg_h],
        fill=(0, 0, 0, 140),
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height), line, fill=(0, 255, 0), font=font)

    return np.array(img)


# Core rendering logic


def render_episode(
    env: RocketEnv,
    policy,
    seed: int = 0,
    max_steps: int = 3000,
    render_kwargs: dict | None = None,
    initial_state: dict[str, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[float], dict]:
    """Run one episode, collect RGB frames and rewards.

    Args:
        env: A RocketEnv with render_mode="rgb_array".
        policy: Callable (env) -> action array.
        seed: Random seed for reset.
        max_steps: Hard cap on number of simulation steps.
        render_kwargs: Extra kwargs forwarded to env.render().
        initial_state: If provided, use these initial conditions.

    Returns:
        (frames, rewards, final_info)
    """
    render_kwargs = render_kwargs or {}
    obs, info = env.reset(seed=seed, initial_state=initial_state)

    frames: list[np.ndarray] = []
    rewards: list[float] = []

    # Capture the initial frame
    frame = env.render(**render_kwargs)
    if frame is not None:
        frames.append(_stamp_telemetry(frame, env))

    for _ in range(max_steps):
        action = policy(env)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        frame = env.render(**render_kwargs)
        if frame is not None:
            frames.append(_stamp_telemetry(frame, env))

        if terminated or truncated:
            break

    # Continue rendering for ~2 seconds after episode ends so the viewer
    post_steps = int(2.0 / 0.016)  # 0.016s per env.step -> 125 steps
    zero_action = np.zeros(env.n_actuators, dtype=np.float32)
    landed = info.get("landed", False)
    crashed = info.get("crashed", False)
    if landed or crashed:
        zero_action[7:11] = 2.4  # keep legs deployed
        # Snapshot leg joint positions to force them to stay deployed
        _leg_qpos_indices = [
            env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"leg_{i}_deploy")]
            for i in range(1, 5)
        ]
        _frozen_leg_qpos = [env.data.qpos[idx] for idx in _leg_qpos_indices]
    if landed or crashed:
        # Freeze body pose so it doesn't tip over post-episode.
        _body_qpos_slice = slice(env._qpos_addr, env._qpos_addr + 7)
        _body_qvel_slice = slice(env._qvel_addr, env._qvel_addr + 6)
        _frozen_qpos = env.data.qpos[_body_qpos_slice].copy()
        _frozen_qvel = np.zeros(6)
    for _ in range(post_steps):
        obs, _, _, _, _ = env.step(zero_action)
        if landed or crashed:
            env.data.qpos[_body_qpos_slice] = _frozen_qpos
            env.data.qvel[_body_qvel_slice] = _frozen_qvel
        if landed or crashed:
            # Force leg joints to stay at their deployed positions
            for idx, val in zip(_leg_qpos_indices, _frozen_leg_qpos):
                env.data.qpos[idx] = val
        frame = env.render(**render_kwargs)
        if frame is not None:
            frames.append(_stamp_telemetry(frame, env))

    return frames, rewards, info


def save_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 60,
) -> None:
    """Write a list of RGB frames to an MP4 file."""
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path)


def build_ppo_policy(ckpt_path: str, deterministic: bool = True):
    """Build a callable PPO policy from a checkpoint and config."""
    ckpt_file = pathlib.Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    cfg_path = ckpt_file.parent / "config.json"
    if not cfg_path.exists():
        cfg_path = ckpt_file.parent.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            "Could not locate config.json next to checkpoint"
        )

    cfg = Config.load(str(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer obs/act dims from checkpoint weights to handle old architectures
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)
    obs_dim = ckpt["policy"]["actor_mean.0.weight"].shape[1]
    act_dim = ckpt["policy"]["actor_mean.4.weight"].shape[0]

    model = SquashedActorCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    model.load_state_dict(ckpt["policy"])
    model.eval()

    obs_rms = None
    if cfg.normalize_obs and "obs_rms" in ckpt:
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.load_state_dict(ckpt["obs_rms"])

    # Cache actuator ranges for rescaling [-1, 1] -> env
    xml_path = str(pathlib.Path(__file__).parent.parent / "scene.xml")
    import mujoco as _mj

    _tmp_model = _mj.MjModel.from_xml_path(xml_path)
    full_low = _tmp_model.actuator_ctrlrange[:, 0].copy()
    full_high = _tmp_model.actuator_ctrlrange[:, 1].copy()
    act_indices = get_action_indices(cfg.action_mode)
    if act_dim == 12:
        # Old collapsed-leg format: [0-6, 7(single leg), 11-14]
        old_indices = list(range(7)) + [7] + list(range(11, 15))
        action_low = full_low[old_indices]
        action_high = full_high[old_indices]
    else:
        action_low = full_low[act_indices[:act_dim]]
        action_high = full_high[act_indices[:act_dim]]
    del _tmp_model

    # Handle frame stacking for old checkpoints
    code_base_dim = get_obs_dim(cfg)
    use_stacking = obs_dim > code_base_dim
    if use_stacking:
        from collections import deque
        history_len = obs_dim // code_base_dim if obs_dim % code_base_dim == 0 else getattr(cfg, 'obs_history_len', 5)
        ckpt_base_dim = obs_dim // history_len
        obs_history: deque[np.ndarray] = deque(maxlen=history_len)
        zero_frame = np.zeros(ckpt_base_dim, dtype=np.float32)
        for _ in range(history_len):
            obs_history.append(zero_frame)

    def _augment_obs(single: np.ndarray, env: RocketEnv) -> np.ndarray:
        """Add vz_error + tilt if checkpoint expects augmented obs."""
        if not use_stacking or len(single) >= ckpt_base_dim:
            return single
        state = env._get_state()
        vz_error = state.vertical_speed - autopilot_vz_target(state.altitude)
        tilt = RocketEnv._body_tilt(state.orientation)
        return np.concatenate([single, [vz_error, tilt]]).astype(np.float32)

    def _policy(
        env: RocketEnv,
        obs: np.ndarray,
        prev_obs: np.ndarray | None,
    ) -> np.ndarray:
        single = process_obs(obs, cfg, prev_obs)
        if use_stacking:
            single = _augment_obs(single, env)
            obs_history.append(single)
            final = np.concatenate(list(obs_history)).astype(np.float32)
        else:
            final = single
        if obs_rms is not None:
            final = obs_rms.normalize(final).astype(np.float32)
        obs_t = (
            torch.as_tensor(final, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            if deterministic:
                action = torch.tanh(model.actor_mean(obs_t))
            else:
                action, _, _ = model.act(obs_t)
        action_np = action.cpu().numpy().flatten()
        # Rescale from [-1, 1] to actuator ranges
        rescaled = action_low + (action_np + 1.0) * 0.5 * (
            action_high - action_low
        )
        # Expand to full 15-dim env action using the same function as training
        if act_dim == 12:
            # Old collapsed-leg format: [0-6, 7(single leg), 11-14]
            full_action = np.zeros(15, dtype=np.float32)
            for i in range(7):
                full_action[i] = rescaled[i]
            # Single leg action -> deploy all 4 legs
            leg_val = 2.4 if rescaled[7] > 1.2 else 0.0
            full_action[7:11] = leg_val
            for i in range(8, 12):
                full_action[i + 3] = rescaled[i]  # RCS: 8->11, 9->12, 10->13, 11->14
        else:
            full_action = expand_action(rescaled, cfg)
        return full_action

    if use_stacking:
        def _reset_history():
            obs_history.clear()
            for _ in range(history_len):
                obs_history.append(zero_frame)
        _policy.reset_history = _reset_history  # type: ignore[attr-defined]

    return _policy, cfg


# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Render rocket landing episodes to MP4 video."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output video path (default: results/landing_ep{N}.mp4)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to render (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting random seed (incremented per episode)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3000,
        help="Maximum simulation steps per episode",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Video framerate (default: 60)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (default: 480)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="autopilot",
        choices=list(POLICIES.keys()),
        help="Which built-in policy to use (default: autopilot)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional PPO checkpoint (.pt). If provided, render policy"
            " from checkpoint instead of built-in policy"
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from PPO policy when using --checkpoint",
    )
    parser.add_argument(
        "--curriculum_stage",
        type=int,
        default=None,
        help="Override curriculum stage (default: use checkpoint's stage)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    # Build environment in offscreen mode
    xml_path = str(pathlib.Path(__file__).parent.parent / "scene.xml")
    curriculum_stage = None
    if args.checkpoint:
        ppo_policy, cfg = build_ppo_policy(
            args.checkpoint, deterministic=not args.stochastic
        )
        max_steps = args.max_steps if args.max_steps != 3000 else cfg.max_steps
        env = RocketEnv(
            xml_path=xml_path,
            render_mode="rgb_array",
            random_init=False,
            max_steps=max_steps,
        )
        # Load curriculum stage from checkpoint
        if cfg.use_curriculum:
            ckpt = torch.load(
                args.checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            curriculum_stage = ckpt.get("curriculum_stage", 0)
            if args.curriculum_stage is not None:
                curriculum_stage = args.curriculum_stage
            print(f"  Using curriculum stage {curriculum_stage}/{cfg.curriculum_stages - 1}")
    else:
        max_steps = args.max_steps
        env = RocketEnv(
            xml_path=xml_path,
            render_mode="rgb_array",
            random_init=True,
            max_steps=max_steps,
        )

    render_kwargs = {"width": args.width, "height": args.height}

    for ep in range(args.num_episodes):
        seed = args.seed + ep
        print(f"Episode {ep + 1}/{args.num_episodes}  (seed={seed}) ...")

        prev_obs: np.ndarray | None = None

        def policy_fn(e: RocketEnv) -> np.ndarray:
            nonlocal prev_obs
            if args.checkpoint:
                current_obs = e._get_observation()
                action = ppo_policy(e, current_obs, prev_obs)
                prev_obs = current_obs.copy()
                return action
            return POLICIES[args.policy](e)

        # Reset observation history for each episode
        if args.checkpoint and hasattr(ppo_policy, "reset_history"):
            ppo_policy.reset_history()

        # Generate curriculum initial conditions if applicable
        ic = None
        if curriculum_stage is not None:
            difficulty = curriculum_stage / max(cfg.curriculum_stages - 1, 1)
            np.random.seed(seed)
            ic = curriculum_initial_state(difficulty)
            alt = ic["position"][2]
            tilt_deg = np.degrees(np.arccos(np.clip(2 * ic["orientation"][0]**2 - 1, -1, 1)))
            vz = ic["linear_velocity"][2]
            print(f"  IC: alt={alt:.0f}m  tilt={tilt_deg:.1f} deg  vz={vz:.1f}m/s")

        frames, rewards, info = render_episode(
            env,
            policy_fn,
            seed=seed,
            max_steps=max_steps,
            render_kwargs=render_kwargs,
            initial_state=ic,
        )

        total_reward = sum(rewards)
        landed = info.get("landed", False)
        crashed = info.get("crashed", False)
        status = "LANDED" if landed else ("CRASHED" if crashed else "TIMEOUT")

        print(
            f"  {status} | steps={len(rewards)} | reward={total_reward:.1f} | "
            f"alt={info['altitude']:.1f}m | speed={info['speed']:.1f}m/s"
        )

        # Determine output path
        if args.output and args.num_episodes == 1:
            out_path = args.output
        else:
            out_path = str(output_dir / f"landing_ep{ep}.mp4")

        print(f"  Writing {len(frames)} frames to {out_path} ...")
        save_video(frames, out_path, fps=args.fps)
        print(f"  Saved: {out_path}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
