"""Run one episode with a trained PPO checkpoint and dump per-step diagnostics.

Usage:
    uv run python -m rocket_sim.dump_episode \
        results/ppo_v16_dtaware/checkpoints/model_409600.pt
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from rocket_sim.core.config import Config
from rocket_sim.core.env import RocketEnv
from rocket_sim.rl.networks import SquashedActorCritic
from rocket_sim.rl.policies import autopilot_policy
from rocket_sim.core.utils import autopilot_vz_target
from rocket_sim.rl.curriculum import curriculum_initial_state
from rocket_sim.rl.normalization import RunningMeanStd
from rocket_sim.rl.obs_act import (
    AGENT_DT,
    expand_action,
    get_action_indices,
    get_obs_dim,
    process_obs,
)
from rocket_sim.rl.reward import compute_reward


def run_and_dump(ckpt_path: str, seed: int = 0) -> dict:
    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        config_path = ckpt_dir.parent / "config.json"
    cfg = Config.load(str(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer obs/act dims from checkpoint weights (handles old vs new architecture)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt["policy"]["actor_mean.0.weight"].shape[1]
    act_dim = ckpt["policy"]["actor_mean.4.weight"].shape[0]

    policy = SquashedActorCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    obs_rms = None
    if cfg.normalize_obs and "obs_rms" in ckpt:
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.load_state_dict(ckpt["obs_rms"])

    xml_path = str(Path(__file__).parent.parent / "scene.xml")
    env = RocketEnv(xml_path=xml_path, render_mode="rgb_array", random_init=False, max_steps=cfg.max_steps)

    full_low = env.model.actuator_ctrlrange[:, 0].copy()
    full_high = env.model.actuator_ctrlrange[:, 1].copy()
    act_indices = get_action_indices(cfg.action_mode)
    action_low = full_low[act_indices]
    action_high = full_high[act_indices]
    if act_dim == 12:
        # Old collapsed-leg format
        old_indices = list(range(7)) + [7] + list(range(11, 15))
        action_low = full_low[old_indices]
        action_high = full_high[old_indices]
    else:
        action_low = full_low[act_indices[:act_dim]]
        action_high = full_high[act_indices[:act_dim]]

    def rescale_action(squashed: np.ndarray) -> np.ndarray:
        return action_low + (squashed + 1.0) * 0.5 * (action_high - action_low)

    def expand_to_full(rescaled: np.ndarray) -> np.ndarray:
        """Expand rescaled action to full 15-dim, handling old collapsed-leg format."""
        if act_dim == 12:
            full = np.zeros(15, dtype=np.float32)
            for i in range(7):
                full[i] = rescaled[i]
            leg_val = 2.4 if rescaled[7] > 1.2 else 0.0
            full[7:11] = leg_val
            for i in range(8, 12):
                full[i + 3] = rescaled[i]
            return full
        return expand_action(rescaled, cfg)

    # Use curriculum initial conditions if the checkpoint was trained with curriculum
    ic = None
    if cfg.use_curriculum:
        curriculum_stage = ckpt.get("curriculum_stage", 0)
        difficulty = curriculum_stage / max(cfg.curriculum_stages - 1, 1)
        np.random.seed(seed)
        ic = curriculum_initial_state(difficulty)

    obs, _ = env.reset(seed=seed, initial_state=ic)
    prev = None
    ep_flags: dict = {}
    decompose_flags: dict = {}
    done = False
    steps = 0

    # Conditional frame stacking for old checkpoints
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

    def _augment_obs(single: np.ndarray) -> np.ndarray:
        """Add vz_error + tilt if checkpoint expects augmented obs."""
        if not use_stacking or len(single) >= ckpt_base_dim:
            return single
        state = env._get_state()
        vz_error = state.vertical_speed - autopilot_vz_target(state.altitude)
        tilt = RocketEnv._body_tilt(state.orientation)
        return np.concatenate([single, [vz_error, tilt]]).astype(np.float32)

    records = []

    while not done and steps < cfg.max_steps:
        single = process_obs(obs, cfg, prev)
        if use_stacking:
            single = _augment_obs(single)
            obs_history.append(single)
            stacked = np.concatenate(list(obs_history)).astype(np.float32)
            normed = obs_rms.normalize(stacked).astype(np.float32) if obs_rms else stacked
        else:
            normed = obs_rms.normalize(single).astype(np.float32) if obs_rms else single

        obs_t = torch.as_tensor(normed, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mean = policy.actor_mean(obs_t)
        action_np = torch.tanh(action_mean).cpu().numpy().flatten()
        env_action = expand_to_full(rescale_action(action_np))

        state_before = env._get_state()
        next_obs, _, terminated, truncated, info = env.step(env_action)
        state_after = env._get_state()

        # Compute shaped reward using the same function as training
        shaped_reward = compute_reward(
            state_after, env_action, info, env, cfg, prev_state=state_before,
            ep_flags=ep_flags,
        )

        # Decompose reward into individual terms for analysis
        reward_terms = decompose_reward(state_after, env_action, info, env, cfg, state_before, ep_flags=decompose_flags)

        tilt = float(RocketEnv._body_tilt(state_after.orientation))
        vz_target = autopilot_vz_target(state_after.altitude)

        record = {
            "step": steps,
            "time": round(steps * AGENT_DT, 4),
            # State
            "pos_x": round(float(state_after.position[0]), 4),
            "pos_y": round(float(state_after.position[1]), 4),
            "altitude": round(float(state_after.altitude), 4),
            "vx": round(float(state_after.linear_velocity[0]), 4),
            "vy": round(float(state_after.linear_velocity[1]), 4),
            "vz": round(float(state_after.vertical_speed), 4),
            "speed": round(float(state_after.speed), 4),
            "h_dist": round(float(state_after.horizontal_distance), 4),
            "h_speed": round(float(np.sqrt(state_after.linear_velocity[0]**2 + state_after.linear_velocity[1]**2)), 4),
            "tilt_rad": round(tilt, 6),
            "tilt_deg": round(float(np.degrees(tilt)), 3),
            "roll_deg": round(float(np.degrees(state_after.euler_angles[0])), 3),
            "pitch_deg": round(float(np.degrees(state_after.euler_angles[1])), 3),
            "yaw_deg": round(float(np.degrees(state_after.euler_angles[2])), 3),
            "angvel_x": round(float(state_after.angular_velocity[0]), 5),
            "angvel_y": round(float(state_after.angular_velocity[1]), 5),
            "angvel_z": round(float(state_after.angular_velocity[2]), 5),
            "angvel_mag": round(float(np.sum(np.abs(state_after.angular_velocity))), 5),
            "vz_target": round(vz_target, 4),
            "vz_error": round(abs(float(state_after.vertical_speed) - vz_target), 4),
            # Action
            "act_thrust": round(float(env_action[0]), 5),
            "act_tvc_pitch": round(float(env_action[1]), 5),
            "act_tvc_yaw": round(float(env_action[2]), 5),
            "act_gridfin_0": round(float(env_action[3]), 5),
            "act_gridfin_1": round(float(env_action[4]), 5),
            "act_gridfin_2": round(float(env_action[5]), 5),
            "act_gridfin_3": round(float(env_action[6]), 5),
            "act_leg_0": round(float(env_action[7]), 5),
            "act_leg_1": round(float(env_action[8]), 5),
            "act_leg_2": round(float(env_action[9]), 5),
            "act_leg_3": round(float(env_action[10]), 5),
            # Rewards
            "reward_total": round(shaped_reward, 6),
            **{f"r_{k}": round(v, 6) for k, v in reward_terms.items()},
            # Terminal
            "terminated": terminated,
            "truncated": truncated,
            "landed": info.get("landed", False),
            "crashed": info.get("crashed", False),
        }
        records.append(record)

        prev = obs.copy()
        obs = next_obs
        steps += 1
        done = terminated or truncated

    env.close()
    return {"config": cfg.experiment_name, "seed": seed, "num_steps": steps, "trajectory": records}


def run_autopilot_dump(seed: int = 0) -> dict:
    """Run one episode with the autopilot PD controller and dump diagnostics."""
    cfg = Config()  # default config for reward evaluation
    xml_path = str(Path(__file__).parent.parent / "scene.xml")
    env = RocketEnv(xml_path=xml_path, render_mode="rgb_array", max_steps=cfg.max_steps)

    env.reset(seed=seed)
    ep_flags: dict = {}
    decompose_flags: dict = {}
    done = False
    steps = 0
    records = []

    while not done and steps < cfg.max_steps:
        env_action = autopilot_policy(env)
        state_before = env._get_state()
        _, _, terminated, truncated, info = env.step(env_action)
        state_after = env._get_state()

        shaped_reward = compute_reward(
            state_after, env_action, info, env, cfg, prev_state=state_before,
            ep_flags=ep_flags,
        )
        reward_terms = decompose_reward(state_after, env_action, info, env, cfg, state_before, ep_flags=decompose_flags)

        tilt = float(RocketEnv._body_tilt(state_after.orientation))
        vz_target = autopilot_vz_target(state_after.altitude)

        record = {
            "step": steps,
            "time": round(steps * AGENT_DT, 4),
            "pos_x": round(float(state_after.position[0]), 4),
            "pos_y": round(float(state_after.position[1]), 4),
            "altitude": round(float(state_after.altitude), 4),
            "vx": round(float(state_after.linear_velocity[0]), 4),
            "vy": round(float(state_after.linear_velocity[1]), 4),
            "vz": round(float(state_after.vertical_speed), 4),
            "speed": round(float(state_after.speed), 4),
            "h_dist": round(float(state_after.horizontal_distance), 4),
            "h_speed": round(float(np.sqrt(state_after.linear_velocity[0]**2 + state_after.linear_velocity[1]**2)), 4),
            "tilt_rad": round(tilt, 6),
            "tilt_deg": round(float(np.degrees(tilt)), 3),
            "roll_deg": round(float(np.degrees(state_after.euler_angles[0])), 3),
            "pitch_deg": round(float(np.degrees(state_after.euler_angles[1])), 3),
            "yaw_deg": round(float(np.degrees(state_after.euler_angles[2])), 3),
            "angvel_x": round(float(state_after.angular_velocity[0]), 5),
            "angvel_y": round(float(state_after.angular_velocity[1]), 5),
            "angvel_z": round(float(state_after.angular_velocity[2]), 5),
            "angvel_mag": round(float(np.sum(np.abs(state_after.angular_velocity))), 5),
            "vz_target": round(vz_target, 4),
            "vz_error": round(abs(float(state_after.vertical_speed) - vz_target), 4),
            "act_thrust": round(float(env_action[0]), 5),
            "act_tvc_pitch": round(float(env_action[1]), 5),
            "act_tvc_yaw": round(float(env_action[2]), 5),
            "act_gridfin_0": round(float(env_action[3]), 5),
            "act_gridfin_1": round(float(env_action[4]), 5),
            "act_gridfin_2": round(float(env_action[5]), 5),
            "act_gridfin_3": round(float(env_action[6]), 5),
            "act_leg_0": round(float(env_action[7]), 5),
            "act_leg_1": round(float(env_action[8]), 5),
            "act_leg_2": round(float(env_action[9]), 5),
            "act_leg_3": round(float(env_action[10]), 5),
            "reward_total": round(shaped_reward, 6),
            **{f"r_{k}": round(v, 6) for k, v in reward_terms.items()},
            "terminated": terminated,
            "truncated": truncated,
            "landed": info.get("landed", False),
            "crashed": info.get("crashed", False),
        }
        records.append(record)

        steps += 1
        done = terminated or truncated

    env.close()
    return {"config": "autopilot", "seed": seed, "num_steps": steps, "trajectory": records}


def decompose_reward(state, action, info, env, cfg, prev_state, ep_flags: dict | None = None) -> dict:
    """Break the v35 reward into named components for analysis."""
    terms = {}
    dt = AGENT_DT
    if ep_flags is None:
        ep_flags = {}

    # 1. Potential-based altitude shaping
    if cfg.reward_use_altitude and prev_state is not None:
        alt_delta = float(prev_state.altitude) - float(state.altitude)
        alt_delta = min(alt_delta, 10.0 * dt)
        if alt_delta < 0:
            terms["alt_ascend"] = -0.9 * abs(alt_delta)
        else:
            coeff = 0.5 if state.altitude > 25.0 else 0.25
            terms["alt_descent"] = coeff * alt_delta

    # 2. State quality penalty
    tilt = RocketEnv._body_tilt(state.orientation)
    if cfg.reward_use_tilt:
        terms["tilt"] = -5.0 * tilt * dt

    angvel_sum = sum(abs(float(state.angular_velocity[i])) for i in range(3))
    if cfg.reward_use_angvel:
        terms["angvel"] = -cfg.reward_angvel_coeff * min(angvel_sum, 50.0) * dt

    vx, vy = float(state.linear_velocity[0]), float(state.linear_velocity[1])
    h_speed = np.sqrt(vx**2 + vy**2)
    if cfg.reward_use_horizontal:
        terms["h_speed"] = -0.5 * h_speed * dt

    terms["time"] = -cfg.time_penalty * dt

    # 2b. Slow-descent penalty (anti-hovering)
    if cfg.reward_use_slow_descent:
        vz_down = -float(state.linear_velocity[2])
        if state.altitude > 30.0 and vz_down < 3.0:
            terms["slow_descent"] = -5.0 * (3.0 - vz_down) * dt
        elif state.altitude > 10.0 and state.altitude <= 30.0 and vz_down < 1.0:
            terms["slow_descent"] = -5.0 * (1.0 - vz_down) * dt

    # 3. Leg deployment (latch)
    if cfg.reward_use_legs and state.altitude < cfg.leg_deploy_alt:
        legs_deployed = all(action[i] > 1.2 for i in range(7, 11))
        vz = float(state.linear_velocity[2])
        if legs_deployed and not ep_flags.get("legs_deployed", False) and vz < -0.5:
            terms["leg_deploy"] = 10.0
            ep_flags["legs_deployed"] = True
        elif not legs_deployed and ep_flags.get("legs_deployed", False):
            terms["leg_retract"] = -15.0
            ep_flags["legs_deployed"] = False

    # 4. Terminal
    if info.get("landed", False):
        terms["landing_bonus"] = cfg.landing_bonus * env._compute_landing_quality(state)
    if info.get("crashed", False):
        terms["crash_penalty"] = -(cfg.crash_penalty + 1.0 * state.speed)

    return terms


def analyze(data: dict):
    """Print a diagnostic analysis of the trajectory."""
    traj = data["trajectory"]
    n = len(traj)
    print(f"\n{'='*70}")
    print(f"TRAJECTORY ANALYSIS: {data['config']} (seed={data['seed']})")
    print(f"{'='*70}")
    print(f"Total steps: {n}")

    last = traj[-1]
    if last["landed"]:
        print("Outcome: LANDED")
    elif last["crashed"]:
        print(f"Outcome: CRASHED at step {last['step']}")
    else:
        print("Outcome: TIMEOUT")

    print(f"Final altitude: {last['altitude']:.1f} m")
    print(f"Final speed: {last['speed']:.1f} m/s")
    print(f"Final tilt: {last['tilt_deg']:.1f} deg")
    print(f"Final h_dist: {last['h_dist']:.1f} m")

    # Total rewards by category - gather keys from ALL steps since some
    reward_keys = set()
    for rec in traj:
        reward_keys.update(k for k in rec if k.startswith("r_"))
    reward_keys = sorted(reward_keys)
    totals = {k: 0.0 for k in reward_keys}
    for rec in traj:
        for k in reward_keys:
            totals[k] += rec.get(k, 0.0)

    total_reward = sum(rec["reward_total"] for rec in traj)
    print(f"\nTotal shaped reward: {total_reward:.1f}")
    print("\nReward breakdown (summed over episode):")
    print(f"  {'Component':<25s} {'Total':>10s} {'% of |sum|':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    abs_sum = sum(abs(v) for v in totals.values())
    for k, v in sorted(totals.items(), key=lambda x: -abs(x[1])):
        pct = 100 * abs(v) / abs_sum if abs_sum > 0 else 0
        print(f"  {k:<25s} {v:>10.2f} {pct:>9.1f}%")

    # Phase analysis: split into altitude bands
    bands = [(500, 9999, "High alt (>500m)"), (100, 500, "Mid alt (100-500m)"),
             (30, 100, "Low alt (30-100m)"), (0, 30, "Terminal (<30m)")]
    print("\n--- Phase Analysis ---")
    for lo, hi, label in bands:
        band_recs = [r for r in traj if lo <= r["altitude"] < hi]
        if not band_recs:
            continue
        avg_speed = np.mean([r["speed"] for r in band_recs])
        avg_vz = np.mean([r["vz"] for r in band_recs])
        avg_tilt = np.mean([r["tilt_deg"] for r in band_recs])
        avg_hspd = np.mean([r["h_speed"] for r in band_recs])
        avg_angvel = np.mean([r["angvel_mag"] for r in band_recs])
        avg_thrust = np.mean([r["act_thrust"] for r in band_recs])
        avg_vz_err = np.mean([r["vz_error"] for r in band_recs])
        vz_targets = [r["vz_target"] for r in band_recs]
        avg_reward = np.mean([r["reward_total"] for r in band_recs])
        print(f"\n  {label} ({len(band_recs)} steps)")
        print(f"    speed={avg_speed:.1f} m/s  vz={avg_vz:.1f}  vz_target=[{min(vz_targets):.1f},{max(vz_targets):.1f}]  vz_err={avg_vz_err:.1f}")
        print(f"    tilt={avg_tilt:.1f} deg  h_speed={avg_hspd:.1f}  angvel={avg_angvel:.3f}")
        print(f"    thrust={avg_thrust:.1f}  avg_reward={avg_reward:.4f}")

        # Sum band rewards by component
        band_totals = {k: sum(r.get(k, 0.0) for r in band_recs) for k in reward_keys}
        top3 = sorted(band_totals.items(), key=lambda x: -abs(x[1]))[:5]
        print("    top reward drivers: ", end="")
        print(", ".join(f"{k}={v:.2f}" for k, v in top3))

    # Identify problematic patterns
    print("\n--- Diagnosed Issues ---")

    # Check if rocket is accelerating instead of decelerating
    accel_steps = sum(1 for i in range(1, n) if traj[i]["speed"] > traj[i-1]["speed"] + 0.5)
    print(f"  Steps where speed increased: {accel_steps}/{n} ({100*accel_steps/n:.1f}%)")

    # Check tilt growth
    max_tilt = max(r["tilt_deg"] for r in traj)
    max_tilt_step = next(r["step"] for r in traj if r["tilt_deg"] == max_tilt)
    print(f"  Max tilt: {max_tilt:.1f} deg at step {max_tilt_step}")

    # Check angular velocity spikes
    max_angvel = max(r["angvel_mag"] for r in traj)
    max_angvel_step = next(r["step"] for r in traj if r["angvel_mag"] == max_angvel)
    print(f"  Max angular velocity: {max_angvel:.3f} rad/s at step {max_angvel_step}")

    # Check if thrust is being used
    avg_thrust = np.mean([r["act_thrust"] for r in traj])
    thrust_zero = sum(1 for r in traj if r["act_thrust"] < 0.1)
    print(f"  Avg thrust: {avg_thrust:.1f}  (zero/near-zero: {thrust_zero}/{n} steps)")

    # Terminal phase detail (last 50 steps)
    print("\n--- Last 50 Steps ---")
    for r in traj[-50:]:
        print(f"  t={r['step']:4d} alt={r['altitude']:7.1f} spd={r['speed']:6.1f} "
              f"vz={r['vz']:7.1f} tilt={r['tilt_deg']:5.1f} deg "
              f"angvel={r['angvel_mag']:.3f} "
              f"thrust={r['act_thrust']:6.1f} "
              f"reward={r['reward_total']:+.4f}"
              f"{'  CRASHED' if r['crashed'] else ''}"
              f"{'  LANDED' if r['landed'] else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="?", default=None)
    parser.add_argument("--autopilot", action="store_true", help="Run autopilot PD controller instead of a checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.autopilot:
        data = run_autopilot_dump(args.seed)
        out_path = args.output or "results/autopilot_episode_dump.json"
    elif args.checkpoint:
        data = run_and_dump(args.checkpoint, args.seed)
        out_path = args.output or str(Path(args.checkpoint).parent / "episode_dump.json")
    else:
        parser.error("Provide a checkpoint path or use --autopilot")

    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=_convert)
    print(f"Saved {len(data['trajectory'])} timesteps to {out_path}")

    analyze(data)


if __name__ == "__main__":
    main()
