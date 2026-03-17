"""Standalone evaluation of a trained PPO checkpoint.

Usage:
    uv run python evaluate.py results/ppo_baseline/final_model.pt
    uv run python evaluate.py results/ppo_baseline/final_model.pt \
          --episodes 50 --seed 9999
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
from rocket_sim.rl.normalization import RunningMeanStd
from rocket_sim.rl.obs_act import (
    expand_action,
    get_act_dim,
    get_action_indices,
    get_obs_dim,
    process_obs,
)
from rocket_sim.rl.reward import compute_reward


def evaluate_checkpoint(
    ckpt_path: str,
    num_episodes: int = 50,
    seed: int = 9999,
    verbose: bool = True,
) -> dict:
    """Load a checkpoint and run evaluation episodes."""
    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        # checkpoint may be inside checkpoints/ subdir
        config_path = ckpt_dir.parent / "config.json"
    cfg = Config.load(str(config_path))

    obs_dim = get_obs_dim(cfg)
    act_dim = get_act_dim(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SquashedActorCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    obs_rms = None
    if cfg.normalize_obs and "obs_rms" in ckpt:
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.load_state_dict(ckpt["obs_rms"])

    xml_path = str(Path(__file__).parent.parent / "scene.xml")
    env = RocketEnv(
        xml_path=xml_path, render_mode="rgb_array", max_steps=cfg.max_steps
    )

    # Cache actuator ranges for rescaling [-1, 1] -> env
    full_low = env.model.actuator_ctrlrange[:, 0].copy()
    full_high = env.model.actuator_ctrlrange[:, 1].copy()
    act_indices = get_action_indices(cfg.action_mode)
    action_low = full_low[act_indices]
    action_high = full_high[act_indices]

    def rescale_action(squashed: np.ndarray) -> np.ndarray:
        return action_low + (squashed + 1.0) * 0.5 * (action_high - action_low)

    rewards, lengths, successes = [], [], []
    td_speeds, td_tilts, td_offsets, fuels = [], [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        prev = None
        ep_reward, ep_fuel = 0.0, 0.0
        done = False
        steps = 0

        while not done and steps < cfg.max_steps:
            processed = process_obs(obs, cfg, prev)
            normed = (
                obs_rms.normalize(processed).astype(np.float32)
                if obs_rms
                else processed
            )
            obs_t = (
                torch.as_tensor(normed, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                action_mean = policy.actor_mean(obs_t)
            # Deterministic: squash mean through tanh, then rescale
            action_np = torch.tanh(action_mean).cpu().numpy().flatten()
            env_action = expand_action(rescale_action(action_np), cfg)

            prev = obs.copy()
            prev_state = env._get_state()
            next_obs, _, terminated, truncated, info = env.step(env_action)
            state = env._get_state()

            ep_reward += compute_reward(
                state,
                env_action,
                info,
                env,
                cfg,
                prev_state=prev_state,
            )
            ep_fuel += float(env_action[0])
            steps += 1
            done = terminated or truncated

            if done:
                landed = info.get("landed", False)
                successes.append(1.0 if landed else 0.0)
                crashed = info.get("crashed", False)
                if landed:
                    td_speeds.append(state.speed)
                    td_tilts.append(
                        float(
                            np.degrees(RocketEnv._body_tilt(state.orientation))
                        )
                    )
                    td_offsets.append(state.horizontal_distance)
                if verbose:
                    status = (
                        "LANDED"
                        if landed
                        else ("CRASHED" if crashed else "TIMEOUT")
                    )
                    print(
                        f"  ep {ep + 1:3d}: {status} | steps={steps} "
                        f"| speed={state.speed:.1f} m/s "
                        f"| tilt={np.degrees(RocketEnv._body_tilt(state.orientation)):.1f} deg"
                    )
            obs = next_obs

        rewards.append(ep_reward)
        lengths.append(steps)
        fuels.append(ep_fuel)

    env.close()

    result = {
        "checkpoint": ckpt_path,
        "num_episodes": num_episodes,
        "success_rate": float(np.mean(successes)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "length_mean": float(np.mean(lengths)),
        "fuel_mean": float(np.mean(fuels)),
    }
    if td_speeds:
        result["td_speed_mean"] = float(np.mean(td_speeds))
        result["td_speed_std"] = float(np.std(td_speeds))
        result["td_tilt_mean"] = float(np.mean(td_tilts))
        result["td_tilt_std"] = float(np.std(td_tilts))
        result["td_offset_mean"] = float(np.mean(td_offsets))
        result["td_offset_std"] = float(np.std(td_offsets))

    if verbose:
        print("\n=== Evaluation Summary ===")
        print(f"  Success rate: {result['success_rate']:.2%}")
        print(
            f"  Mean reward:  {result['reward_mean']:.1f} +/- {result['reward_std']:.1f}"
        )
        print(f"  Mean length:  {result['length_mean']:.0f}")
        print(f"  Mean fuel:    {result['fuel_mean']:.1f}")
        if td_speeds:
            print(
                f"  TD speed:     {result['td_speed_mean']:.2f} +/- {result['td_speed_std']:.2f} m/s"
            )
            print(
                f"  TD tilt:      {result['td_tilt_mean']:.2f} +/- {result['td_tilt_std']:.2f} deg"
            )
            print(
                f"  TD offset:    {result['td_offset_mean']:.2f} +/- {result['td_offset_std']:.2f} m"
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO checkpoint"
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument(
        "--output", type=str, default=None, help="Save results JSON"
    )
    args = parser.parse_args()

    result = evaluate_checkpoint(args.checkpoint, args.episodes, args.seed)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
