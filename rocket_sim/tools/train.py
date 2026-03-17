"""CLI entry point for PPO training with ablation support.

Usage examples:
    # Baseline PPO
    uv run python train.py

    # Sparse reward ablation
    uv run python train.py --reward_mode sparse --experiment_name reward_sparse

    # No angular velocity observation
    uv run python train.py --obs_mode no_angvel --experiment_name obs_no_angvel

    # No grid fins
    uv run python train.py --action_mode no_gridfins --experiment_name act_no_gridfins

    # Curriculum training
    uv run python train.py --use_curriculum --experiment_name stab_curriculum

    # Multiple seeds
    for seed in 0 1 2; do
        uv run python train.py --seed $seed --experiment_name "baseline_s$seed"
    done
"""

import argparse
import os

# Use EGL for headless rendering (needed even though we don't render during training
os.environ.setdefault("MUJOCO_GL", "egl")

from rocket_sim.core.config import Config
from rocket_sim.rl.ppo import PPO


def parse_args() -> tuple[Config, str | None]:
    defaults = Config()
    p = argparse.ArgumentParser(description="Train PPO on 3D rocket landing")

    # Environment
    p.add_argument("--max_steps", type=int, default=3000)

    # PPO
    p.add_argument("--total_timesteps", type=int, default=5_000_000)
    p.add_argument("--rollout_steps", type=int, default=defaults.rollout_steps)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--entropy_coeff", type=float, default=defaults.entropy_coeff
    )
    p.add_argument("--value_coeff", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--num_mini_batches", type=int, default=32)
    p.add_argument("--target_kl", type=float, default=defaults.target_kl)

    # Network
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])

    # Reward ablation
    p.add_argument(
        "--reward_mode", choices=["dense", "sparse"], default="dense"
    )
    p.add_argument("--no_reward_altitude", action="store_true")
    p.add_argument("--no_reward_tilt", action="store_true")
    p.add_argument("--no_reward_slow_descent", action="store_true")
    p.add_argument("--no_reward_horizontal", action="store_true")
    p.add_argument("--no_reward_angvel", action="store_true")
    p.add_argument("--no_reward_legs", action="store_true")
    p.add_argument(
        "--reward_angvel_coeff",
        type=float,
        default=defaults.reward_angvel_coeff,
    )
    p.add_argument(
        "--landing_bonus", type=float, default=defaults.landing_bonus
    )
    p.add_argument(
        "--crash_penalty", type=float, default=defaults.crash_penalty
    )
    p.add_argument("--time_penalty", type=float, default=defaults.time_penalty)
    p.add_argument(
        "--timeout_penalty",
        type=float,
        default=defaults.timeout_penalty,
    )
    p.add_argument(
        "--leg_deploy_alt",
        type=float,
        default=defaults.leg_deploy_alt,
    )

    # Observation ablation
    p.add_argument(
        "--obs_mode",
        choices=["full", "no_angvel", "noisy", "delayed"],
        default="full",
    )
    p.add_argument("--obs_noise_std", type=float, default=0.1)

    # Action ablation
    p.add_argument(
        "--action_mode",
        choices=["full", "no_gridfins", "no_rcs", "no_gridfins_rcs"],
        default="full",
    )

    # Training stabilizers
    p.add_argument("--no_normalize_obs", action="store_true")
    p.add_argument("--no_normalize_advantages", action="store_true")
    p.add_argument("--normalize_rewards", action="store_true", default=True)
    p.add_argument("--no_normalize_rewards", action="store_true")
    p.add_argument(
        "--lr_schedule", choices=["linear", "constant"], default="linear"
    )
    p.add_argument("--use_curriculum", action="store_true", default=True)
    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--curriculum_stages", type=int, default=40)
    p.add_argument("--curriculum_advance_threshold", type=float, default=0.3)

    # Evaluation
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument(
        "--eval_mode",
        choices=["curriculum", "default"],
        default=defaults.eval_mode,
    )
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=50)

    # Reproducibility
    p.add_argument("--seed", type=int, default=0)

    # Debugging
    p.add_argument(
        "--debug_log",
        action="store_true",
        help="Log per-step x,y,z,vx,vy,vz,wx,wy,wz,tilt to stdout",
    )

    # Output
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--experiment_name", type=str, default="ppo_baseline")

    # Resume from checkpoint
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt file to resume training from")

    args = p.parse_args()

    cfg = Config(
        max_steps=args.max_steps,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        num_mini_batches=args.num_mini_batches,
        target_kl=args.target_kl,
        hidden_sizes=tuple(args.hidden_sizes),
        reward_mode=args.reward_mode,
        reward_use_altitude=not args.no_reward_altitude,
        reward_use_tilt=not args.no_reward_tilt,
        reward_use_slow_descent=not args.no_reward_slow_descent,
        reward_use_horizontal=not args.no_reward_horizontal,
        reward_use_angvel=not args.no_reward_angvel,
        reward_use_legs=not args.no_reward_legs,
        reward_angvel_coeff=args.reward_angvel_coeff,
        landing_bonus=args.landing_bonus,
        crash_penalty=args.crash_penalty,
        time_penalty=args.time_penalty,
        timeout_penalty=args.timeout_penalty,
        leg_deploy_alt=args.leg_deploy_alt,
        obs_mode=args.obs_mode,
        obs_noise_std=args.obs_noise_std,
        action_mode=args.action_mode,
        normalize_obs=not args.no_normalize_obs,
        normalize_advantages=not args.no_normalize_advantages,
        normalize_rewards=args.normalize_rewards
        and not args.no_normalize_rewards,
        lr_schedule=args.lr_schedule,
        use_curriculum=args.use_curriculum and not args.no_curriculum,
        curriculum_stages=args.curriculum_stages,
        curriculum_advance_threshold=args.curriculum_advance_threshold,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_mode=args.eval_mode,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        debug_log=args.debug_log,
    )
    return cfg, args.resume


def main():
    cfg, resume_path = parse_args()
    trainer = PPO(cfg)
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
    trainer.train()


if __name__ == "__main__":
    main()
