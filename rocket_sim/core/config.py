"""Configuration for PPO training and ablation experiments."""

import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    """All hyperparameters and ablation flags in one place."""

    # Environment
    max_steps: int = 3000

    # PPO core
    total_timesteps: int = 5_000_000
    rollout_steps: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    entropy_coeff: float = 0.001
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 10
    num_mini_batches: int = 32
    target_kl: float | None = 0.05

    # Network
    hidden_sizes: tuple[int, ...] = (256, 256)

    # Reward ablation
    reward_mode: str = "dense"  # "dense" or "sparse"
    reward_use_altitude: bool = True
    reward_use_tilt: bool = True
    reward_use_horizontal: bool = True
    reward_use_angvel: bool = True
    reward_use_slow_descent: bool = True
    reward_use_legs: bool = True
    reward_angvel_coeff: float = 1.0
    landing_bonus: float = 100.0
    crash_penalty: float = 50.0
    time_penalty: float = 5.0
    timeout_penalty: float = 200.0
    leg_deploy_alt: float = 200.0  # reward leg deployment below this altitude

    # Observation ablation
    obs_mode: str = "full"  # "full", "no_angvel", "noisy", "delayed"
    obs_noise_std: float = 0.1
    obs_augment: bool = False  # append vz_error + tilt to obs (not implemented in training)
    obs_history_len: int = 1  # stack N frames (1 = no stacking, not implemented in training)

    # Action ablation
    action_mode: str = (
        "full"  # "full", "no_gridfins", "no_rcs", "no_gridfins_rcs"
    )

    # Training stabilizers
    normalize_obs: bool = True
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    lr_schedule: str = "linear"  # "constant" or "linear"
    use_curriculum: bool = True
    curriculum_stages: int = 40
    curriculum_advance_threshold: float = 0.3

    # Evaluation
    eval_interval: int = 10
    eval_episodes: int = 20
    eval_mode: str = "curriculum"  # "curriculum" or "default"
    log_interval: int = 1
    save_interval: int = 50

    # Reproducibility
    seed: int = 0

    # Debugging
    debug_log: bool = False  # log per-step state vectors to stdout

    # Output
    output_dir: str = "results"
    experiment_name: str = "ppo_baseline"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        # Convert list back to tuple for hidden_sizes
        if "hidden_sizes" in d and isinstance(d["hidden_sizes"], list):
            d["hidden_sizes"] = tuple(d["hidden_sizes"])
        # Backwards compat: old configs lack obs_augment /
        d.setdefault("obs_augment", False)
        d.setdefault("obs_history_len", 1)
        # Filter to known fields so old configs missing new
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)
