"""Proximal Policy Optimization (PPO) trainer.

The PPO class orchestrates the full training pipeline:
  - Environment setup with optional curriculum
  - Rollout collection with GAE
  - Clipped surrogate policy updates
  - Evaluation and checkpointing
"""

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

from rocket_sim.core.config import Config
from rocket_sim.core.env import RocketEnv
from .curriculum import CurriculumSchedule, curriculum_initial_state
from .networks import SquashedActorCritic
from .normalization import RolloutBuffer, RunningMeanStd
from .obs_act import (
    expand_action,
    get_act_dim,
    get_action_indices,
    get_obs_dim,
    process_obs,
)
from .reward import compute_reward


class PPO:
    """Full PPO training pipeline with ablation support."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Seed
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Environment
        xml_path = str(Path(__file__).parent.parent / "scene.xml")
        self.env = RocketEnv(
            xml_path=xml_path,
            render_mode="rgb_array",
            max_steps=cfg.max_steps,
            random_init=not cfg.use_curriculum,
        )

        # Dimensions
        self.obs_dim = get_obs_dim(cfg)
        self.act_dim = get_act_dim(cfg)

        # Cache actuator control ranges for action rescaling.
        full_low = self.env.model.actuator_ctrlrange[:, 0].copy()
        full_high = self.env.model.actuator_ctrlrange[:, 1].copy()
        self._act_indices = get_action_indices(cfg.action_mode)
        self._action_low = full_low[self._act_indices]
        self._action_high = full_high[self._act_indices]

        # Policy - use tanh-squashed Gaussian so that the log-prob
        self.policy = SquashedActorCritic(
            self.obs_dim, self.act_dim, cfg.hidden_sizes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.lr, eps=1e-5
        )

        # Normalization
        self.obs_rms = (
            RunningMeanStd(shape=(self.obs_dim,))
            if cfg.normalize_obs
            else None
        )
        self.ret_rms = (
            RunningMeanStd(shape=()) if cfg.normalize_rewards else None
        )
        self._discounted_return = 0.0

        # Buffer
        self.buffer = RolloutBuffer(
            cfg.rollout_steps, self.obs_dim, self.act_dim
        )

        # Curriculum
        self.curriculum = (
            CurriculumSchedule(
                cfg.curriculum_stages, cfg.curriculum_advance_threshold
            )
            if cfg.use_curriculum
            else None
        )

        # Output dir
        self.out_dir = Path(cfg.output_dir) / cfg.experiment_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        cfg.save(str(self.out_dir / "config.json"))

        # Logging
        self.metrics_path = self.out_dir / "metrics.jsonl"
        if self.metrics_path.exists():
            self.metrics_path.unlink()

        # State
        self.global_step = 0
        self._prev_raw_obs: np.ndarray | None = None

    def _process_and_normalize(
        self, raw_obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process obs for ablation, update rms, return (processed, normalized)."""
        processed = process_obs(raw_obs, self.cfg, self._prev_raw_obs)
        if self.obs_rms is not None:
            self.obs_rms.update(processed)
            normed = self.obs_rms.normalize(processed).astype(np.float32)
        else:
            normed = processed
        return processed, normed

    def _normalize_only(self, processed: np.ndarray) -> np.ndarray:
        if self.obs_rms is not None:
            return self.obs_rms.normalize(processed).astype(np.float32)
        return processed
    
    def _rescale_action(self, squashed: np.ndarray) -> np.ndarray:
        """Map action from [-1, 1] -> actuator [low, high]."""
        return self._action_low + (squashed + 1.0) * 0.5 * (
            self._action_high - self._action_low
        )

    def _reset_env(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        if self.curriculum is not None:
            init = curriculum_initial_state(self.curriculum.difficulty)
            self.env._set_state(init)
            mujoco.mj_forward(self.env.model, self.env.data)  # type: ignore[attr-defined]
            obs = self.env._get_observation()
            info = self.env._get_info()
        self._prev_raw_obs = None
        return obs, info

    def collect_rollout(self, obs: np.ndarray):
        """Fill the buffer with rollout_steps transitions. Returns updated obs and episode stats."""
        cfg = self.cfg
        self.buffer.reset()
        ep_rewards: list[float] = []
        ep_raw_rewards: list[float] = []
        ep_lengths: list[int] = []
        ep_successes: list[float] = []
        cur_ep_reward = 0.0
        cur_ep_raw_reward = 0.0
        cur_ep_len = 0
        cur_ep_flags: dict = {}

        for _ in range(cfg.rollout_steps):
            self.global_step += 1
            processed, normed = self._process_and_normalize(obs)
            prev_state = self.env._get_state()
            obs_t = (
                torch.as_tensor(normed, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                action, log_prob, value = self.policy.act(obs_t)
            # action is in [-1, 1] (squashed); store it for PPO ratio
            action_np = action.cpu().numpy().flatten()

            # Rescale to actuator ranges for the environment
            rescaled = self._rescale_action(action_np)
            env_action = expand_action(rescaled, cfg)

            next_obs, reward, terminated, truncated, info = self.env.step(
                env_action
            )

            # Override reward for ablation
            state = self.env._get_state()
            reward = compute_reward(
                state,
                env_action,
                info,
                self.env,
                cfg,
                prev_state=prev_state,
                ep_flags=cur_ep_flags,
            )

            # Timeout penalty: explicit cost for running out of time
            if (
                truncated
                and not info.get("landed", False)
                and not info.get("crashed", False)
            ):
                reward -= cfg.timeout_penalty

            # Reward normalization: scale by running std of returns
            if self.ret_rms is not None:
                self._discounted_return = (
                    reward + cfg.gamma * self._discounted_return
                )
                self.ret_rms.update(np.array([self._discounted_return]))
                cur_ep_raw_reward += reward
                reward = float(reward / np.sqrt(self.ret_rms.var + 1e-8))
            else:
                cur_ep_raw_reward += reward

            cur_ep_reward += reward
            cur_ep_len += 1

            # Bootstrap for truncation
            if truncated and not terminated:
                next_proc = process_obs(next_obs, cfg, obs)
                next_norm = self._normalize_only(next_proc)
                with torch.no_grad():
                    boot = self.policy.get_value(
                        torch.as_tensor(next_norm, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    ).item()
                reward += cfg.gamma * boot

            done_f = float(terminated or truncated)
            self.buffer.add(
                processed,
                action_np,
                reward,
                value.item(),
                log_prob.item(),
                done_f,
            )

            if terminated or truncated:
                ep_rewards.append(cur_ep_reward)
                ep_raw_rewards.append(cur_ep_raw_reward)
                ep_lengths.append(cur_ep_len)
                ep_successes.append(1.0 if info.get("landed", False) else 0.0)
                cur_ep_reward, cur_ep_raw_reward, cur_ep_len = 0.0, 0.0, 0
                cur_ep_flags = {}
                if self.ret_rms is not None:
                    self._discounted_return = 0.0
                obs, _ = self._reset_env()
            else:
                self._prev_raw_obs = obs.copy()
                obs = next_obs

        # GAE
        _, norm_final = self._process_and_normalize(obs)
        with torch.no_grad():
            last_val = self.policy.get_value(
                torch.as_tensor(norm_final, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            ).item()
        self.buffer.compute_gae(last_val, cfg.gamma, cfg.gae_lambda)

        return obs, ep_rewards, ep_raw_rewards, ep_lengths, ep_successes

    def update(self):
        cfg = self.cfg
        pg_losses, vf_losses, entropies, kls = [], [], [], []

        for _ in range(cfg.num_epochs):
            for batch in self.buffer.get_batches(cfg.num_mini_batches):
                obs_b = batch["obs"].to(self.device)
                act_b = batch["actions"].to(self.device)
                old_lp = batch["log_probs"].to(self.device)
                adv_b = batch["advantages"].to(self.device)
                ret_b = batch["returns"].to(self.device)

                # Apply normalization directly (buffer stores processed raw obs)
                if self.obs_rms is not None:
                    mean_t = torch.as_tensor(
                        self.obs_rms.mean,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    std_t = torch.sqrt(
                        torch.as_tensor(
                            self.obs_rms.var,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        + 1e-8
                    )
                    obs_b = (obs_b - mean_t) / std_t

                if cfg.normalize_advantages:
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                new_lp, new_val, entropy = self.policy.evaluate_actions(
                    obs_b, act_b
                )
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv_b
                surr2 = (
                    torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                    * adv_b
                )
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * (new_val - ret_b).pow(2).mean()
                ent_loss = -entropy.mean()

                loss = (
                    pg_loss
                    + cfg.value_coeff * vf_loss
                    + cfg.entropy_coeff * ent_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_lp - new_lp).mean().item()
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(entropy.mean().item())
                kls.append(approx_kl)

                # Early stop within epoch if KL too large
                if (
                    cfg.target_kl is not None
                    and abs(approx_kl) > cfg.target_kl * 1.5
                ):
                    break

            if cfg.target_kl is not None and abs(np.mean(kls)) > cfg.target_kl:
                break

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "vf_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
        }

    def evaluate(self, num_episodes: int = 20, eval_seed: int = 10000) -> dict:
        """Run deterministic evaluation episodes.

        When eval_mode == "curriculum", episodes use the current
        curriculum difficulty so metrics track what PPO is actually
        learning.  When "default", the environment's built-in reset
        (full difficulty) is used for generalization monitoring.
        """
        rewards, lengths, successes = [], [], []
        td_speeds, td_tilts, td_offsets, fuels = [], [], [], []

        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=eval_seed + ep)
            if self.cfg.eval_mode == "curriculum" and self.curriculum is not None:
                init = curriculum_initial_state(self.curriculum.difficulty)
                self.env._set_state(init)
                mujoco.mj_forward(self.env.model, self.env.data) # type: ignore
                obs = self.env._get_observation()
            prev = None
            ep_reward, ep_fuel = 0.0, 0.0
            eval_ep_flags: dict = {}
            done = False
            steps = 0

            while not done and steps < self.cfg.max_steps:
                processed = process_obs(obs, self.cfg, prev)
                normed = self._normalize_only(processed)
                prev_state = self.env._get_state()
                obs_t = (
                    torch.as_tensor(normed, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    mean = self.policy.actor_mean(obs_t)
                # Deterministic action: squash mean through tanh, then rescale
                action_np = torch.tanh(mean).cpu().numpy().flatten()
                rescaled = self._rescale_action(action_np)
                env_action = expand_action(rescaled, self.cfg)

                prev = obs.copy()
                next_obs, _, terminated, truncated, info = self.env.step(
                    env_action
                )
                state = self.env._get_state()
                reward = compute_reward(
                    state,
                    env_action,
                    info,
                    self.env,
                    self.cfg,
                    prev_state=prev_state,
                    ep_flags=eval_ep_flags,
                )
                ep_reward += reward
                ep_fuel += float(env_action[0])
                steps += 1
                done = terminated or truncated

                if done:
                    landed = info.get("landed", False)
                    successes.append(1.0 if landed else 0.0)
                    if landed:
                        td_speeds.append(state.speed)
                        td_tilts.append(
                            float(
                                np.degrees(
                                    RocketEnv._body_tilt(state.orientation)
                                )
                            )
                        )
                        td_offsets.append(state.horizontal_distance)
                obs = next_obs

            rewards.append(ep_reward)
            lengths.append(steps)
            fuels.append(ep_fuel)

        result = {
            "eval_reward_mean": float(np.mean(rewards)),
            "eval_reward_std": float(np.std(rewards)),
            "eval_length_mean": float(np.mean(lengths)),
            "eval_success_rate": float(np.mean(successes))
            if successes
            else 0.0,
            "eval_fuel_mean": float(np.mean(fuels)),
        }
        if td_speeds:
            result["eval_td_speed_mean"] = float(np.mean(td_speeds))
            result["eval_td_tilt_mean"] = float(np.mean(td_tilts))
            result["eval_td_offset_mean"] = float(np.mean(td_offsets))
        return result

    def save_checkpoint(self, path: str) -> None:
        d = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.obs_rms is not None:
            d["obs_rms"] = self.obs_rms.state_dict()
        if self.ret_rms is not None:
            d["ret_rms"] = self.ret_rms.state_dict()
        if self.curriculum is not None:
            d["curriculum_stage"] = self.curriculum.stage
        torch.save(d, path)

    def load_checkpoint(self, path: str) -> None:
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(d["policy"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.global_step = d["global_step"]
        if self.obs_rms is not None and "obs_rms" in d:
            self.obs_rms.load_state_dict(d["obs_rms"])
        if self.ret_rms is not None and "ret_rms" in d:
            self.ret_rms.load_state_dict(d["ret_rms"])
        if self.curriculum is not None and "curriculum_stage" in d:
            self.curriculum.stage = d["curriculum_stage"]

    def train(self) -> Path:
        cfg = self.cfg
        num_iters = cfg.total_timesteps // cfg.rollout_steps
        start_iter = self.global_step // cfg.rollout_steps + 1
        obs, _ = self._reset_env(seed=cfg.seed)

        print(
            f"PPO training: {cfg.total_timesteps} steps, {num_iters} iterations"
        )
        if start_iter > 1:
            print(f"  resuming from iteration {start_iter} (step {self.global_step})")
        print(
            f"  obs_dim={self.obs_dim}, act_dim={self.act_dim}, device={self.device}"
        )
        print(f"  experiment: {cfg.experiment_name}")
        print(f"  output: {self.out_dir}")

        for iteration in range(start_iter, num_iters + 1):
            t0 = time.time()

            # LR schedule
            if cfg.lr_schedule == "linear":
                frac = 1.0 - (iteration - 1) / num_iters
                for pg in self.optimizer.param_groups:
                    pg["lr"] = cfg.lr * frac

            # Collect & update
            obs, ep_rewards, ep_raw_rewards, ep_lengths, ep_successes = self.collect_rollout(
                obs
            )
            update_stats = self.update()

            # Curriculum
            if self.curriculum is not None and ep_successes:
                self.curriculum.update(ep_successes)

            elapsed = time.time() - t0
            fps = cfg.rollout_steps / elapsed if elapsed > 0 else 0

            # Build log record
            record: dict = {
                "iteration": iteration,
                "global_step": self.global_step,
                "fps": round(fps, 1),
                **update_stats,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if ep_rewards:
                record["ep_reward_mean"] = float(np.mean(ep_rewards))
                record["ep_reward_std"] = float(np.std(ep_rewards))
                record["ep_raw_reward_mean"] = float(np.mean(ep_raw_rewards))
                record["ep_raw_reward_std"] = float(np.std(ep_raw_rewards))
                record["ep_length_mean"] = float(np.mean(ep_lengths))
                record["ep_success_rate"] = float(np.mean(ep_successes))
                record["episodes"] = len(ep_rewards)
            if self.curriculum is not None:
                record["curriculum_stage"] = self.curriculum.stage
                record["curriculum_difficulty"] = self.curriculum.difficulty

            # Evaluate
            if iteration % cfg.eval_interval == 0:
                eval_stats = self.evaluate(cfg.eval_episodes)
                record.update(eval_stats)

            # Log
            if iteration % cfg.log_interval == 0:
                with open(self.metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if ep_rewards:
                    sr = record.get("ep_success_rate", 0)
                    print(
                        f"[{iteration}/{num_iters}] step={self.global_step} "
                        f"reward={record['ep_reward_mean']:.1f} "
                        f"success={sr:.2f} "
                        f"pg={update_stats['pg_loss']:.4f} "
                        f"vf={update_stats['vf_loss']:.4f} "
                        f"ent={update_stats['entropy']:.3f} "
                        f"fps={fps:.0f}"
                    )
                    if self.curriculum is not None:
                        print(
                            f"  curriculum stage {self.curriculum.stage}/{cfg.curriculum_stages - 1}"
                        )

            # Checkpoint
            if iteration % cfg.save_interval == 0:
                self.save_checkpoint(
                    str(self.ckpt_dir / f"model_{self.global_step}.pt")
                )

        # Final save
        self.save_checkpoint(str(self.out_dir / "final_model.pt"))
        self.env.close()
        print(f"Training complete. Results saved to {self.out_dir}")
        return self.out_dir
