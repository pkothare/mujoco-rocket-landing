"""Running statistics and rollout buffer for PPO."""

import numpy as np
import torch


class RunningMeanStd:
    """Mean / variance tracker."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (
            m_a + m_b + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class RolloutBuffer:
    """Fixed-size buffer for on-policy rollouts."""

    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.size = size
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        last_gae = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            non_term = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_value * non_term
                - self.values[t]
            )
            self.advantages[t] = last_gae = (
                delta + gamma * gae_lambda * non_term * last_gae
            )
        self.returns[:] = self.advantages + self.values

    def get_batches(self, num_mini_batches: int):
        indices = np.random.permutation(self.size)
        batch_size = self.size // num_mini_batches
        for start in range(0, self.size, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32),
                "actions": torch.as_tensor(
                    self.actions[idx], dtype=torch.float32
                ),
                "log_probs": torch.as_tensor(
                    self.log_probs[idx], dtype=torch.float32
                ),
                "advantages": torch.as_tensor(
                    self.advantages[idx], dtype=torch.float32
                ),
                "returns": torch.as_tensor(
                    self.returns[idx], dtype=torch.float32
                ),
            }

    def reset(self):
        self.ptr = 0
