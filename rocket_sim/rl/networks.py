"""Actor-Critic neural networks for PPO.

Provides a tanh-squashed Gaussian actor-critic (SquashedActorCritic)
that outputs actions in [-1, 1] with a correct log-prob correction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    output_gain: float = 0.01,
) -> nn.Sequential:
    """Build an MLP with orthogonal initialization and Tanh activations."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        linear = nn.Linear(prev, h)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        prev = h
    out = nn.Linear(prev, output_dim)
    nn.init.orthogonal_(out.weight, gain=output_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class SquashedActorCritic(nn.Module):
    """Actor-critic with tanh-squashed Gaussian policy.

    Actions are guaranteed to lie in [-1, 1].  The log-probability
    includes the tanh Jacobian correction so that the PPO likelihood
    ratio is computed for the *executed* action, not the pre-squash
    sample.

    The caller is responsible for rescaling [-1, 1] -> actuator range.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.actor_mean = _build_mlp(
            obs_dim, act_dim, hidden_sizes, output_gain=0.01
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = _build_mlp(obs_dim, 1, hidden_sizes, output_gain=1.0)

    def _get_dist(self, obs: torch.Tensor):
        mu = self.actor_mean(obs)
        log_std = torch.clamp(self.actor_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mu, log_std.exp().expand_as(mu))

    @staticmethod
    def _squashed_log_prob(
        dist: Normal, u: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Log-prob of tanh(u) under *dist*, with Jacobian correction."""
        log_prob_u = dist.log_prob(u).sum(-1)
        # Jacobian: log |det d tanh/du| = sum log(1 - tanh(u)^2)
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + eps).sum(-1)
        return log_prob_u - correction

    def forward(self, obs: torch.Tensor):
        dist = self._get_dist(obs)
        value = self.critic(obs).squeeze(-1)
        return dist.mean, dist.stddev, value

    def act(self, obs: torch.Tensor):
        """Sample a squashed action; returns (action_in_[-1,1], log_prob, value)."""
        dist = self._get_dist(obs)
        u = dist.rsample()  # pre-squash sample
        action = torch.tanh(u)
        log_prob = self._squashed_log_prob(dist, u)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log-prob for *squashed* actions already in [-1, 1].

        We need to recover the pre-squash value u = atanh(action) so we
        can compute the correct log-probability.
        """
        dist = self._get_dist(obs)
        # Recover u from squashed action (clamp to avoid atanh at +/-1)
        u = torch.atanh(actions.clamp(-0.999, 0.999))
        log_prob = self._squashed_log_prob(dist, u)
        # Entropy of the squashed distribution has no closed form;
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, value, entropy
