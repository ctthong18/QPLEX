# algorithms/mappo/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def mlp(sizes: List[int], activation=nn.ReLU, output_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else (output_activation or activation)
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """
    Gaussian actor for continuous action spaces.
    Returns mean and uses a learnable log_std (diagonal covariance).
    If action needs squashing (e.g. bounded), apply tanh outside or modify sample().
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=[256, 256], log_std_init: float = -0.5):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes + [act_dim], activation=nn.ReLU, output_activation=None)
        # Parameterize log std as a vector
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns:
            action (torch.Tensor): shape (..., act_dim)
            logp (torch.Tensor): shape (...)
            mu (torch.Tensor)
        """
        mu, std = self.forward(obs)
        if deterministic:
            action = mu
            # approximate logp as distribution at mu
            logp = -0.5 * (((mu - mu) / (std + 1e-8)) ** 2 + 2 * torch.log(std + 1e-8) + torch.log(2 * torch.pi))
            logp = logp.sum(-1)
            return action, logp, mu
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        logp = dist.log_prob(x).sum(-1)
        return x, logp, mu


class CentralizedValue(nn.Module):
    """
    Centralized critic: V(s). Takes global state (e.g. state vector from env).
    """

    def __init__(self, state_dim: int, hidden_sizes=[256, 256]):
        super().__init__()
        self.net = mlp([state_dim] + hidden_sizes + [1], activation=nn.ReLU, output_activation=None)

    def forward(self, state: torch.Tensor):
        return self.net(state).squeeze(-1)  # (batch,)
