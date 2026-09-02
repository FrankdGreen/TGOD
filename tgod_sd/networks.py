from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def activation_class(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation {name!r}; choose from {sorted(activations)}") from error


def mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    dimensions = [input_dim, *[int(value) for value in hidden_dims]]
    layers: list[nn.Module] = []
    activation_type = activation_class(activation)
    for first, second in zip(dimensions[:-1], dimensions[1:]):
        layers.extend([nn.Linear(first, second), activation_type()])
    layers.append(nn.Linear(dimensions[-1], output_dim))
    return nn.Sequential(*layers)


def hidden_mlp(input_dim: int, hidden_dims: Sequence[int], activation: str) -> nn.Sequential:
    dimensions = [input_dim, *[int(value) for value in hidden_dims]]
    layers: list[nn.Module] = []
    activation_type = activation_class(activation)
    for first, second in zip(dimensions[:-1], dimensions[1:]):
        layers.extend([nn.Linear(first, second), activation_type()])
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        hidden_dims = [int(value) for value in hidden_dims]
        if not hidden_dims:
            raise ValueError("Actor hidden_dims cannot be empty")
        self.trunk = hidden_mlp(observation_dim + skill_dim, hidden_dims, activation)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

    def forward(
        self,
        observation: torch.Tensor,
        skill: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.trunk(torch.cat([observation, skill], dim=-1))
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), self.LOG_STD_MIN, self.LOG_STD_MAX)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_probability = None
        if with_log_prob:
            log_probability = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
            correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
            log_probability -= correction.sum(dim=-1, keepdim=True)
        return action, log_probability


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.network = mlp(observation_dim + skill_dim + action_dim, hidden_dims, 1, activation)

    def forward(self, observation: torch.Tensor, skill: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observation, skill, action], dim=-1))


class MINEEstimator(nn.Module):
    """Donsker-Varadhan neural mutual-information estimator."""

    def __init__(self, context_dim: int, skill_dim: int, hidden_dims: Sequence[int], activation: str) -> None:
        super().__init__()
        self.statistics_network = mlp(context_dim + skill_dim, hidden_dims, 1, activation)

    def score(self, context: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        return self.statistics_network(torch.cat([context, skill], dim=-1))

    def dv_bound(self, context: torch.Tensor, skill: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(context) < 2:
            raise ValueError("MINE requires a batch with at least two samples")
        joint_score = self.score(context, skill)
        permutation = torch.randperm(len(skill), device=skill.device)
        marginal_score = self.score(context, skill[permutation])
        log_mean_exponential = torch.logsumexp(marginal_score, dim=0) - math.log(len(marginal_score))
        bound = joint_score.mean() - log_mean_exponential.mean()
        return bound, joint_score, marginal_score

    @torch.no_grad()
    def pointwise_reward(self, context: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        joint_score = self.score(context, skill)
        permutation = torch.randperm(len(skill), device=skill.device)
        marginal_score = self.score(context, skill[permutation])
        baseline = torch.logsumexp(marginal_score, dim=0) - math.log(len(marginal_score))
        return joint_score - baseline
