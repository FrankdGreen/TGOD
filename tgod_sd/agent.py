from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .expert import RELATION_PROXIMITY_INDEX
from .networks import MINEEstimator, QNetwork, SquashedGaussianActor
from .schema import OBSERVATION_SCALE


class RunningMoments:
    def __init__(self) -> None:
        self.mean = 0.0
        self.variance = 1.0
        self.count = 1e-4

    def update(self, values: torch.Tensor) -> None:
        batch_mean = float(values.mean().item())
        batch_variance = float(values.var(unbiased=False).item())
        batch_count = float(values.numel())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        first = self.variance * self.count
        second = batch_variance * batch_count
        correction = delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.variance = max((first + second + correction) / total, 1e-8)
        self.count = total

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / math.sqrt(self.variance + 1e-8)

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "variance": self.variance, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean = float(state["mean"])
        self.variance = float(state["variance"])
        self.count = float(state["count"])


class TGODSACAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        relation_dim: int,
        config: dict[str, Any],
        device: str,
    ) -> None:
        self.device = torch.device(device)
        network_config = config["network"]
        tgod_config = config["tgod"]
        sac_config = config["sac"]
        self.skill_dim = int(tgod_config["num_skills"])
        hidden_dims = network_config["hidden_dims"]
        mine_hidden_dims = network_config["mine_hidden_dims"]
        activation = str(network_config["activation"])

        self.actor = SquashedGaussianActor(
            observation_dim, self.skill_dim, action_dim, hidden_dims, activation
        ).to(self.device)
        self.q1 = QNetwork(observation_dim, self.skill_dim, action_dim, hidden_dims, activation).to(self.device)
        self.q2 = QNetwork(observation_dim, self.skill_dim, action_dim, hidden_dims, activation).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device).requires_grad_(False)
        self.state_mine = MINEEstimator(observation_dim, self.skill_dim, mine_hidden_dims, activation).to(self.device)
        self.demo_mine = MINEEstimator(relation_dim, self.skill_dim, mine_hidden_dims, activation).to(self.device)

        learning_rate = float(sac_config["learning_rate"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate
        )
        self.mine_optimizer = torch.optim.Adam(
            list(self.state_mine.parameters()) + list(self.demo_mine.parameters()),
            lr=float(tgod_config["mine_learning_rate"]),
        )
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(sac_config["alpha_learning_rate"])
        )

        target_entropy = sac_config["target_entropy"]
        self.target_entropy = -float(action_dim) if str(target_entropy).lower() == "auto" else float(target_entropy)
        self.gamma = float(sac_config["gamma"])
        self.tau = float(sac_config["tau"])
        self.state_mi_weight = float(tgod_config["state_mi_weight"])
        self.demonstration_mi_weight = float(tgod_config["demonstration_mi_weight"])
        self.demonstration_support_weight = float(tgod_config["demonstration_support_weight"])
        self.demonstration_progress_weight = float(tgod_config["demonstration_progress_weight"])
        self.pseudo_reward_clip = float(tgod_config["pseudo_reward_clip"])
        self.reward_normalization = bool(tgod_config["reward_normalization"])
        self.mine_gradient_clip = float(tgod_config["mine_gradient_clip"])
        self.gradient_clip = float(sac_config["gradient_clip"])
        self.reward_moments = RunningMoments()
        self.observation_scale = torch.as_tensor(OBSERVATION_SCALE, device=self.device)
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return observation / self.observation_scale

    @torch.no_grad()
    def act(self, observation: np.ndarray, skill: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation_tensor = self._normalize_observation(self._tensor(observation).unsqueeze(0))
        skill_tensor = self._tensor(skill).unsqueeze(0)
        action, _ = self.actor(observation_tensor, skill_tensor, deterministic, with_log_prob=False)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        observation = self._normalize_observation(self._tensor(batch["observation"]))
        next_observation = self._normalize_observation(self._tensor(batch["next_observation"]))
        action = self._tensor(batch["action"])
        skill = self._tensor(batch["skill"])
        relation = self._tensor(batch["relation"])
        next_relation = self._tensor(batch["next_relation"])
        terminal = self._tensor(batch["terminal"])

        state_bound, _, _ = self.state_mine.dv_bound(observation, skill)
        demo_bound, _, _ = self.demo_mine.dv_bound(relation, skill)
        mine_loss = -(self.state_mi_weight * state_bound + self.demonstration_mi_weight * demo_bound)
        self.mine_optimizer.zero_grad(set_to_none=True)
        mine_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.state_mine.parameters()) + list(self.demo_mine.parameters()), self.mine_gradient_clip
        )
        self.mine_optimizer.step()

        with torch.no_grad():
            demonstration_support = torch.clamp(
                relation[:, RELATION_PROXIMITY_INDEX : RELATION_PROXIMITY_INDEX + 1],
                min=1e-6,
                max=1.0,
            )
            next_demonstration_support = torch.clamp(
                next_relation[:, RELATION_PROXIMITY_INDEX : RELATION_PROXIMITY_INDEX + 1],
                min=1e-6,
                max=1.0,
            )
            support_reward = self.demonstration_support_weight * torch.log(demonstration_support)
            # Potential-based dense guidance: reward actions that move the next
            # state closer to the time-aligned expert state and penalize regress.
            progress_reward = self.demonstration_progress_weight * (
                self.gamma * torch.log(next_demonstration_support)
                - torch.log(demonstration_support)
            )
            raw_reward = (
                self.state_mi_weight * self.state_mine.pointwise_reward(observation, skill)
                + self.demonstration_mi_weight * self.demo_mine.pointwise_reward(relation, skill)
                + support_reward
                + progress_reward
            )
            if self.reward_normalization:
                self.reward_moments.update(raw_reward)
                pseudo_reward = self.reward_moments.normalize(raw_reward)
            else:
                pseudo_reward = raw_reward
            pseudo_reward = torch.clamp(pseudo_reward, -self.pseudo_reward_clip, self.pseudo_reward_clip)

            next_action, next_log_probability = self.actor(next_observation, skill)
            assert next_log_probability is not None
            target_q = torch.minimum(
                self.target_q1(next_observation, skill, next_action),
                self.target_q2(next_observation, skill, next_action),
            ) - self.alpha.detach() * next_log_probability
            q_target = pseudo_reward + self.gamma * (1.0 - terminal) * target_q

        q1_value = self.q1(observation, skill, action)
        q2_value = self.q2(observation, skill, action)
        q_loss = F.mse_loss(q1_value, q_target) + F.mse_loss(q2_value, q_target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.gradient_clip)
        self.q_optimizer.step()

        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        policy_action, log_probability = self.actor(observation, skill)
        assert log_probability is not None
        policy_q = torch.minimum(
            self.q1(observation, skill, policy_action), self.q2(observation, skill, policy_action)
        )
        actor_loss = (self.alpha.detach() * log_probability - policy_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_probability + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        self.update_count += 1
        return {
            "mine_loss": float(mine_loss.item()),
            "state_mi_bound": float(state_bound.item()),
            "demonstration_mi_bound": float(demo_bound.item()),
            "pseudo_reward_raw_mean": float(raw_reward.mean().item()),
            "pseudo_reward_mean": float(pseudo_reward.mean().item()),
            "demonstration_support_mean": float(demonstration_support.mean().item()),
            "demonstration_progress_reward_mean": float(progress_reward.mean().item()),
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
        }

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(source_parameter, alpha=self.tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "state_mine": self.state_mine.state_dict(),
            "demo_mine": self.demo_mine.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "mine_optimizer": self.mine_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "reward_moments": self.reward_moments.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, Any], *, load_optimizers: bool = True) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.target_q1.load_state_dict(state["target_q1"])
        self.target_q2.load_state_dict(state["target_q2"])
        self.state_mine.load_state_dict(state["state_mine"])
        self.demo_mine.load_state_dict(state["demo_mine"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.reward_moments.load_state_dict(state["reward_moments"])
        self.update_count = int(state.get("update_count", 0))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.q_optimizer.load_state_dict(state["q_optimizer"])
            self.mine_optimizer.load_state_dict(state["mine_optimizer"])
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
