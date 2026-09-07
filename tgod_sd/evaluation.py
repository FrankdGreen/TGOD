from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch


class EpisodeDiagnostics:
    """Task diagnostics only; these signals are never added to the reward.

    Event times are one-based environment steps. An event that never occurred
    is represented by ``None``. Action saturation counts individual normalized
    action components with absolute value >= 0.95, not whole time steps.
    ``update(None, reset_info)`` optionally includes the initial cup height.
    """

    def __init__(self, success_radius: float = 0.06, success_z_max: float = 0.14) -> None:
        self.success_radius = float(success_radius)
        self.success_z_max = float(success_z_max)
        self.steps = 0
        self.success = False
        self.ever_grasped = False
        self.cup_lifted = False
        self.goal_reached = False
        self.placed = False
        self.first_grasp_step: int | None = None
        self.first_lift_step: int | None = None
        self.first_goal_step: int | None = None
        self.first_placed_step: int | None = None
        self.contacts = 0
        self.collisions = 0
        self.final_goal_distance: float | None = None
        self.final_cup_height: float | None = None
        self.max_cup_height: float | None = None
        self._action_components = 0
        self._saturated_components = 0

    def update(self, action: np.ndarray | None, info: dict[str, Any]) -> None:
        if action is not None:
            self.steps += 1
            values = np.asarray(action)
            self._action_components += int(values.size)
            self._saturated_components += int(np.count_nonzero(np.abs(values) >= 0.95))
            self.contacts += int(bool(info.get("contact", False)))
            self.collisions += int(bool(info.get("collision", False)))
        event_step = int(info.get("step", self.steps))

        distance = info.get("cup_goal_distance")
        self.final_goal_distance = float(distance) if distance is not None else None
        cup_position = info.get("cup_pos")
        self.final_cup_height = float(cup_position[2]) if cup_position is not None else None
        if self.final_cup_height is not None:
            self.max_cup_height = (
                self.final_cup_height
                if self.max_cup_height is None
                else max(self.max_cup_height, self.final_cup_height)
            )

        grasped = bool(info.get("ever_grasped", False) or info.get("grasped", False))
        lifted = bool(info.get("cup_lifted", False))
        goal_reached = bool(
            self.final_goal_distance is not None
            and self.final_goal_distance < self.success_radius
            and self.final_cup_height is not None
            and self.final_cup_height <= self.success_z_max
        )
        placed = bool(info.get("placed", False))
        for happened, first_field in (
            (grasped, "first_grasp_step"),
            (lifted, "first_lift_step"),
            (goal_reached, "first_goal_step"),
            (placed, "first_placed_step"),
        ):
            if happened and getattr(self, first_field) is None:
                setattr(self, first_field, event_step)
        self.success = self.success or bool(info.get("success", False))
        self.ever_grasped = self.ever_grasped or grasped
        self.cup_lifted = self.cup_lifted or lifted
        self.goal_reached = self.goal_reached or goal_reached
        self.placed = self.placed or placed

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "success": self.success,
            "ever_grasped": self.ever_grasped,
            "cup_lifted": self.cup_lifted,
            "goal_reached": self.goal_reached,
            "placed": self.placed,
            "first_grasp_step": self.first_grasp_step,
            "first_lift_step": self.first_lift_step,
            "first_goal_step": self.first_goal_step,
            "first_placed_step": self.first_placed_step,
            "contacts": self.contacts,
            "collisions": self.collisions,
            "final_goal_distance": self.final_goal_distance,
            "final_cup_height": self.final_cup_height,
            "max_cup_height": self.max_cup_height,
            "action_saturation_fraction": (
                self._saturated_components / self._action_components
                if self._action_components else 0.0
            ),
        }


@contextmanager
def _isolated_policy_rng(actor: torch.nn.Module) -> Iterator[None]:
    """Restore the caller's random streams and actor mode, including on failure."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = actor.training
    try:
        actor.eval()
        with torch.inference_mode():
            yield
    finally:
        actor.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(episodes)
    distances = [
        float(episode["final_goal_distance"])
        for episode in episodes
        if episode["final_goal_distance"] is not None
    ]
    result: dict[str, Any] = {"episode_count": count}
    for field, rate_name in (
        ("success", "success_rate"),
        ("ever_grasped", "grasp_rate"),
        ("cup_lifted", "lift_rate"),
        ("goal_reached", "goal_reached_rate"),
        ("placed", "placed_rate"),
    ):
        result[rate_name] = sum(bool(episode[field]) for episode in episodes) / count
    result["mean_final_goal_distance"] = sum(distances) / len(distances) if distances else None
    return result


def evaluate_policy(
    env: Any,
    agent: Any,
    *,
    seed: int,
    episodes_per_skill: int = 10,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Evaluate a frozen policy in a caller-owned, separate environment.

    Each checkpoint sees the same reset and policy-sampling seeds. Episode j of
    skill k uses ``seed + j * skill_dim + k`` modulo 2**32. Evaluation does not
    update the learner, calculate Sinkhorn scores, or change training RNG state.
    The caller owns environment lifetime and must not pass the training env.
    """
    if isinstance(episodes_per_skill, bool) or not isinstance(episodes_per_skill, (int, np.integer)):
        raise ValueError("episodes_per_skill must be a positive integer")
    if episodes_per_skill < 1 or int(agent.skill_dim) < 1:
        raise ValueError("episodes_per_skill and skill_dim must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    max_steps = int(env.max_episode_steps)
    if max_steps < 1:
        raise ValueError("env.max_episode_steps must be positive")

    episodes: list[dict[str, Any]] = []
    skill_dim = int(agent.skill_dim)
    with _isolated_policy_rng(agent.actor):
        for skill_index in range(skill_dim):
            skill = np.zeros(skill_dim, dtype=np.float32)
            skill[skill_index] = 1.0
            for episode_index in range(int(episodes_per_skill)):
                episode_seed = (int(seed) + episode_index * skill_dim + skill_index) % (2**32)
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                observation, info = env.reset(seed=episode_seed)
                diagnostics = EpisodeDiagnostics(env.success_radius, env.success_z_max)
                diagnostics.update(None, info)
                terminated = truncated = False
                for _ in range(max_steps):
                    action = agent.act(observation, skill, deterministic=deterministic)
                    observation, _, terminated, truncated, info = env.step(action)
                    diagnostics.update(action, info)
                    if terminated or truncated:
                        break
                if not terminated and not truncated:
                    truncated = True
                episodes.append({
                    "skill_index": skill_index,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    **diagnostics.as_dict(),
                })

    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "episodes_per_skill": int(episodes_per_skill),
        **_summary(episodes),
        "per_skill": {
            str(skill_index): _summary([
                episode for episode in episodes if episode["skill_index"] == skill_index
            ])
            for skill_index in range(skill_dim)
        },
        "episodes": episodes,
    }
