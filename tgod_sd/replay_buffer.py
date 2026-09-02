from __future__ import annotations

from typing import Any

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        skill_dim: int,
        relation_dim: int,
        seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.skills = np.empty((capacity, skill_dim), dtype=np.float32)
        self.relations = np.empty((capacity, relation_dim), dtype=np.float32)
        self.terminals = np.empty((capacity, 1), dtype=np.float32)
        self._index = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        skill: np.ndarray,
        relation: np.ndarray,
        terminal: bool,
    ) -> None:
        index = self._index
        self.observations[index] = observation
        self.actions[index] = action
        self.next_observations[index] = next_observation
        self.skills[index] = skill
        self.relations[index] = relation
        self.terminals[index, 0] = float(terminal)
        self._index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self._size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from replay size {self._size}.")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observation": self.observations[indices],
            "action": self.actions[indices],
            "next_observation": self.next_observations[indices],
            "skill": self.skills[indices],
            "relation": self.relations[indices],
            "terminal": self.terminals[indices],
        }

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "index": self._index, "size": self._size}
