from __future__ import annotations

import copy
from typing import Any

import numpy as np


class ReplayBuffer:
    _ARRAY_NAMES = (
        "observations", "actions", "next_observations", "skills",
        "relations", "next_relations", "terminals",
    )

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
        self.next_relations = np.empty((capacity, relation_dim), dtype=np.float32)
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
        next_relation: np.ndarray,
        terminal: bool,
    ) -> None:
        index = self._index
        self.observations[index] = observation
        self.actions[index] = action
        self.next_observations[index] = next_observation
        self.skills[index] = skill
        self.relations[index] = relation
        self.next_relations[index] = next_relation
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
            "next_relation": self.next_relations[indices],
            "terminal": self.terminals[indices],
        }

    def state_dict(self) -> dict[str, Any]:
        """Snapshot occupied slots in their ring order, without uninitialized memory."""
        return {
            "capacity": self.capacity,
            "index": self._index,
            "size": self._size,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            **{name: getattr(self, name)[:self._size].copy() for name in self._ARRAY_NAMES},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a full replay snapshot; reject incompatible or incomplete data."""
        required = {"capacity", "index", "size", "rng_state", *self._ARRAY_NAMES}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Replay checkpoint is incomplete; missing keys: {sorted(missing)}")
        for name in ("capacity", "index", "size"):
            value = state[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Replay checkpoint {name} must be an integer.")
        capacity, index, size = (int(state[name]) for name in ("capacity", "index", "size"))
        if capacity != self.capacity:
            raise ValueError(
                f"Replay checkpoint capacity {capacity} differs from configured capacity {self.capacity}. "
                "Use the checkpoint replay_size to restore its data."
            )
        if not 0 <= size <= capacity or not 0 <= index < capacity:
            raise ValueError("Replay checkpoint size or index is outside its capacity.")
        if size < capacity and index != size:
            raise ValueError("Partially filled replay checkpoint must have index equal to size.")

        # Validate and copy everything before mutating the live buffer.
        arrays: dict[str, np.ndarray] = {}
        for name in self._ARRAY_NAMES:
            value = state[name]
            expected_shape = (size, *getattr(self, name).shape[1:])
            if not isinstance(value, np.ndarray) or value.shape != expected_shape:
                raise ValueError(f"Replay checkpoint {name} must have shape {expected_shape}.")
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Replay checkpoint {name} must contain finite float32 values.")
            arrays[name] = value.copy()
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("Replay checkpoint contains an invalid RNG state.") from error
        for name, value in arrays.items():
            getattr(self, name)[:size] = value
        self._index = index
        self._size = size
        self._rng = restored_rng

    def skill_counts(self) -> np.ndarray:
        """Count occupied transitions for each discrete (one-hot) skill."""
        if self._size == 0:
            return np.zeros(self.skills.shape[1], dtype=np.int64)
        return np.bincount(
            np.argmax(self.skills[:self._size], axis=1), minlength=self.skills.shape[1]
        ).astype(np.int64)
