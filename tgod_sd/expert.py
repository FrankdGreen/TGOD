from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .schema import matching_feature_from_observation


RELATION_PROXIMITY_INDEX = 36


def resample_trajectory(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"Trajectory must be a non-empty 2D array, got {values.shape}.")
    if points <= 0:
        raise ValueError("points must be positive")
    if len(values) == 1:
        return np.repeat(values, points, axis=0).astype(np.float32)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    columns = [np.interp(target, source, values[:, index]) for index in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


@dataclass(frozen=True)
class ExpertTrajectory:
    tcp_pose: np.ndarray
    tcp_velocity: np.ndarray
    qpos: np.ndarray
    cup_position: np.ndarray
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    red_mat_center: np.ndarray
    blue_mat_center: np.ndarray
    cup_initial_position: np.ndarray

    @classmethod
    def load(cls, directory: str | Path) -> "ExpertTrajectory":
        directory = Path(directory)
        required = {
            "expert_demo.npy",
            "expert_qpos.npy",
            "expert_cup.npy",
            "expert_initial_state.npz",
        }
        missing = sorted(name for name in required if not (directory / name).is_file())
        if missing:
            raise FileNotFoundError(f"Missing expert files in {directory}: {missing}")

        demo = np.asarray(np.load(directory / "expert_demo.npy"), dtype=np.float32)
        qpos = np.asarray(np.load(directory / "expert_qpos.npy"), dtype=np.float32)
        cup = np.asarray(np.load(directory / "expert_cup.npy"), dtype=np.float32)
        with np.load(directory / "expert_initial_state.npz") as initial:
            initial_values = {key: np.asarray(initial[key], dtype=np.float32) for key in initial.files}

        if demo.ndim != 2 or demo.shape[1] != 12 or len(demo) == 0:
            raise ValueError(f"expert_demo.npy must have shape (T, 12), got {demo.shape}.")
        if qpos.shape != (len(demo), 6):
            raise ValueError(f"expert_qpos.npy must have shape ({len(demo)}, 6), got {qpos.shape}.")
        if cup.shape != (len(demo), 3):
            raise ValueError(f"expert_cup.npy must have shape ({len(demo)}, 3), got {cup.shape}.")
        for name, value in (("expert_demo", demo), ("expert_qpos", qpos), ("expert_cup", cup)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity.")

        expected_initial = {
            "qpos": (6,),
            "qvel": (6,),
            "red_mat_center": (2,),
            "blue_mat_center": (2,),
            "cup_initial_position": (3,),
        }
        for name, shape in expected_initial.items():
            if name not in initial_values or initial_values[name].shape != shape:
                actual = None if name not in initial_values else initial_values[name].shape
                raise ValueError(f"expert_initial_state.npz[{name!r}] must have shape {shape}, got {actual}.")
            if not np.isfinite(initial_values[name]).all():
                raise ValueError(f"expert_initial_state.npz[{name!r}] contains NaN or infinity.")

        return cls(
            tcp_pose=demo[:, :6].copy(),
            tcp_velocity=demo[:, 6:].copy(),
            qpos=qpos.copy(),
            cup_position=cup.copy(),
            initial_qpos=initial_values["qpos"].copy(),
            initial_qvel=initial_values["qvel"].copy(),
            red_mat_center=initial_values["red_mat_center"].copy(),
            blue_mat_center=initial_values["blue_mat_center"].copy(),
            cup_initial_position=initial_values["cup_initial_position"].copy(),
        )

    def __len__(self) -> int:
        return len(self.qpos)

    @cached_property
    def matching_features(self) -> np.ndarray:
        return np.concatenate([self.qpos, self.tcp_pose[:, :3], self.cup_position], axis=1).astype(np.float32)

    @cached_property
    def feature_mean(self) -> np.ndarray:
        return self.matching_features.mean(axis=0).astype(np.float32)

    @cached_property
    def feature_scale(self) -> np.ndarray:
        empirical = self.matching_features.std(axis=0)
        floor = np.asarray([0.10] * 6 + [0.05] * 6, dtype=np.float32)
        return np.maximum(empirical, floor).astype(np.float32)

    def feature_at(self, progress: float) -> np.ndarray:
        position = float(np.clip(progress, 0.0, 1.0)) * (len(self) - 1)
        lower = int(np.floor(position))
        upper = min(lower + 1, len(self) - 1)
        fraction = position - lower
        return ((1.0 - fraction) * self.matching_features[lower] + fraction * self.matching_features[upper]).astype(
            np.float32
        )

    def relation_feature(self, observation: np.ndarray, progress: float) -> np.ndarray:
        candidate = matching_feature_from_observation(observation)
        target = self.feature_at(progress)
        candidate_normalized = (candidate - self.feature_mean) / self.feature_scale
        target_normalized = (target - self.feature_mean) / self.feature_scale
        delta = candidate_normalized - target_normalized
        proximity = np.exp(-0.5 * np.mean(np.square(delta), keepdims=True))
        phase = np.asarray(
            [np.sin(2.0 * np.pi * progress), np.cos(2.0 * np.pi * progress)], dtype=np.float32
        )
        return np.concatenate([candidate_normalized, target_normalized, delta, proximity, phase]).astype(np.float32)

    @property
    def relation_dim(self) -> int:
        return 12 * 3 + 3

    def normalized_matching_features(
        self,
        candidate_features: np.ndarray,
        *,
        qpos_weight: float,
        tcp_weight: float,
        cup_weight: float,
        time_weight: float,
    ) -> np.ndarray:
        candidate_features = np.asarray(candidate_features, dtype=np.float32)
        if candidate_features.ndim != 2 or candidate_features.shape[1] != 12:
            raise ValueError(f"Candidate matching features must have shape (T, 12), got {candidate_features.shape}.")
        normalized = (candidate_features - self.feature_mean) / self.feature_scale
        weights = np.sqrt(
            np.asarray([qpos_weight] * 6 + [tcp_weight] * 3 + [cup_weight] * 3, dtype=np.float32)
        )
        normalized = normalized * weights
        if time_weight > 0:
            time = np.linspace(0.0, 1.0, len(normalized), dtype=np.float32)[:, None]
            normalized = np.concatenate([normalized, time * np.sqrt(time_weight)], axis=1)
        return normalized.astype(np.float32)
