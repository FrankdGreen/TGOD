from __future__ import annotations

import numpy as np


QPOS = slice(0, 6)
QVEL = slice(6, 12)
EE_POS = slice(12, 15)
CUP_POS = slice(15, 18)
EE_TO_CUP = slice(18, 21)
CUP_TO_GOAL = slice(21, 24)
FLAGS = slice(24, 26)
PROGRESS = 26
OBS_DIM = 27

OBSERVATION_SCALE = np.asarray(
    [np.pi] * 6 + [5.0] * 6 + [1.0] * 12 + [1.0, 1.0, 1.0],
    dtype=np.float32,
)


def validate_observation(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape[-1] != OBS_DIM:
        raise ValueError(f"Expected observation dimension {OBS_DIM}, got {observation.shape}.")
    return observation


def normalize_observation(observation: np.ndarray) -> np.ndarray:
    return validate_observation(observation) / OBSERVATION_SCALE


def matching_feature_from_observation(observation: np.ndarray) -> np.ndarray:
    observation = validate_observation(observation)
    return np.concatenate(
        [observation[..., QPOS], observation[..., EE_POS], observation[..., CUP_POS]],
        axis=-1,
    ).astype(np.float32)
