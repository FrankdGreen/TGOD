from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from tgod_sd.agent import TGODSACAgent
from tgod_sd.config import load_config, resolve_input_path, select_device
from tgod_sd.env import UR5ePickPlaceEnv
from tgod_sd.expert import ExpertTrajectory
from tgod_sd.schema import OBS_DIM
from tgod_sd.sinkhorn import sinkhorn_distance
from tgod_sd.trajectory import one_hot_skill


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "ur5e_pick_place.yaml")
    config = copy.deepcopy(config)
    config["device"] = "cpu"
    config["network"]["hidden_dims"] = [32, 32]
    config["network"]["mine_hidden_dims"] = [32, 32]
    config["tgod"]["num_skills"] = 4
    scene = resolve_input_path(config["paths"]["scene_xml"], kind="scene")
    expert_directory = resolve_input_path(config["paths"]["expert_dir"], kind="expert")
    expert = ExpertTrajectory.load(expert_directory)
    env = UR5ePickPlaceEnv(scene, expert, config["environment"])
    observation, _ = env.reset(seed=0)
    next_observation, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert observation.shape == (OBS_DIM,) and next_observation.shape == (OBS_DIM,)
    assert reward == 0.0 and not terminated and not truncated
    assert np.isfinite(next_observation).all() and not info["success"]

    agent = TGODSACAgent(OBS_DIM, 4, expert.relation_dim, config, select_device(config["device"]))
    rng = np.random.default_rng(0)
    batch_size = 8
    observations = np.repeat(observation[None, :], batch_size, axis=0)
    observations += np.asarray(
        rng.normal(0.0, 0.001, size=observations.shape), dtype=np.float32
    )
    next_observations = np.repeat(next_observation[None, :], batch_size, axis=0)
    skills = np.stack([one_hot_skill(index % agent.skill_dim, agent.skill_dim) for index in range(batch_size)])
    relations = []
    for index in range(batch_size):
        progress = index / (batch_size - 1)
        observations[index, -1] = progress
        relations.append(expert.relation_feature(observations[index], progress))
    batch = {
        "observation": observations.astype(np.float32),
        "action": rng.uniform(-1.0, 1.0, size=(batch_size, 4)).astype(np.float32),
        "next_observation": next_observations.astype(np.float32),
        "skill": skills.astype(np.float32),
        "relation": np.asarray(relations, dtype=np.float32),
        "terminal": np.zeros((batch_size, 1), dtype=np.float32),
    }
    metrics = agent.update(batch)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"Non-finite update metrics: {metrics}")

    reference = expert.matching_features[::10]
    same = sinkhorn_distance(reference, reference, epsilon=0.05)
    shifted = sinkhorn_distance(reference + 0.5, reference, epsilon=0.05)
    if not same < shifted:
        raise AssertionError(f"Sinkhorn sanity check failed: same={same}, shifted={shifted}")
    env.close()
    print(
        f"Smoke test passed: scene nq={env.model.nq}, expert={len(expert)} frames, "
        f"agent update finite, Sinkhorn same={same:.6f} < shifted={shifted:.6f}."
    )


if __name__ == "__main__":
    main()
