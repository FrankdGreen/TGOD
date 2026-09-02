from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import TGODSACAgent
from .env import UR5ePickPlaceEnv
from .expert import ExpertTrajectory, resample_trajectory
from .sinkhorn import SinkhornResult, solve_sinkhorn


@dataclass
class CandidateTrajectory:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    qpos: np.ndarray
    ee_positions: np.ndarray
    cup_positions: np.ndarray
    contacts: np.ndarray
    collisions: np.ndarray
    grasped: np.ndarray
    skill_index: int
    success: bool

    @property
    def matching_features(self) -> np.ndarray:
        return np.concatenate([self.qpos, self.ee_positions, self.cup_positions], axis=1).astype(np.float32)

    def save(self, path: str | Path, sinkhorn_score: float | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        score = np.nan if sinkhorn_score is None else float(sinkhorn_score)
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            next_observations=self.next_observations,
            qpos=self.qpos,
            ee_positions=self.ee_positions,
            cup_positions=self.cup_positions,
            contacts=self.contacts,
            collisions=self.collisions,
            grasped=self.grasped,
            skill_index=np.asarray(self.skill_index, dtype=np.int64),
            success=np.asarray(self.success, dtype=np.bool_),
            sinkhorn_distance=np.asarray(score, dtype=np.float64),
        )


def one_hot_skill(index: int, skill_dim: int) -> np.ndarray:
    if not 0 <= index < skill_dim:
        raise ValueError(f"Skill index {index} is outside [0, {skill_dim}).")
    skill = np.zeros(skill_dim, dtype=np.float32)
    skill[index] = 1.0
    return skill


def rollout_candidate(
    env: UR5ePickPlaceEnv,
    agent: TGODSACAgent,
    *,
    skill_index: int,
    seed: int,
    deterministic: bool,
) -> CandidateTrajectory:
    skill = one_hot_skill(skill_index, agent.skill_dim)
    observation, _ = env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []
    cup_positions: list[np.ndarray] = []
    contacts: list[bool] = []
    collisions: list[bool] = []
    grasped: list[bool] = []
    success = False

    while True:
        action = agent.act(observation, skill, deterministic=deterministic)
        next_observation, _, terminated, truncated, info = env.step(action)
        observations.append(observation.copy())
        actions.append(action.copy())
        next_observations.append(next_observation.copy())
        qpos.append(np.asarray(info["qpos"], dtype=np.float32))
        ee_positions.append(np.asarray(info["ee_pos"], dtype=np.float32))
        cup_positions.append(np.asarray(info["cup_pos"], dtype=np.float32))
        contacts.append(bool(info["contact"]))
        collisions.append(bool(info["collision"]))
        grasped.append(bool(info["grasped"]))
        observation = next_observation
        success = bool(info["success"])
        if terminated or truncated:
            break

    return CandidateTrajectory(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        qpos=np.asarray(qpos, dtype=np.float32),
        ee_positions=np.asarray(ee_positions, dtype=np.float32),
        cup_positions=np.asarray(cup_positions, dtype=np.float32),
        contacts=np.asarray(contacts, dtype=np.bool_),
        collisions=np.asarray(collisions, dtype=np.bool_),
        grasped=np.asarray(grasped, dtype=np.bool_),
        skill_index=skill_index,
        success=success,
    )


def _normalized_for_matching(
    expert: ExpertTrajectory,
    features: np.ndarray,
    matching_config: dict[str, Any],
) -> np.ndarray:
    normalized = expert.normalized_matching_features(
        features,
        qpos_weight=float(matching_config["qpos_weight"]),
        tcp_weight=float(matching_config["tcp_weight"]),
        cup_weight=float(matching_config["cup_weight"]),
        time_weight=float(matching_config["time_weight"]),
    )
    return resample_trajectory(normalized, int(matching_config["resample_points"]))


def pad_terminal_hold(features: np.ndarray, target_length: int) -> np.ndarray:
    """Keep an early-success trajectory on its final state until the expert horizon."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f"features must be a non-empty 2D array, got {features.shape}.")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(features) >= target_length:
        return features
    padding = np.repeat(features[-1:, :], target_length - len(features), axis=0)
    return np.concatenate([features, padding], axis=0)


def score_candidate(
    expert: ExpertTrajectory,
    candidate: CandidateTrajectory,
    matching_config: dict[str, Any],
) -> SinkhornResult:
    expert_values = _normalized_for_matching(expert, expert.matching_features, matching_config)
    candidate_features = candidate.matching_features
    if candidate.success:
        candidate_features = pad_terminal_hold(candidate_features, len(expert))
    candidate_values = _normalized_for_matching(expert, candidate_features, matching_config)
    result = solve_sinkhorn(
        candidate_values,
        expert_values,
        epsilon=float(matching_config["epsilon"]),
        max_iterations=int(matching_config["max_iterations"]),
        tolerance=float(matching_config["tolerance"]),
    )
    return result


def generate_and_match(
    env: UR5ePickPlaceEnv,
    expert: ExpertTrajectory,
    agent: TGODSACAgent,
    matching_config: dict[str, Any],
    output_directory: str | Path,
    *,
    seed: int,
) -> tuple[Path, list[dict[str, Any]]]:
    output_directory = Path(output_directory)
    candidates_directory = output_directory / "candidates"
    candidates_directory.mkdir(parents=True, exist_ok=True)
    candidate_count = int(matching_config["candidate_count"])
    deterministic = bool(matching_config["deterministic_policy"])
    candidate_seed = int(seed + 1_000_000)
    random.seed(candidate_seed)
    np.random.seed(candidate_seed)
    torch.manual_seed(candidate_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(candidate_seed)
    candidates: list[CandidateTrajectory] = []
    scores: list[float] = []
    sinkhorn_results: list[SinkhornResult] = []

    for index in range(candidate_count):
        candidate = rollout_candidate(
            env,
            agent,
            skill_index=index % agent.skill_dim,
            seed=seed + 10_000 + index,
            deterministic=deterministic,
        )
        sinkhorn_result = score_candidate(expert, candidate, matching_config)
        score = sinkhorn_result.distance
        if not sinkhorn_result.converged:
            warnings.warn(
                "Sinkhorn solver reached its iteration limit for candidate "
                f"{index}: marginal_error={sinkhorn_result.marginal_error:.3e}. "
                "The finite score is retained; increase matching.max_iterations or epsilon ",
                RuntimeWarning,
                stacklevel=2,
            )
        candidates.append(candidate)
        scores.append(score)
        sinkhorn_results.append(sinkhorn_result)
        candidate.save(candidates_directory / f"candidate_{index:03d}.npz", score)
        print(
            f"Candidate {index + 1}/{candidate_count}: skill={candidate.skill_index}, "
            f"success={candidate.success}, steps={len(candidate.actions)}, SD={score:.6f}, "
            f"converged={sinkhorn_result.converged}"
        )

    eligible = list(range(candidate_count))
    successful = [index for index, candidate in enumerate(candidates) if candidate.success]
    filtered_to_success = bool(matching_config["prefer_successful"] and successful)
    if filtered_to_success:
        eligible = successful
    selected_index = min(eligible, key=lambda index: scores[index])
    selected_path = output_directory / "selected_trajectory.npz"
    candidates[selected_index].save(selected_path, scores[selected_index])

    records = [
        {
            "candidate_index": index,
            "skill_index": candidate.skill_index,
            "success": candidate.success,
            "steps": int(len(candidate.actions)),
            "sinkhorn_distance": scores[index],
            "sinkhorn_converged": sinkhorn_results[index].converged,
            "sinkhorn_iterations": sinkhorn_results[index].iterations,
            "sinkhorn_marginal_error": sinkhorn_results[index].marginal_error,
            "selected": index == selected_index,
        }
        for index, candidate in enumerate(candidates)
    ]
    score_document = {
        "selected_candidate": selected_index,
        "selected_sinkhorn_distance": scores[selected_index],
        "selected_success": candidates[selected_index].success,
        "prefer_successful": bool(matching_config["prefer_successful"]),
        "candidate_generation_seed": candidate_seed,
        "filtered_to_successful_candidates": filtered_to_success,
        "candidates": records,
    }
    with (output_directory / "candidate_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(score_document, handle, indent=2, ensure_ascii=False)
    print(
        f"Selected candidate {selected_index}: success={candidates[selected_index].success}, "
        f"SD={scores[selected_index]:.6f} -> {selected_path}"
    )
    return selected_path, records
