from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tgod_sd.checkpoint_config import evaluation_config
from tgod_sd.config import resolve_output_path
from tgod_sd.evaluation import EpisodeDiagnostics
from tgod_sd.trainer import build_components, seed_everything
from tgod_sd.trajectory import SELECTION_RULE, generate_and_match


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "ur5e_pick_place" / "checkpoints" / "latest.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TGOD candidates and select one with Sinkhorn distance.")
    parser.add_argument(
        "--config", default=None,
        help="Optional YAML for matching, resource paths, and device; task/model must match the checkpoint.",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--seed", type=int, help="Override candidate-generation seed.")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument(
        "--skill-index",
        type=int,
        help="Generate every candidate with one skill instead of cycling all skills.",
    )
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        help="Use the actor mean action instead of sampling its distribution.",
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def default_evaluation_output(checkpoint_path: Path) -> Path:
    run_path = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    location_id = hashlib.sha256(str(run_path).encode("utf-8")).hexdigest()[:8]
    return PROJECT_ROOT / "outputs" / "evaluations" / f"{run_path.name}_{location_id}" / checkpoint_path.stem


def summarize_saved_candidates(
    output_directory: Path, records: list[dict[str, Any]], config: dict[str, Any],
    initial_cup_height: float,
) -> dict[str, Any]:
    """Diagnose the exact scored candidates, without generating new trajectories."""
    environment = config["environment"]
    goal = np.asarray([*environment["blue_mat_center"], initial_cup_height])
    for record in records:
        path = output_directory / "candidates" / f"candidate_{int(record['candidate_index']):03d}.npz"
        with np.load(path, allow_pickle=False) as candidate:
            cups = candidate["cup_positions"]
            grasped = candidate["grasped"]
            distances = np.linalg.norm(cups - goal, axis=1)
            diagnostics = EpisodeDiagnostics(environment["success_radius"], environment["success_z_max"])
            for step, action in enumerate(candidate["actions"]):
                success = bool(record["success"]) and step == len(cups) - 1
                diagnostics.update(action, {
                    "step": step + 1, "cup_pos": cups[step],
                    "cup_goal_distance": distances[step],
                    "grasped": bool(grasped[step]) or success,
                    "cup_lifted": bool(grasped[step] and cups[step, 2] >= environment["minimum_lift_height"]) or success,
                    "placed": success, "success": success,
                    "contact": candidate["contacts"][step],
                    "collision": candidate["collisions"][step],
                })
            record.update(diagnostics.as_dict())

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episode_count": len(items),
            **{
                name: float(np.mean([item[key] for item in items]))
                for name, key in (
                    ("success_rate", "success"), ("grasp_rate", "ever_grasped"),
                    ("lift_rate", "cup_lifted"), ("goal_reached_rate", "goal_reached"),
                    ("placed_rate", "placed"), ("mean_final_goal_distance", "final_goal_distance"),
                    ("mean_final_cup_height", "final_cup_height"),
                )
            },
        }

    return {
        **aggregate(records),
        "per_skill": {
            str(skill): aggregate([item for item in records if int(item["skill_index"]) == skill])
            for skill in sorted({int(item["skill_index"]) for item in records})
        },
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; expected format_version=1.")
    # Old checkpoints may carry a task-success filter. Evaluation now always
    # follows the paper's all-candidate SD rule, including with legacy settings.
    overrides: dict[str, Any] = {"matching": {"prefer_successful": False}}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.candidate_count is not None:
        overrides.setdefault("matching", {})["candidate_count"] = args.candidate_count
    if args.skill_index is not None:
        overrides.setdefault("matching", {})["skill_index"] = args.skill_index
    if args.deterministic_policy:
        overrides.setdefault("matching", {})["deterministic_policy"] = True
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides.setdefault("paths", {})["output_dir"] = args.output_dir
    output_directory = (
        resolve_output_path(args.output_dir) if args.output_dir
        else default_evaluation_output(checkpoint_path)
    )
    overrides.setdefault("paths", {})["output_dir"] = str(output_directory)
    config = evaluation_config(checkpoint, args.config, overrides)
    seed_everything(int(config["seed"]))
    expert, env, agent, _, device = build_components(config)
    try:
        agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
        output_directory.mkdir(parents=True, exist_ok=True)
        with (output_directory / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        provenance = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_episode": checkpoint.get("episode"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "config_source": "checkpoint",
            "evaluation_override_config_path": str(Path(args.config).resolve()) if args.config else None,
            "training_config_path": checkpoint["config"].get("_config_path"),
            "evaluation_seed": int(config["seed"]),
            "actual_device": str(device),
            "selection_rule": SELECTION_RULE,
            "legacy_prefer_successful_ignored": bool(
                checkpoint["config"].get("matching", {}).get("prefer_successful", False)
            ),
            "actual_config": config,
        }
        manifest_path = output_directory / "evaluation_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, ensure_ascii=False)
        print(
            f"Evaluating checkpoint on {device}: {checkpoint_path}; "
            f"episode={checkpoint.get('episode')}, max_ee_step={config['environment']['max_ee_step']}"
        )
        _, records = generate_and_match(
            env, expert, agent, config["matching"], output_directory,
            seed=int(config["seed"]),
        )
        summary = summarize_saved_candidates(
            output_directory, records, config, float(expert.cup_initial_position[2])
        )
        scores_path = output_directory / "candidate_scores.json"
        with scores_path.open("r", encoding="utf-8") as handle:
            scores = json.load(handle)
        scores.update(
            provenance=provenance, summary=summary, candidates=records,
            selection_rule=provenance["selection_rule"],
            legacy_prefer_successful_ignored=provenance["legacy_prefer_successful_ignored"],
        )
        with scores_path.open("w", encoding="utf-8") as handle:
            json.dump(scores, handle, indent=2, ensure_ascii=False)
        print(
            f"Evaluation: success={summary['success_rate']:.1%}, grasp={summary['grasp_rate']:.1%}, "
            f"lift={summary['lift_rate']:.1%}, goal={summary['goal_reached_rate']:.1%}; "
            f"manifest={manifest_path}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
