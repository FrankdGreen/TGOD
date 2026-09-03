from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    config = copy.deepcopy(loaded)
    if overrides:
        _merge(config, overrides)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    validate_config(config)
    return config


def resolve_input_path(value: str | Path, *, kind: str) -> Path:
    """Resolve an input and support both the final sibling layout and build layout."""
    raw = Path(os.path.expandvars(os.path.expanduser(str(value))))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((PROJECT_ROOT / raw).resolve())

        # During development the project may temporarily sit inside SAC_ur5e.
        source_root = PROJECT_ROOT.parent
        parts = list(raw.parts)
        if source_root.name.lower() == "sac_ur5e":
            if "SAC_ur5e" in parts:
                index = parts.index("SAC_ur5e")
                candidates.append(source_root.joinpath(*parts[index + 1 :]).resolve())
            if kind == "scene":
                candidates.append((source_root / "universal_robots_ur5e" / "scene.xml").resolve())
            elif kind == "expert":
                candidates.append((source_root / "data" / "similar_expert").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = "\n  - ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Could not resolve {kind} path. Tried:\n  - {rendered}")


def resolve_output_path(value: str | Path) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def validate_config(config: Mapping[str, Any]) -> None:
    required_sections = ("paths", "environment", "network", "tgod", "sac", "training", "matching")
    missing = [section for section in required_sections if not isinstance(config.get(section), Mapping)]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    integer_values = {
        "environment.max_episode_steps": config["environment"]["max_episode_steps"],
        "environment.frame_skip": config["environment"]["frame_skip"],
        "environment.ik_iterations": config["environment"]["ik_iterations"],
        "environment.grasp_confirm_steps": config["environment"]["grasp_confirm_steps"],
        "environment.release_confirm_steps": config["environment"]["release_confirm_steps"],
        "environment.regrasp_cooldown_steps": config["environment"]["regrasp_cooldown_steps"],
        "tgod.num_skills": config["tgod"]["num_skills"],
        "sac.batch_size": config["sac"]["batch_size"],
        "sac.replay_size": config["sac"]["replay_size"],
        "sac.update_after": config["sac"]["update_after"],
        "sac.update_every": config["sac"]["update_every"],
        "sac.gradient_steps": config["sac"]["gradient_steps"],
        "training.episodes": config["training"]["episodes"],
        "training.log_every_episodes": config["training"]["log_every_episodes"],
        "training.checkpoint_every_episodes": config["training"]["checkpoint_every_episodes"],
        "matching.candidate_count": config["matching"]["candidate_count"],
        "matching.resample_points": config["matching"]["resample_points"],
        "matching.max_iterations": config["matching"]["max_iterations"],
    }
    invalid_integers = [
        name
        for name, value in integer_values.items()
        if isinstance(value, bool)
        or not float(value).is_integer()
        or int(value) <= 0
    ]
    if invalid_integers:
        raise ValueError(f"Configuration values must be positive integers: {invalid_integers}")
    if int(config["sac"]["batch_size"]) < 2:
        raise ValueError("sac.batch_size must be at least 2 for MINE negative sampling.")
    minimum_replay = max(int(config["sac"]["batch_size"]), int(config["sac"]["update_after"]))
    if int(config["sac"]["replay_size"]) < minimum_replay:
        raise ValueError(
            "sac.replay_size must be at least max(sac.batch_size, sac.update_after)."
        )
    random_steps = config["sac"]["random_steps"]
    if (
        isinstance(random_steps, bool)
        or not float(random_steps).is_integer()
        or int(random_steps) < 0
    ):
        raise ValueError("sac.random_steps cannot be negative.")

    positive_values = {
        "environment.max_ee_step": config["environment"]["max_ee_step"],
        "environment.ik_damping": config["environment"]["ik_damping"],
        "environment.grasp_radius": config["environment"]["grasp_radius"],
        "environment.success_radius": config["environment"]["success_radius"],
        "sac.learning_rate": config["sac"]["learning_rate"],
        "sac.alpha_learning_rate": config["sac"]["alpha_learning_rate"],
        "tgod.mine_learning_rate": config["tgod"]["mine_learning_rate"],
        "matching.epsilon": config["matching"]["epsilon"],
        "matching.tolerance": config["matching"]["tolerance"],
    }
    invalid_positive = [name for name, value in positive_values.items() if float(value) <= 0]
    if invalid_positive:
        raise ValueError(f"Configuration values must be positive: {invalid_positive}")
    for name in ("gamma", "tau"):
        value = float(config["sac"][name])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"sac.{name} must be in (0, 1].")
    nonnegative_values = {
        "tgod.state_mi_weight": config["tgod"]["state_mi_weight"],
        "tgod.demonstration_mi_weight": config["tgod"]["demonstration_mi_weight"],
        "tgod.demonstration_support_weight": config["tgod"]["demonstration_support_weight"],
        "tgod.demonstration_progress_weight": config["tgod"]["demonstration_progress_weight"],
        "matching.qpos_weight": config["matching"]["qpos_weight"],
        "matching.tcp_weight": config["matching"]["tcp_weight"],
        "matching.cup_weight": config["matching"]["cup_weight"],
        "matching.time_weight": config["matching"]["time_weight"],
    }
    invalid_nonnegative = [
        name for name, value in nonnegative_values.items() if float(value) < 0
    ]
    if invalid_nonnegative:
        raise ValueError(f"Configuration values cannot be negative: {invalid_nonnegative}")
    if not any(float(config["matching"][name]) > 0 for name in ("qpos_weight", "tcp_weight", "cup_weight", "time_weight")):
        raise ValueError("At least one matching feature weight must be positive.")
    for name in ("hidden_dims", "mine_hidden_dims"):
        dimensions = config["network"][name]
        if not isinstance(dimensions, list) or not dimensions or any(
            isinstance(value, bool) or not float(value).is_integer() or int(value) <= 0
            for value in dimensions
        ):
            raise ValueError(f"network.{name} must be a non-empty list of positive integers.")


def select_device(requested: str) -> str:
    import torch

    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {requested}")
    return requested
