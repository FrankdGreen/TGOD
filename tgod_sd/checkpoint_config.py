from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .config import PROJECT_ROOT, _merge, load_config, validate_config


def assert_checkpoint_compatible(
    saved_config: Mapping[str, Any], current_config: Mapping[str, Any]
) -> None:
    """Reject silent changes to the task, action scale, or model architecture."""
    if not isinstance(saved_config, Mapping):
        raise ValueError("Checkpoint has no saved configuration; cannot verify task compatibility.")
    differences: list[str] = []
    for section in ("environment", "network"):
        old, new = saved_config.get(section), current_config.get(section)
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            differences.append(f"{section}: missing configuration section")
            continue
        for key in sorted(old.keys() | new.keys()):
            if old.get(key) != new.get(key):
                differences.append(f"{section}.{key}: checkpoint={old.get(key)!r}, requested={new.get(key)!r}")
    old_tgod, new_tgod = saved_config.get("tgod"), current_config.get("tgod")
    old_skills = old_tgod.get("num_skills") if isinstance(old_tgod, Mapping) else None
    new_skills = new_tgod.get("num_skills") if isinstance(new_tgod, Mapping) else None
    if old_skills is None or old_skills != new_skills:
        differences.append(f"tgod.num_skills: checkpoint={old_skills!r}, requested={new_skills!r}")
    if differences:
        raise ValueError("Configuration is incompatible with checkpoint:\n  " + "\n  ".join(differences))


def evaluation_config(
    checkpoint: Mapping[str, Any],
    explicit_config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore checkpoint settings, permitting matching and resource overrides only.

    CLI overrides such as the evaluation seed are applied after the optional YAML.
    Training settings and the checkpoint's original dictionary remain untouched.
    """
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError(
            "Checkpoint has no saved configuration. Use a checkpoint containing its actual "
            "training config; evaluation will not fall back to the default YAML."
        )
    config = copy.deepcopy(dict(saved))
    if explicit_config_path is not None:
        requested = load_config(explicit_config_path)
        assert_checkpoint_compatible(saved, requested)
        for section in ("matching", "paths"):
            config.setdefault(section, {}).update(copy.deepcopy(requested[section]))
        config["device"] = requested["device"]
    if overrides:
        _merge(config, overrides)
    config["_project_root"] = str(PROJECT_ROOT)
    config["_config_path"] = (
        str(Path(explicit_config_path).expanduser().resolve())
        if explicit_config_path is not None else "<checkpoint-embedded>"
    )
    try:
        for name in ("seed", "device"):
            if name not in config:
                raise KeyError(name)
        validate_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint evaluation configuration is incomplete or invalid: {exc}") from exc
    return config
