from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tgod_sd.config import load_config
from tgod_sd.trainer import train


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "ur5e_pick_place.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGOD-SD on the UR5e white-cup pick-and-place task.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML configuration file.")
    parser.add_argument("--episodes", type=int, help="Override training.episodes.")
    parser.add_argument("--candidate-count", type=int, help="Override matching.candidate_count.")
    parser.add_argument("--device", help="Override device (auto/cpu/cuda/cuda:0).")
    parser.add_argument("--output-dir", help="Override paths.output_dir.")
    parser.add_argument("--resume", help="Resume policy/optimizers from a checkpoint; replay is refilled.")
    parser.add_argument("--no-match", action="store_true", help="Skip candidate generation after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, Any] = {}
    if args.episodes is not None:
        overrides.setdefault("training", {})["episodes"] = args.episodes
    if args.candidate_count is not None:
        overrides.setdefault("matching", {})["candidate_count"] = args.candidate_count
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides.setdefault("paths", {})["output_dir"] = args.output_dir
    if args.no_match:
        overrides.setdefault("training", {})["match_after_training"] = False
    config = load_config(args.config, overrides)
    train(config, args.resume)


if __name__ == "__main__":
    main()
