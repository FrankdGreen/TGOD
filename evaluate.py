from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tgod_sd.config import load_config, resolve_output_path
from tgod_sd.trainer import build_components, load_checkpoint, seed_everything
from tgod_sd.trajectory import generate_and_match


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ur5e_pick_place.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "ur5e_pick_place" / "checkpoints" / "latest.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TGOD candidates and select one with Sinkhorn distance.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--prefer-successful",
        action="store_true",
        help="Engineering extension: filter to successful candidates before SD selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, Any] = {}
    if args.candidate_count is not None:
        overrides.setdefault("matching", {})["candidate_count"] = args.candidate_count
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides.setdefault("paths", {})["output_dir"] = args.output_dir
    if args.prefer_successful:
        overrides.setdefault("matching", {})["prefer_successful"] = True
    config = load_config(args.config, overrides)
    seed_everything(int(config["seed"]))
    expert, env, agent, default_output, device = build_components(config)
    load_checkpoint(args.checkpoint, agent, load_optimizers=False)
    output_directory = resolve_output_path(args.output_dir) if args.output_dir else default_output
    print(f"Evaluating checkpoint on {device}: {args.checkpoint}")
    generate_and_match(
        env,
        expert,
        agent,
        config["matching"],
        output_directory,
        seed=int(config["seed"]),
    )
    env.close()


if __name__ == "__main__":
    main()
