from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tgod_sd.config import load_config, resolve_input_path
from tgod_sd.env import UR5ePickPlaceEnv
from tgod_sd.expert import ExpertTrajectory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ur5e_pick_place.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a selected TGOD-SD q-pos trajectory in MuJoCo.")
    parser.add_argument("trajectory", help="selected_trajectory.npz or another candidate NPZ.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    config = load_config(args.config)
    scene = resolve_input_path(config["paths"]["scene_xml"], kind="scene")
    expert_directory = resolve_input_path(config["paths"]["expert_dir"], kind="expert")
    expert = ExpertTrajectory.load(expert_directory)
    env = UR5ePickPlaceEnv(
        scene,
        expert,
        config["environment"],
        render_mode="human" if args.render else None,
    )
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        if "qpos" not in trajectory or "cup_positions" not in trajectory:
            raise KeyError("Trajectory NPZ must contain qpos and cup_positions arrays.")
        qpos = np.asarray(trajectory["qpos"], dtype=np.float32)
        cup_positions = np.asarray(trajectory["cup_positions"], dtype=np.float32)
        grasped = (
            np.asarray(trajectory["grasped"], dtype=np.bool_)
            if "grasped" in trajectory
            else np.zeros(len(qpos), dtype=np.bool_)
        )
        saved_success = bool(trajectory["success"].item()) if "success" in trajectory else False
        saved_score = float(trajectory["sinkhorn_distance"].item()) if "sinkhorn_distance" in trajectory else np.nan
    if qpos.ndim != 2 or qpos.shape[1] != 6 or cup_positions.shape != (len(qpos), 3):
        raise ValueError(f"Invalid trajectory shapes: qpos={qpos.shape}, cup_positions={cup_positions.shape}.")

    env.reset(seed=int(config["seed"]))
    frame_indices = list(range(0, len(qpos), args.stride))
    if frame_indices[-1] != len(qpos) - 1:
        frame_indices.append(len(qpos) - 1)
    replayed = 0
    try:
        for index in frame_indices:
            env.set_replay_state(qpos[index], cup_positions[index])
            if args.render:
                env.render()
            if args.realtime:
                time.sleep(float(env.model.opt.timestep) * env.frame_skip * args.stride)
            replayed += 1
    finally:
        env.close()
    goal = np.asarray([*expert.blue_mat_center, expert.cup_initial_position[2]], dtype=np.float32)
    final_distance = float(np.linalg.norm(cup_positions[-1] - goal))
    recomputed_success = bool(
        np.any(grasped)
        and float(np.max(cup_positions[:, 2])) >= env.minimum_lift_height
        and final_distance < env.success_radius
        and float(cup_positions[-1, 2]) <= env.success_z_max
    )
    print(
        f"Replayed {replayed}/{len(qpos)} frames; saved_success={saved_success}; "
        f"recomputed_success={recomputed_success}; "
        f"SD={saved_score:.6f}; final cup-to-blue distance={final_distance:.6f} m"
    )


if __name__ == "__main__":
    main()
