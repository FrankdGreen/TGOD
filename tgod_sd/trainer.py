from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import TGODSACAgent
from .config import resolve_input_path, resolve_output_path, select_device
from .env import UR5ePickPlaceEnv
from .expert import ExpertTrajectory
from .replay_buffer import ReplayBuffer
from .schema import OBS_DIM
from .trajectory import generate_and_match, one_hot_skill


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_components(
    config: dict[str, Any], *, render_mode: str | None = None
) -> tuple[ExpertTrajectory, UR5ePickPlaceEnv, TGODSACAgent, Path, str]:
    scene_path = resolve_input_path(config["paths"]["scene_xml"], kind="scene")
    expert_directory = resolve_input_path(config["paths"]["expert_dir"], kind="expert")
    output_directory = resolve_output_path(config["paths"]["output_dir"])
    device = select_device(str(config["device"]))
    expert = ExpertTrajectory.load(expert_directory)
    env = UR5ePickPlaceEnv(scene_path, expert, config["environment"], render_mode=render_mode)
    agent = TGODSACAgent(OBS_DIM, 4, expert.relation_dim, config, device)
    return expert, env, agent, output_directory, device


def save_checkpoint(
    path: Path,
    *,
    agent: TGODSACAgent,
    config: dict[str, Any],
    episode: int,
    global_step: int,
    rng: np.random.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format_version": 1,
        "agent": agent.state_dict(),
        "config": config,
        "episode": int(episode),
        "global_step": int(global_step),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    agent: TGODSACAgent,
    *,
    load_optimizers: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=agent.device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    agent.load_state_dict(checkpoint["agent"], load_optimizers=load_optimizers)
    return checkpoint


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    return {
        key: float(np.mean([entry[key] for entry in metrics]))
        for key in metrics[0]
    }


def train(config: dict[str, Any], resume_path: str | Path | None = None) -> Path:
    seed = int(config["seed"])
    seed_everything(seed)
    expert, env, agent, output_directory, device = build_components(config)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_directory = output_directory / "checkpoints"
    metrics_path = output_directory / "metrics.jsonl"
    sac_config = config["sac"]
    training_config = config["training"]
    replay = ReplayBuffer(
        int(sac_config["replay_size"]),
        OBS_DIM,
        4,
        agent.skill_dim,
        expert.relation_dim,
        seed,
    )
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    start_episode = 0
    global_step = 0
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, agent, load_optimizers=True)
        start_episode = int(checkpoint["episode"])
        global_step = int(checkpoint["global_step"])
        if "numpy_rng_state" in checkpoint:
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        print(
            f"Resumed episode={start_episode}, global_step={global_step}. "
            "The replay buffer is intentionally refilled before updates resume."
        )

    total_episodes = int(training_config["episodes"])
    if start_episode > total_episodes:
        env.close()
        raise ValueError(
            f"Checkpoint is at episode {start_episode}, beyond configured total {total_episodes}. "
            "Increase --episodes instead of rewinding the checkpoint."
        )
    if start_episode == total_episodes:
        print(f"Checkpoint already reached configured episode count {total_episodes}.")
    print(
        f"TGOD-SD training on {device}: episodes={total_episodes}, skills={agent.skill_dim}, "
        f"expert_frames={len(expert)}, output={output_directory}"
    )

    latest_path = checkpoints_directory / "latest.pt"
    try:
        for episode in range(start_episode, total_episodes):
            skill_index = int(rng.integers(agent.skill_dim))
            skill = one_hot_skill(skill_index, agent.skill_dim)
            observation, _ = env.reset(seed=seed + episode)
            episode_steps = 0
            episode_updates: list[dict[str, float]] = []
            collisions = 0
            contacts = 0
            success = False

            while True:
                if global_step < int(sac_config["random_steps"]):
                    action = env.action_space.sample()
                else:
                    action = agent.act(observation, skill, deterministic=False)
                relation = expert.relation_feature(observation, float(observation[-1]))
                next_observation, environment_reward, terminated, truncated, info = env.step(action)
                if environment_reward != 0.0:
                    raise RuntimeError("TGOD environment reward must remain zero; MINE supplies the pseudo-reward.")
                replay_terminal = bool(
                    terminated
                    or (truncated and not bool(sac_config["bootstrap_on_timeout"]))
                )
                replay.add(
                    observation,
                    action,
                    next_observation,
                    skill,
                    relation,
                    terminal=replay_terminal,
                )
                observation = next_observation
                global_step += 1
                episode_steps += 1
                collisions += int(info["collision"])
                contacts += int(info["contact"])
                success = bool(info["success"])

                minimum_replay = max(int(sac_config["batch_size"]), int(sac_config["update_after"]))
                if (
                    len(replay) >= minimum_replay
                    and global_step % int(sac_config["update_every"]) == 0
                ):
                    for _ in range(int(sac_config["gradient_steps"])):
                        batch = replay.sample(int(sac_config["batch_size"]))
                        episode_updates.append(agent.update(batch))
                if terminated or truncated:
                    break

            update_metrics = _mean_metrics(episode_updates)
            record: dict[str, Any] = {
                "episode": episode + 1,
                "global_step": global_step,
                "skill_index": skill_index,
                "steps": episode_steps,
                "success": success,
                "contacts": contacts,
                "collisions": collisions,
                "replay_size": len(replay),
                **update_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (episode + 1) % int(training_config["log_every_episodes"]) == 0 or episode == start_episode:
                pseudo = update_metrics.get("pseudo_reward_raw_mean", float("nan"))
                print(
                    f"Episode {episode + 1}/{total_episodes}: steps={episode_steps}, "
                    f"success={success}, skill={skill_index}, pseudo={pseudo:.4f}, replay={len(replay)}"
                )
            if (episode + 1) % int(training_config["checkpoint_every_episodes"]) == 0:
                save_checkpoint(
                    checkpoints_directory / f"episode_{episode + 1:05d}.pt",
                    agent=agent,
                    config=config,
                    episode=episode + 1,
                    global_step=global_step,
                    rng=rng,
                )
                save_checkpoint(
                    latest_path,
                    agent=agent,
                    config=config,
                    episode=episode + 1,
                    global_step=global_step,
                    rng=rng,
                )
    except KeyboardInterrupt:
        print("Training interrupted; saving latest checkpoint before exiting.")
        save_checkpoint(
            latest_path,
            agent=agent,
            config=config,
            episode=episode,
            global_step=global_step,
            rng=rng,
        )
        env.close()
        raise

    save_checkpoint(
        latest_path,
        agent=agent,
        config=config,
        episode=total_episodes,
        global_step=global_step,
        rng=rng,
    )
    if bool(training_config["match_after_training"]):
        generate_and_match(
            env,
            expert,
            agent,
            config["matching"],
            output_directory,
            seed=seed,
        )
    env.close()
    print(f"Training complete. Latest checkpoint: {latest_path}")
    return latest_path
