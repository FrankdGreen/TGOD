import argparse
import os
import sys
from pathlib import Path

from .env import UR5eSingleArmEnv

if "--allow-duplicate-openmp" in sys.argv:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-arm UR5e SAC planner without expert demos.")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Stop after this many completed episodes; overrides the timestep limit.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto/cpu/cuda")
    parser.add_argument("--model-dir", type=str, default="models", help="Where to save trained models.")
    parser.add_argument(
        "--xml-path",
        type=str,
        default="universal_robots_ur5e/scene.xml",
        help="UR5e MuJoCo XML path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--allow-duplicate-openmp",
        action="store_true",
        help="Allow duplicate OpenMP runtimes on Windows.",
    )
    return parser.parse_args()


class EpisodeLimitCallback(BaseCallback):
    def __init__(self, episode_limit: int):
        super().__init__(verbose=0)
        self.episode_limit = episode_limit
        self.completed_episodes = 0

    def _on_step(self) -> bool:
        self.completed_episodes += int(self.locals["dones"].sum())
        return self.completed_episodes < self.episode_limit


def main():
    args = parse_args()
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive.")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = UR5eSingleArmEnv(xml_path=args.xml_path)
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    callback = EpisodeLimitCallback(args.episodes) if args.episodes is not None else None
    total_timesteps = args.episodes * 300 if args.episodes is not None else args.timesteps
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    if callback is not None:
        print(f"Completed episodes: {callback.completed_episodes}")

    save_path = model_dir / "sac_ur5e_single_arm"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
