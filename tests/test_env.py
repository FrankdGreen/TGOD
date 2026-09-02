from __future__ import annotations

import copy
import unittest
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from tgod_sd.config import load_config, resolve_input_path
from tgod_sd.env import UR5ePickPlaceEnv, _solve_spd_3x3
from tgod_sd.expert import ExpertTrajectory
from tgod_sd.schema import OBS_DIM


class EnvironmentTests(unittest.TestCase):
    config: ClassVar[dict[str, Any]]
    scene: ClassVar[Path]
    expert: ClassVar[ExpertTrajectory]

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.config = load_config(root / "configs" / "ur5e_pick_place.yaml")
        cls.scene = resolve_input_path(cls.config["paths"]["scene_xml"], kind="scene")
        expert_directory = resolve_input_path(cls.config["paths"]["expert_dir"], kind="expert")
        cls.expert = ExpertTrajectory.load(expert_directory)

    def test_zero_task_reward_and_timeout_semantics(self) -> None:
        environment_config = copy.deepcopy(self.config["environment"])
        environment_config["max_episode_steps"] = 1
        env = UR5ePickPlaceEnv(self.scene, self.expert, environment_config)
        try:
            observation, _ = env.reset(seed=1)
            next_observation, reward, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            self.assertEqual(observation.shape, (OBS_DIM,))
            self.assertEqual(next_observation.shape, (OBS_DIM,))
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_small_cholesky_solver_matches_numpy(self) -> None:
        matrix = np.asarray(
            [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]],
            dtype=np.float64,
        )
        right_hand_side = np.asarray([0.4, -0.2, 0.8], dtype=np.float64)
        np.testing.assert_allclose(
            _solve_spd_3x3(matrix, right_hand_side),
            np.linalg.solve(matrix, right_hand_side),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_expert_tcp_guidance_completes_pick_and_place(self) -> None:
        env = UR5ePickPlaceEnv(self.scene, self.expert, self.config["environment"])
        try:
            env.reset(seed=2)
            terminated = truncated = False
            info = {}
            for index, tcp_position in enumerate(self.expert.tcp_pose[:, :3]):
                action = np.zeros(4, dtype=np.float32)
                action[:3] = np.clip(
                    (tcp_position - env._get_ee_pos()) / env.max_ee_step,
                    -1.0,
                    1.0,
                )
                action[3] = 1.0 if 80 <= index < 370 else -1.0
                _, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(reward, 0.0)
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["success"])
            self.assertLess(info["cup_goal_distance"], 0.01)
        finally:
            env.close()

    def test_initial_grip_command_cannot_grasp_at_a_distance(self) -> None:
        env = UR5ePickPlaceEnv(self.scene, self.expert, self.config["environment"])
        try:
            env.reset(seed=3)
            _, _, _, _, info = env.step(
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )
            self.assertFalse(info["contact"])
            self.assertFalse(info["grasped"])
            self.assertFalse(info["ever_grasped"])
        finally:
            env.close()

    def test_cup_at_goal_is_not_success_without_pick_and_lift(self) -> None:
        env = UR5ePickPlaceEnv(self.scene, self.expert, self.config["environment"])
        try:
            env.reset(seed=4)
            cup_goal = np.asarray(
                [*self.expert.blue_mat_center, self.expert.cup_initial_position[2]],
                dtype=np.float32,
            )
            env.set_replay_state(env.data.qpos[:6].copy(), cup_goal)
            _, _, terminated, _, info = env.step(
                np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            )
            self.assertFalse(terminated)
            self.assertFalse(info["success"])
            self.assertFalse(info["ever_grasped"])
            self.assertFalse(info["cup_lifted"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
