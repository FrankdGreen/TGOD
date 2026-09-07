from __future__ import annotations

import unittest
from pathlib import Path
from typing import ClassVar

from tgod_sd.config import load_config, validate_reproduction_training


class ConfigurationTests(unittest.TestCase):
    path: ClassVar[Path]

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).resolve().parents[1] / "configs" / "ur5e_pick_place.yaml"

    def test_default_configuration_is_valid(self) -> None:
        config = load_config(self.path)
        self.assertEqual(config["environment"]["max_episode_steps"], 500)
        validate_reproduction_training(config)

    def test_fractional_candidate_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.path, {"matching": {"candidate_count": 0.5}})

    def test_mine_batch_requires_negative_sample(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.path, {"sac": {"batch_size": 1}})

    def test_replay_must_reach_update_threshold(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.path, {"sac": {"replay_size": 500, "update_after": 1000}})

    def test_paper_run_retains_environment_and_optimizer_parameters(self) -> None:
        paper = load_config(self.path.parent / "paper_seed45.yaml")
        old = load_config(self.path.parent / "retrain_seed42.yaml")
        validate_reproduction_training(paper)
        self.assertEqual(paper["seed"], 45)
        self.assertEqual(paper["paths"]["output_dir"], "outputs/paper_seed45")
        self.assertEqual(paper["environment"], old["environment"])
        self.assertEqual(paper["network"], old["network"])
        self.assertEqual(paper["sac"], old["sac"])
        self.assertEqual(paper["tgod"]["mine_learning_rate"], old["tgod"]["mine_learning_rate"])
        self.assertEqual(paper["training"]["episodes"], 2000)

    def test_historical_configurations_are_readable_but_cannot_train(self) -> None:
        for name in ("retrain_seed42.yaml", "finetune_seed45.yaml"):
            with self.subTest(name=name):
                config = load_config(self.path.parent / name)
                with self.assertRaisesRegex(ValueError, "Reproduction-first training rejects"):
                    validate_reproduction_training(config)

    def test_training_rejects_added_reward_transformations(self) -> None:
        for name, value in (
            ("state_mi_weight", 0.05), ("demonstration_mi_weight", 0.25),
            ("demonstration_support_weight", 1.0), ("demonstration_progress_weight", 5.0),
            ("reward_normalization", True), ("pseudo_reward_clip", 10.0),
        ):
            with self.subTest(name=name):
                config = load_config(self.path, {"tgod": {name: value}})
                with self.assertRaisesRegex(ValueError, name):
                    validate_reproduction_training(config)

    def test_training_rejects_success_filtered_matching(self) -> None:
        config = load_config(self.path, {"matching": {"prefer_successful": True}})
        with self.assertRaisesRegex(ValueError, "minimum SD over all candidates"):
            validate_reproduction_training(config)

    def test_training_rejects_removed_adaptive_controls(self) -> None:
        for name, value in (
            ("evaluation_every_episodes", 25), ("early_stop_patience", 3),
            ("resume_replay_steps", 10000), ("reset_reward_statistics_on_resume", True),
        ):
            with self.subTest(name=name):
                config = load_config(self.path, {"training": {name: value}})
                with self.assertRaisesRegex(ValueError, name):
                    validate_reproduction_training(config)

    def test_training_requires_resumable_latest_checkpoint(self) -> None:
        config = load_config(self.path, {"training": {"save_replay_buffer": False}})
        with self.assertRaisesRegex(ValueError, "save_replay_buffer"):
            validate_reproduction_training(config)


if __name__ == "__main__":
    unittest.main()
