from __future__ import annotations

import unittest
from pathlib import Path
from typing import ClassVar

from tgod_sd.config import load_config


class ConfigurationTests(unittest.TestCase):
    path: ClassVar[Path]

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).resolve().parents[1] / "configs" / "ur5e_pick_place.yaml"

    def test_default_configuration_is_valid(self) -> None:
        config = load_config(self.path)
        self.assertEqual(config["environment"]["max_episode_steps"], 500)

    def test_fractional_candidate_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.path, {"matching": {"candidate_count": 0.5}})

    def test_mine_batch_requires_negative_sample(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.path, {"sac": {"batch_size": 1}})

    def test_replay_must_reach_update_threshold(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.path, {"sac": {"replay_size": 500, "update_after": 1000}})


if __name__ == "__main__":
    unittest.main()
