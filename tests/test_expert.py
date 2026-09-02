from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from tgod_sd.config import load_config, resolve_input_path
from tgod_sd.expert import RELATION_PROXIMITY_INDEX, ExpertTrajectory, resample_trajectory
from tgod_sd.schema import CUP_POS, EE_POS, OBS_DIM, QPOS
from tgod_sd.sinkhorn import sinkhorn_distance
from tgod_sd.trajectory import pad_terminal_hold


class ExpertTrajectoryTests(unittest.TestCase):
    expert: ClassVar[ExpertTrajectory]
    config: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.config = load_config(root / "configs" / "ur5e_pick_place.yaml")
        directory = resolve_input_path(cls.config["paths"]["expert_dir"], kind="expert")
        cls.expert = ExpertTrajectory.load(directory)

    def test_expected_layout(self) -> None:
        self.assertEqual(len(self.expert), 500)
        self.assertEqual(self.expert.matching_features.shape, (500, 12))
        self.assertEqual(self.expert.relation_dim, 39)
        np.testing.assert_allclose(self.expert.cup_position[0, :2], self.expert.red_mat_center, atol=1e-6)
        np.testing.assert_allclose(self.expert.cup_position[-1, :2], self.expert.blue_mat_center, atol=1e-6)

    def test_resampling_preserves_endpoints(self) -> None:
        resampled = resample_trajectory(self.expert.matching_features, 37)
        self.assertEqual(resampled.shape, (37, 12))
        np.testing.assert_allclose(resampled[0], self.expert.matching_features[0], atol=1e-6)
        np.testing.assert_allclose(resampled[-1], self.expert.matching_features[-1], atol=1e-6)

    def test_terminal_hold_padding_preserves_real_time_axis(self) -> None:
        prefix = self.expert.matching_features[:371]
        padded = pad_terminal_hold(prefix, len(self.expert))
        self.assertEqual(padded.shape, self.expert.matching_features.shape)
        np.testing.assert_allclose(padded[:371], prefix, atol=0.0)
        np.testing.assert_allclose(padded[-1], prefix[-1], atol=0.0)

        matching = self.config["matching"]
        kwargs = {
            "qpos_weight": float(matching["qpos_weight"]),
            "tcp_weight": float(matching["tcp_weight"]),
            "cup_weight": float(matching["cup_weight"]),
            "time_weight": float(matching["time_weight"]),
        }
        full = resample_trajectory(
            self.expert.normalized_matching_features(self.expert.matching_features, **kwargs), 100
        )
        early = resample_trajectory(
            self.expert.normalized_matching_features(prefix, **kwargs), 100
        )
        held = resample_trajectory(
            self.expert.normalized_matching_features(padded, **kwargs), 100
        )
        early_distance = sinkhorn_distance(early, full, epsilon=0.05)
        held_distance = sinkhorn_distance(held, full, epsilon=0.05)
        self.assertLess(held_distance, 0.05)
        self.assertLess(held_distance, early_distance)

    def test_demonstration_relation_has_directional_support(self) -> None:
        progress = 0.5
        feature = self.expert.feature_at(progress)
        near_observation = np.zeros(OBS_DIM, dtype=np.float32)
        near_observation[QPOS] = feature[:6]
        near_observation[EE_POS] = feature[6:9]
        near_observation[CUP_POS] = feature[9:12]
        far_observation = near_observation.copy()
        far_observation[QPOS] += 1.0
        far_observation[EE_POS] += 0.5
        far_observation[CUP_POS] += 0.5
        near_relation = self.expert.relation_feature(near_observation, progress)
        far_relation = self.expert.relation_feature(far_observation, progress)
        near_support = float(near_relation[RELATION_PROXIMITY_INDEX])
        far_support = float(far_relation[RELATION_PROXIMITY_INDEX])
        self.assertAlmostEqual(near_support, 1.0, places=6)
        self.assertGreater(near_support, far_support)
        self.assertGreater(np.log(near_support), np.log(max(far_support, 1e-6)))


if __name__ == "__main__":
    unittest.main()
