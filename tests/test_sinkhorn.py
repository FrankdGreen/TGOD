from __future__ import annotations

import unittest

import numpy as np

from tgod_sd.sinkhorn import sinkhorn_distance, solve_sinkhorn


class SinkhornTests(unittest.TestCase):
    def test_identical_is_closer_than_shifted(self) -> None:
        trajectory = np.stack([np.linspace(0.0, 1.0, 20), np.linspace(1.0, 0.0, 20)], axis=1)
        identical = sinkhorn_distance(trajectory, trajectory, epsilon=0.05)
        shifted = sinkhorn_distance(trajectory, trajectory + 0.5, epsilon=0.05)
        self.assertLess(identical, shifted)

    def test_is_symmetric(self) -> None:
        first = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float64)
        second = np.asarray([[0.1], [0.6]], dtype=np.float64)
        forward = sinkhorn_distance(first, second, epsilon=0.1)
        backward = sinkhorn_distance(second, first, epsilon=0.1)
        self.assertAlmostEqual(forward, backward, places=6)

    def test_rejects_nonpositive_epsilon(self) -> None:
        with self.assertRaises(ValueError):
            sinkhorn_distance(np.zeros((2, 1)), np.zeros((2, 1)), epsilon=0.0)

    def test_reports_converged_marginals(self) -> None:
        first = np.linspace(0.0, 1.0, 20)[:, None]
        second = np.linspace(0.1, 0.9, 17)[:, None]
        result = solve_sinkhorn(first, second, epsilon=0.1, max_iterations=500)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.marginal_error, 1e-7)


if __name__ == "__main__":
    unittest.main()
