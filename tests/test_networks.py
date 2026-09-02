from __future__ import annotations

import unittest

from torch import nn

from tgod_sd.networks import SquashedGaussianActor


class NetworkTests(unittest.TestCase):
    def test_actor_activates_every_configured_hidden_layer(self) -> None:
        actor = SquashedGaussianActor(27, 8, 4, [32, 16], "relu")
        linear_layers = [module for module in actor.trunk if isinstance(module, nn.Linear)]
        activations = [module for module in actor.trunk if isinstance(module, nn.ReLU)]
        self.assertEqual([layer.out_features for layer in linear_layers], [32, 16])
        self.assertEqual(len(activations), 2)


if __name__ == "__main__":
    unittest.main()
