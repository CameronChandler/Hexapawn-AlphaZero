import random
import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

import train
import hexapawn_core as core


def _seed_all(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestSelfPlay(unittest.TestCase):
    def setUp(self):
        _seed_all(123)

    def test_self_play_examples_sane(self):
        net = core.HexapawnNet(3)
        agent = core.AlphaZeroAgent(net, simulations=5)
        examples = train.self_play_game(agent, temperature=1.0, n=3)
        self.assertTrue(examples)
        for state, policy, value in examples:
            self.assertIn(value, (-1, 1))
            # Policy keys should be legal moves at that state.
            legal = set(state.get_possible_moves())
            self.assertTrue(set(policy.keys()).issubset(legal))
            total = sum(policy.values())
            self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
