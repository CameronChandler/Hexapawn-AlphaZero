import math
import random
import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

import hexapawn_core as core


def _seed_all(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestHexapawnState(unittest.TestCase):
    def setUp(self):
        _seed_all(123)

    def test_initial_moves_count(self):
        state = core.HexapawnState(3)
        moves = state.get_possible_moves()
        self.assertEqual(len(moves), 3)

    def test_make_move_switches_player(self):
        state = core.HexapawnState(3)
        move = state.get_possible_moves()[0]
        current = state.player
        state.make_move(move)
        self.assertNotEqual(state.player, current)

    def test_no_moves_is_loss(self):
        state = core.HexapawnState(3)
        # Create a position where O to move has no legal moves.
        state.board = np.array(
            [
                [core.BLANK, core.X, core.BLANK],
                [core.X, core.O, core.X],
                [core.BLANK, core.BLANK, core.BLANK],
            ],
            dtype=np.int8,
        )
        state.player = core.O
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.get_winner(), core.X)

    def test_canonical_board_flips_for_x(self):
        state = core.HexapawnState(3)
        state.board = np.array(
            [
                [core.X, core.BLANK, core.O],
                [core.BLANK, core.O, core.BLANK],
                [core.O, core.BLANK, core.X],
            ],
            dtype=np.int8,
        )
        state.player = core.X
        expected = -np.flipud(np.fliplr(state.board))
        self.assertTrue(np.array_equal(state.canonical_board(), expected))

    def test_to_tensor_canonical_marks_current_player_as_o(self):
        state = core.HexapawnState(3)
        state.board = np.array(
            [
                [core.X, core.BLANK, core.O],
                [core.BLANK, core.O, core.BLANK],
                [core.O, core.BLANK, core.X],
            ],
            dtype=np.int8,
        )
        state.player = core.X
        tensor = state.to_tensor(canonical=True).squeeze(0).numpy()
        self.assertTrue(np.all(tensor[2] == 1.0))
        expected_board = -np.flipud(np.fliplr(state.board))
        self.assertTrue(np.array_equal(tensor[0], (expected_board == core.X).astype(np.float32)))
        self.assertTrue(np.array_equal(tensor[1], (expected_board == core.O).astype(np.float32)))

    def test_move_to_policy_index_uses_canonical_orientation(self):
        state = core.HexapawnState(3)
        state.board = np.array(
            [
                [core.BLANK, core.BLANK, core.BLANK],
                [core.BLANK, core.X, core.BLANK],
                [core.BLANK, core.BLANK, core.BLANK],
            ],
            dtype=np.int8,
        )
        state.player = core.X
        move = (1, 1, 2, 1)  # X moves down; canonical should map to O moving up.
        canonical_move = state.canonical_move(move)
        r, c, nr, nc = canonical_move
        dc = nc - c
        expected_idx = (r * state.n + c) * 3 + (dc + 1)
        self.assertEqual(state.move_to_policy_index(move, canonical=True), expected_idx)


class TestMCTS(unittest.TestCase):
    def setUp(self):
        _seed_all(123)

    def test_search_expands_root_children(self):
        net = core.HexapawnNet(3)
        agent = core.AlphaZeroAgent(net, simulations=5)
        state = core.HexapawnState(3)
        root = agent.search(state, add_root_noise=False)
        self.assertEqual(len(root.children), len(state.get_possible_moves()))
        self.assertTrue(all(c.visits > 0 for c in root.children))

    def test_policy_target_distribution(self):
        net = core.HexapawnNet(3)
        agent = core.AlphaZeroAgent(net, simulations=5)
        state = core.HexapawnState(3)
        policy = agent.get_policy_target(state, temperature=1.0)
        self.assertTrue(policy)
        total = sum(policy.values())
        self.assertTrue(math.isclose(total, 1.0, rel_tol=1e-5))
        legal = set(state.get_possible_moves())
        self.assertTrue(set(policy.keys()).issubset(legal))


if __name__ == "__main__":
    unittest.main()
