import math
import random
import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn

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

def _seed_all(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class _ZeroNet(nn.Module):
    """Network that outputs uniform policy and zero value"""
    def __init__(self, n=3):
        super().__init__()
        self.n = n

    def forward(self, x):
        batch = x.size(0)
        policy = torch.zeros((batch, self.n * self.n * 3), dtype=x.dtype)
        value = torch.zeros((batch, 1), dtype=x.dtype)
        return policy, value

class TestWinningMovesN3(unittest.TestCase):
    """Test MCTS can find forced wins on 3x3 board"""
    
    def setUp(self):
        _seed_all(123)
        self.net = _ZeroNet(3)
    
    def test_immediate_win_move_forward(self):
        """O can win by moving forward to top row"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.BLANK, core.BLANK, core.BLANK],  # O can move here to win
            [core.X, core.O, core.BLANK],      # O at (1,1)
            [core.BLANK, core.BLANK, core.BLANK],
        ], dtype=np.int8)
        state.player = core.O
        
        agent = core.AlphaZeroAgent(self.net, simulations=40)
        move = agent.choose_move(state, temperature=0)
        
        # Should choose (1,1) -> (0,1) to reach top row
        self.assertEqual(move, (1, 1, 0, 1))
        
        # Verify it wins
        next_state = state.copy()
        next_state.make_move(move)
        self.assertTrue(next_state.is_terminal())
        self.assertEqual(next_state.get_winner(), core.O)
    
    def test_immediate_win_by_capture(self):
        """O can win by capturing last X piece"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.X, core.X, core.BLANK],      # Last X at (0,1)
            [core.O, core.BLANK, core.BLANK],      # O at (1,0) can capture
            [core.BLANK, core.BLANK, core.BLANK],
        ], dtype=np.int8)
        state.player = core.O
        
        agent = core.AlphaZeroAgent(self.net, simulations=40)
        move = agent.choose_move(state, temperature=0)
        
        # Should capture: (1,0) -> (0,1)
        self.assertEqual(move, (1, 0, 0, 1))
        
        next_state = state.copy()
        next_state.make_move(move)
        self.assertTrue(next_state.is_terminal())
        self.assertEqual(next_state.get_winner(), core.O)
    
    def test_avoid_losing_move(self):
        """O should avoid moves that let X win immediately"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.BLANK, core.BLANK, core.BLANK],
            [core.X, core.BLANK, core.BLANK],      # X at (1,0)
            [core.O, core.O, core.BLANK],          # O at (2,0) and (2,1)
        ], dtype=np.int8)
        state.player = core.O
        
        agent = core.AlphaZeroAgent(self.net, simulations=100)
        move = agent.choose_move(state, temperature=0)
        
        # O should NOT move (2,0) forward, as that allows X to reach bottom row
        # Moving (2,0) -> (1,0) would capture X, which is safe
        # Moving (2,1) -> (1,1) is also safe
        
        # The bad move would be moving (2,0) to anywhere that doesn't block X
        bad_moves = [(2, 1, 1, 1)]  # This lets X win by (1,0) -> (2,0)
        
        self.assertNotIn(move, bad_moves)


class TestWinningMovesN4(unittest.TestCase):
    """Test MCTS can find forced wins on 4x4 board"""
    
    def setUp(self):
        _seed_all(456)
        self.net = _ZeroNet(4)
    
    def test_immediate_win_4x4(self):
        """O can win by reaching top row on 4x4"""
        state = core.HexapawnState(4)
        state.board = np.array([
            [core.BLANK, core.BLANK, core.BLANK, core.BLANK],
            [core.BLANK, core.O, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.X, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK, core.BLANK],
        ], dtype=np.int8)
        state.player = core.O
        
        agent = core.AlphaZeroAgent(self.net, simulations=50)
        move = agent.choose_move(state, temperature=0)
        
        # Should move to top row
        self.assertEqual(move[2], 0)
        
        next_state = state.copy()
        next_state.make_move(move)
        self.assertTrue(next_state.is_terminal())
        self.assertEqual(next_state.get_winner(), core.O)
    
    def test_capture_all_pieces_4x4(self):
        """O wins by capturing all X pieces"""
        state = core.HexapawnState(4)
        state.board = np.array([
            [core.X, core.X, core.BLANK, core.BLANK],  # Last X
            [core.O, core.BLANK, core.BLANK, core.BLANK],  # O can capture
            [core.BLANK, core.BLANK, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK, core.BLANK],
        ], dtype=np.int8)
        state.player = core.O
        
        agent = core.AlphaZeroAgent(self.net, simulations=50)
        move = agent.choose_move(state, temperature=0)
        
        # Should capture
        self.assertEqual(move, (1, 0, 0, 1))
    
    def test_sees_victory(self):
        """O wins by capturing all X pieces"""
        root_state = core.HexapawnState(4)
        root_state.board = np.array([
            [core.X, core.BLANK, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK, core.O],
        ], dtype=np.int8)
        root_state.player = core.O
        
        agent = core.AlphaZeroAgent(self.net, simulations=2)
        move = agent.choose_move(root_state, temperature=0)
                
        # Should capture
        self.assertEqual(move, (3, 3, 2, 3))

        root = core.MCTSNode(root_state.copy())

        print('#'*40)
        print('#'*40)
        
        for _ in range(agent.simulations):
            node = root
            state = root_state.copy()
            search_path = [node]
            
            # STEP 1: SELECTION
            # Traverse down the tree until we hit a leaf (unexpanded node or terminal)
            while node.children and not node.is_terminal():
                node = node.best_child(agent.c_puct)
                state.make_move(node.move)
                search_path.append(node)
                print(search_path)
            
            # Now 'node' is a leaf: either terminal, or not yet expanded
            
            # STEP 2: EXPANSION & EVALUATION
            if state.is_terminal():
                # Terminal state - compute exact value
                winner = state.get_winner()
                # The player who just moved is opposite of current player
                last_mover = core.O if state.player == core.X else core.X
                value = 1.0 if winner == last_mover else -1.0
                print(value, state)
            else:
                # Non-terminal leaf - evaluate with network and expand
                with torch.no_grad():
                    policy_logits, value_tensor = agent.net(state.to_tensor(canonical=True))
                    print(policy_logits, value_tensor)
                
                # Get move priors
                move_priors = agent._policy_logits_to_priors(state, policy_logits)
                print('move_priors', move_priors)
                
                # Expand all children of this leaf
                for move in node.untried_moves[:]:
                    prior = move_priors.get(move, 1.0 / len(node.untried_moves) if node.untried_moves else 0)
                    node.expand(move, prior)
                    print(node)
                
                # Get value from network
                value = value_tensor.item()
                print('value', value)
            
            # STEP 3: BACKPROPAGATION
            # Value is from the perspective of the player at the leaf
            leaf_player = state.player
            for backup_node in reversed(search_path):
                if backup_node.state.player == leaf_player:
                    backup_node.update(value)
                else:
                    backup_node.update(-value)

            print('Search path:')
            for node in search_path:
                print(node)

        
        print('#'*40)
        print('#'*40)


class TestDeepTactics(unittest.TestCase):
    """Test MCTS can find wins requiring 3+ moves of calculation"""
    
    def setUp(self):
        _seed_all(789)
        self.net = _ZeroNet(3)
    
    def test_forced_win_in_three_plies(self):
        """O has forced win requiring 3-ply search (O-X-O)"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.X, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.O],
        ], dtype=np.int8)
        state.player = core.O
        
        # O should advance pawns to create multiple threats
        # With 3 pawns vs 1, O has a forced win
        agent = core.AlphaZeroAgent(self.net, simulations=200)
        move = agent.choose_move(state, temperature=0)
        
        # After O's move, verify X cannot prevent eventual loss
        next_state = state.copy()
        next_state.make_move(move)
        
        # Check that O's position is winning
        # (This is a heuristic check - full proof would require minimax)
        self.assertFalse(next_state.is_terminal())
        
        # O should be advancing pawns
        self.assertEqual(move[2], 1)  # Moving to row 1
    
    def test_zugzwang_position(self):
        """Position where having to move is a disadvantage"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.BLANK, core.BLANK, core.BLANK],
            [core.X, core.BLANK, core.X],
            [core.BLANK, core.O, core.BLANK],
        ], dtype=np.int8)
        state.player = core.O
        
        # O is in trouble - any forward move allows X to advance
        # O should make the best defensive move
        agent = core.AlphaZeroAgent(self.net, simulations=150)
        move = agent.choose_move(state, temperature=0)
        
        # Verify it's a legal move
        self.assertIn(move, state.get_possible_moves())


class TestSearchDepth(unittest.TestCase):
    """Test that more simulations find better moves"""
    
    def setUp(self):
        _seed_all(999)
        self.net = _ZeroNet(3)
    
    def test_shallow_vs_deep_search(self):
        """Deeper search should find better moves in complex position"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.BLANK, core.X, core.X],
            [core.BLANK, core.BLANK, core.BLANK],
            [core.O, core.O, core.BLANK],
        ], dtype=np.int8)
        state.player = core.O
        
        # Shallow search
        shallow_agent = core.AlphaZeroAgent(self.net, simulations=10)
        shallow_move = shallow_agent.choose_move(state, temperature=0)
        
        # Deep search
        deep_agent = core.AlphaZeroAgent(self.net, simulations=200)
        deep_move = deep_agent.choose_move(state, temperature=0)
        
        # Both should be legal
        self.assertIn(shallow_move, state.get_possible_moves())
        self.assertIn(deep_move, state.get_possible_moves())
        
        # Compare quality by playing out both moves
        shallow_state = state.copy()
        shallow_state.make_move(shallow_move)
        
        deep_state = state.copy()
        deep_state.make_move(deep_move)
        
        # The deep search move should be at least as good
        # (harder to test definitively without perfect play)
        print(f"Shallow ({10} sims): {shallow_move}")
        print(f"Deep ({200} sims): {deep_move}")


class TestMCTSConsistency(unittest.TestCase):
    """Test that MCTS is deterministic when it should be"""
    
    def setUp(self):
        _seed_all(111)
        self.net = _ZeroNet(3)
    
    def test_same_move_with_same_seed(self):
        """Same seed should produce same move (temperature=0)"""
        state = core.HexapawnState(3)
        
        _seed_all(555)
        agent1 = core.AlphaZeroAgent(self.net, simulations=50)
        move1 = agent1.choose_move(state, temperature=0)
        
        _seed_all(555)
        agent2 = core.AlphaZeroAgent(self.net, simulations=50)
        move2 = agent2.choose_move(state, temperature=0)
        
        self.assertEqual(move1, move2)
    
    def test_exploration_with_temperature(self):
        """Temperature > 0 should produce varied moves"""
        state = core.HexapawnState(3)
        agent = core.AlphaZeroAgent(self.net, simulations=50)
        
        moves = []
        for _ in range(20):
            move = agent.choose_move(state, temperature=1.0)
            moves.append(move)
        
        # Should see some variety (at least 2 different moves)
        unique_moves = len(set(moves))
        self.assertGreaterEqual(unique_moves, 2, "Temperature=1.0 should produce varied moves")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        _seed_all(222)
        self.net = _ZeroNet(3)
    
    def test_only_one_legal_move(self):
        """When only one move is legal, MCTS should pick it"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.X, core.X, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK],
            [core.BLANK, core.BLANK, core.O],
        ], dtype=np.int8)
        state.player = core.O
        
        # Only move is (2,2) -> (1,2)
        legal_moves = state.get_possible_moves()
        self.assertEqual(len(legal_moves), 1)
        
        agent = core.AlphaZeroAgent(self.net, simulations=20)
        move = agent.choose_move(state, temperature=0)
        
        self.assertEqual(move, legal_moves[0])
    
    def test_x_player_can_win(self):
        """Test that X (not just O) can find winning moves"""
        state = core.HexapawnState(3)
        state.board = np.array([
            [core.BLANK, core.BLANK, core.BLANK],
            [core.O, core.X, core.BLANK],
            [core.BLANK, core.BLANK, core.BLANK],
        ], dtype=np.int8)
        state.player = core.X
        
        agent = core.AlphaZeroAgent(self.net, simulations=40)
        move = agent.choose_move(state, temperature=0)
        
        # X should move to bottom row to win
        self.assertEqual(move[2], 2)
        
        next_state = state.copy()
        next_state.make_move(move)
        self.assertTrue(next_state.is_terminal())
        self.assertEqual(next_state.get_winner(), core.X)


def run_position_tests():
    """Helper to run specific position tests with verbose output"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWinningMovesN3)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    unittest.main(verbosity=2)