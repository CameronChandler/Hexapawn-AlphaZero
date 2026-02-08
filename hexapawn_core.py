"""
Core classes for AlphaZero Hexapawn
Shared between training and playing
"""

import numpy as np
import math
import torch
import torch.nn as nn
import os

X, O, BLANK = 1, -1, 0
WEIGHTS_DIR = "weights"


class HexapawnState:
    def __init__(self, n=3):
        self.n = n
        self.board = np.full((n, n), BLANK, dtype=np.int8)
        self.board[0] = X
        self.board[-1] = O
        self.player = O

    def get_possible_moves(self):
        moves = []
        player_pieces = np.where(self.board == self.player)
        for r, c in zip(*player_pieces):
            if self.player == O:
                if r > 0:
                    if c > 0 and self.board[r-1, c-1] == X:
                        moves.append((r, c, r-1, c-1))
                    if self.board[r-1, c] == BLANK:
                        moves.append((r, c, r-1, c))
                    if c < self.n-1 and self.board[r-1, c+1] == X:
                        moves.append((r, c, r-1, c+1))
            else:
                if r < self.n-1:
                    if c > 0 and self.board[r+1, c-1] == O:
                        moves.append((r, c, r+1, c-1))
                    if self.board[r+1, c] == BLANK:
                        moves.append((r, c, r+1, c))
                    if c < self.n-1 and self.board[r+1, c+1] == O:
                        moves.append((r, c, r+1, c+1))
        return moves

    def make_move(self, move):
        r, c, nr, nc = move
        self.board[nr, nc] = self.board[r, c]
        self.board[r, c] = BLANK
        self.player = X if self.player == O else O

    def is_terminal(self):
        if O in self.board[0] or X in self.board[-1]:
            return True
        if not np.any(self.board == O) or not np.any(self.board == X):
            return True
        if not self.get_possible_moves():
            return True
        return False

    def get_winner(self):
        if not self.is_terminal():
            return None
        if O in self.board[0] or not np.any(self.board == X):
            return O
        if X in self.board[-1] or not np.any(self.board == O):
            return X
        # No moves available: current player to move loses.
        return X if self.player == O else O

    def copy(self):
        new_state = HexapawnState.__new__(HexapawnState)
        new_state.n = self.n
        new_state.board = self.board.copy()
        new_state.player = self.player
        return new_state

    def canonical_board(self):
        """
        Return board oriented to the current player's perspective.
        The current player is always represented as O to move.
        """
        if self.player == O:
            return self.board
        # Flip 180 degrees and swap X/O so the current player becomes O.
        return -np.flipud(np.fliplr(self.board))

    def canonical_move(self, move):
        """Map a move into the canonical orientation (current player as O)."""
        if self.player == O:
            return move
        r, c, nr, nc = move
        n = self.n - 1
        return (n - r, n - c, n - nr, n - nc)

    def move_to_policy_index(self, move, canonical=True):
        """Map a move to the policy output index."""
        if canonical:
            move = self.canonical_move(move)
        r, c, nr, nc = move
        dc = nc - c
        if dc < -1 or dc > 1:
            return None
        return (r * self.n + c) * 3 + (dc + 1)

    def to_tensor(self, canonical=True):
        """Convert board to neural network input (3 channels: X, O, current player)."""
        board = self.canonical_board() if canonical else self.board
        player = O if (canonical and self.player == X) else self.player
        channels = np.zeros((3, self.n, self.n), dtype=np.float32)
        channels[0] = (board == X).astype(np.float32)
        channels[1] = (board == O).astype(np.float32)
        channels[2] = np.full((self.n, self.n), 1.0 if player == O else 0.0)
        return torch.FloatTensor(channels).unsqueeze(0)

    def display_board(self):
        print("  " + " ".join(str(i) for i in range(self.n)))
        for r in range(self.n):
            print(f"{r} {' '.join('X' if cell == X else 'O' if cell == O else '.' for cell in self.board[r])}")

    def __repr__(self):
        rows = []
        for r in range(self.n):
            row = ''.join('X' if cell == X else 'O' if cell == O else '.' for cell in self.board[r])
            rows.append(row)
        return '\n'.join(rows) + f"\nPlayer to move: {'O' if self.player == O else 'X'}"


class HexapawnNet(nn.Module):
    """Neural network that outputs policy (move probabilities) and value (position evaluation)"""
    def __init__(self, n=3, hidden_size=64):
        super().__init__()
        self.n = n
        
        # Shared layers
        self.conv1 = nn.Conv2d(3, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        
        # Policy head (move probabilities)
        self.policy_conv = nn.Conv2d(hidden_size, 32, 1)
        self.policy_fc = nn.Linear(32 * n * n, n * n * 3)
        
        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(hidden_size, 32, 1)
        self.value_fc1 = nn.Linear(32 * n * n, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Shared representation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Policy head
        policy = torch.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = torch.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.untried_moves = state.get_possible_moves()
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self.state.is_terminal()
    
    def get_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def best_child(self, c_puct=1.0):
        """Select child using PUCT (Polynomial Upper Confidence Trees)"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            q_value = child.get_value()
            u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, move, prior):
        next_state = self.state.copy()
        next_state.make_move(move)
        child = MCTSNode(next_state, parent=self, move=move, prior=prior)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child
    
    def update(self, value):
        self.visits += 1
        self.value_sum += value

    def __repr__(self):
        return f"MCTSNode(move={self.move}, visits={self.visits}, value={self.get_value():.3f})\n{self.state.__repr__()}"


class AlphaZeroAgent:
    def __init__(self, net, simulations=100, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
    def get_move_priors(self, state):
        """Get policy (move probabilities) from neural network"""
        with torch.no_grad():
            policy_logits, _ = self.net(state.to_tensor(canonical=True))
        return self._policy_logits_to_priors(state, policy_logits)

    def _policy_logits_to_priors(self, state, policy_logits):
        policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()

        moves = state.get_possible_moves()
        move_probs = {}
        for move in moves:
            idx = state.move_to_policy_index(move, canonical=True)
            if idx is None:
                continue
            move_probs[move] = policy[idx]

        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: p / total for m, p in move_probs.items()}
        else:
            move_probs = {m: 1.0 / len(moves) for m in moves}

        return move_probs
    
    def _add_dirichlet_noise(self, move_priors):
        if not move_priors:
            return move_priors
        moves = list(move_priors.keys())
        alpha = self.dirichlet_alpha
        epsilon = self.dirichlet_epsilon
        noise = np.random.dirichlet([alpha] * len(moves))
        return {
            move: (1 - epsilon) * move_priors[move] + epsilon * noise[i]
            for i, move in enumerate(moves)
        }

    def search(self, root_state: HexapawnState, add_root_noise:bool=False) -> MCTSNode:
        """
        Run MCTS guided by neural network
        
        Proper MCTS flow:
        1. Selection: traverse tree using PUCT
        2. Expansion: expand leaf node
        3. Evaluation: get value
        4. Backpropagation: update path
        """
        root = MCTSNode(root_state.copy())

        # if root_state.is_terminal():
        #     return root
        
        for _ in range(self.simulations):
            node = root
            state = root_state.copy()
            search_path = [node]
            
            # STEP 1: SELECTION
            # Traverse down the tree until we hit a leaf (unexpanded node or terminal)
            while node.children and not node.is_terminal():
                node = node.best_child(self.c_puct)
                state.make_move(node.move)
                search_path.append(node)
            
            # Now 'node' is a leaf: either terminal, or not yet expanded
            
            # STEP 2: EXPANSION & EVALUATION
            if state.is_terminal():
                # Terminal state - compute exact value
                winner = state.get_winner()
                # The player who just moved is opposite of current player
                last_mover = O if state.player == X else X
                value = 1.0 if winner == last_mover else -1.0
            else:
                # Non-terminal leaf - evaluate with network and expand
                with torch.no_grad():
                    policy_logits, value_tensor = self.net(state.to_tensor(canonical=True))
                
                # Get move priors
                move_priors = self._policy_logits_to_priors(state, policy_logits)
                
                # Add Dirichlet noise to root
                if node is root and add_root_noise:
                    move_priors = self._add_dirichlet_noise(move_priors)
                
                # Expand all children of this leaf
                for move in node.untried_moves[:]:
                    prior = move_priors.get(move, 1.0 / len(node.untried_moves) if node.untried_moves else 0)
                    node.expand(move, prior)
                
                # Get value from network
                value = value_tensor.item()
            
            # STEP 3: BACKPROPAGATION
            # Value is from the perspective of the player at the leaf
            leaf_player = state.player
            for backup_node in reversed(search_path):
                if backup_node.state.player == leaf_player:
                    backup_node.update(value)
                else:
                    backup_node.update(-value)
        
        return root
    
    def choose_move(self, state, temperature=0):
        """Choose move based on MCTS visit counts"""
        root = self.search(state, add_root_noise=False)
        
        if temperature == 0:
            return max(root.children, key=lambda c: c.visits).move
        else:
            moves = [c.move for c in root.children]
            visits = np.array([c.visits for c in root.children])
            probs = visits ** (1.0 / temperature)
            probs = probs / probs.sum()
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
    
    def get_policy_target(self, state, temperature=1.0):
        """Get policy target for training (visit count distribution)"""
        root = self.search(state, add_root_noise=True)
        if not root.children:
            return {}

        visits = np.array([c.visits for c in root.children], dtype=np.float32)
        moves = [c.move for c in root.children]

        if temperature == 0:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            total = visits.sum()
            if total <= 0:
                probs = np.ones_like(visits) / len(visits)
            else:
                probs = visits / total

        return dict(zip(moves, probs))


def save_model(net, n, iteration=None):
    """Save model weights to disk"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    if iteration is not None:
        filename = f"hexapawn_n{n}_iter{iteration}.pt"
    else:
        filename = f"hexapawn_n{n}_latest.pt"
    
    filepath = os.path.join(WEIGHTS_DIR, filename)
    torch.save({
        'model_state_dict': net.state_dict(),
        'n': n,
        'iteration': iteration
    }, filepath)
    print(f"Model saved to {filepath}")
    return filepath


def load_model(n, iteration=None):
    """Load model weights from disk"""
    if iteration is not None:
        filename = f"hexapawn_n{n}_iter{iteration}.pt"
    else:
        filename = f"hexapawn_n{n}_latest.pt"
    
    filepath = os.path.join(WEIGHTS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"No saved model found at {filepath}")
        return None
    
    checkpoint = torch.load(filepath)
    net = HexapawnNet(n)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {filepath}")
    if checkpoint.get('iteration') is not None:
        print(f"  Trained for {checkpoint['iteration']} iterations")
    
    return net


def find_latest_model(n):
    """Find the most recent model for a given board size"""
    if not os.path.exists(WEIGHTS_DIR):
        return None
    
    latest_path = os.path.join(WEIGHTS_DIR, f"hexapawn_n{n}_latest.pt")
    if os.path.exists(latest_path):
        return load_model(n)
    
    pattern = f"hexapawn_n{n}_iter"
    max_iter = -1
    
    for filename in os.listdir(WEIGHTS_DIR):
        if filename.startswith(pattern) and filename.endswith('.pt'):
            try:
                iter_str = filename[len(pattern):-3]
                iteration = int(iter_str)
                if iteration > max_iter:
                    max_iter = iteration
            except ValueError:
                continue
    
    if max_iter >= 0:
        return load_model(n, max_iter)
    
    return None
