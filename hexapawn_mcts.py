import numpy as np
import random
import math
from copy import deepcopy
from tqdm import trange

X, O, BLANK = 1, -1, 0
N = 5

class HexapawnState:
    def __init__(self, n=N):
        self.n = n
        self.board = np.full((n, n), BLANK, dtype=np.int8)
        self.board[0] = X
        self.board[-1] = O
        self.player = O

    def get_possible_moves(self):
        moves = []
        player_pieces = np.where(self.board == self.player)
        for r, c in zip(*player_pieces):
            if self.player == O:  # O moves up
                if r > 0:
                    if c > 0 and self.board[r-1, c-1] == X:
                        moves.append((r, c, r-1, c-1))
                    if self.board[r-1, c] == BLANK:
                        moves.append((r, c, r-1, c))
                    if c < self.n-1 and self.board[r-1, c+1] == X:
                        moves.append((r, c, r-1, c+1))
            else:  # X moves down
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
        if O in self.board[0] or not np.any(self.board == X):
            return O
        if X in self.board[-1] or not np.any(self.board == O):
            return X
        return BLANK

    def copy(self):
        new_state = HexapawnState.__new__(HexapawnState)
        new_state.n = self.n
        new_state.board = self.board.copy()
        new_state.player = self.player
        return new_state

    def display_board(self):
        print("  " + " ".join(str(i) for i in range(self.n)))
        for r in range(self.n):
            print(f"{r} {' '.join('X' if cell == X else 'O' if cell == O else '.' for cell in self.board[r])}")


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children = []
        self.visits = 0
        self.wins = 0  # Wins for the player who made the move to reach this node
        self.untried_moves = state.get_possible_moves()
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self.state.is_terminal()
    
    def best_child(self, exploration_weight=1.414):
        """Select best child using UCB1 formula"""
        # First prioritize any unvisited children
        for child in self.children:
            if child.visits == 0:
                return child
        
        # All children visited - use UCB1
        # At this point self.visits must be > 0 since children have been visited
        choices_weights = []
        for child in self.children:
            exploit = child.wins / child.visits
            explore = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            choices_weights.append(exploit + explore)
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self):
        """Expand tree by one child node"""
        move = self.untried_moves.pop()
        next_state = self.state.copy()
        next_state.make_move(move)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def update(self, result):
        """Update node statistics after simulation"""
        self.visits += 1
        self.wins += result


class MCTSAgent:
    def __init__(self, simulations=1000):
        self.simulations = simulations
    
    def search(self, root_state):
        """Run MCTS from root state and return best move"""
        root = MCTSNode(root_state.copy())
        
        for _ in trange(self.simulations):
            node = root
            state = root_state.copy()
            
            # Selection: traverse tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded() and node.visits > 0:
                node = node.best_child()
                state.make_move(node.move)
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
                state = node.state.copy()
            
            # Simulation: play random game to the end
            while not state.is_terminal():
                moves = state.get_possible_moves()
                state.make_move(random.choice(moves))
            
            # Backpropagation: update all nodes in path
            winner = state.get_winner()
            while node is not None:
                # Result is from perspective of player who moved to create this node
                if node.parent is not None:
                    # The player who moved to this node
                    mover = X if node.state.player == O else O
                    if winner == mover:
                        result = 1
                    elif winner == BLANK:
                        result = 0.5
                    else:
                        result = 0
                    node.update(result)
                node = node.parent
        
        # Return move with highest visit count (most robust)
        return max(root.children, key=lambda c: c.visits).move
    
    def choose_move(self, state):
        """Choose best move using MCTS"""
        return self.search(state)


def play_game(agent, human_player=X, n=N, show_stats=False):
    state = HexapawnState(n)
    print("Initial board:")
    state.display_board()
    print()

    while not state.is_terminal():
        if state.player == human_player:
            moves = state.get_possible_moves()
            print("Your turn. Available moves:")
            for i, move in enumerate(moves):
                print(f"{chr(97+i)}. From ({move[0]},{move[1]}) to ({move[2]},{move[3]})")
            
            while True:
                choice = input("Enter your move (a, b, c, ...): ").lower()
                if choice.isalpha() and ord(choice) - ord('a') < len(moves):
                    move = moves[ord(choice) - ord('a')]
                    break
                print("Invalid input. Please try again.")
        else:
            print("Agent thinking...")
            move = agent.choose_move(state)
            print(f"Agent's move: From ({move[0]},{move[1]}) to ({move[2]},{move[3]})")

        state.make_move(move)
        print("Current board:")
        state.display_board()
        print()

    winner = state.get_winner()
    if winner == BLANK:
        print("It's a draw!")
    else:
        print(f"Player {'X' if winner == X else 'O'} wins!")


def agent_vs_agent(agent1, agent2, n=N, num_games=100):
    """Play multiple games between two agents"""
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game_num in range(num_games):
        state = HexapawnState(n)
        
        while not state.is_terminal():
            if state.player == O:
                move = agent1.choose_move(state)
            else:
                move = agent2.choose_move(state)
            state.make_move(move)
        
        winner = state.get_winner()
        if winner == O:
            agent1_wins += 1
        elif winner == X:
            agent2_wins += 1
        else:
            draws += 1
        
        if (game_num + 1) % 10 == 0:
            print(f"Games {game_num + 1}/{num_games}: Agent1(O)={agent1_wins}, Agent2(X)={agent2_wins}, Draws={draws}")
    
    print(f"\nFinal Results:")
    print(f"Agent 1 (O) wins: {agent1_wins}")
    print(f"Agent 2 (X) wins: {agent2_wins}")
    print(f"Draws: {draws}")
    return agent1_wins, agent2_wins, draws


if __name__ == "__main__":
    # Create MCTS agent with 1000 simulations per move
    # More simulations = stronger play but slower
    mcts_agent = MCTSAgent(simulations=100_000)
    
    print("MCTS Hexapawn Agent")
    print("=" * 50)
    print(f"Simulations per move: {mcts_agent.simulations}")
    print()
    
    # Play against human
    play_game(mcts_agent, human_player=X, n=N)
    
    # Optional: Compare MCTS agents with different simulation counts
    # print("\n\nComparing MCTS agents:")
    # weak_agent = MCTSAgent(simulations=1000)
    # strong_agent = MCTSAgent(simulations=10000)
    # agent_vs_agent(strong_agent, weak_agent, n=N, num_games=10)