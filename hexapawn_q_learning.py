import numpy as np
import random
from tqdm import trange

X, O, BLANK = 1, -1, 0

N = 5

class HexapawnState:
    def __init__(self, n=N):
        self.n = n
        self.board = np.full((n, n), BLANK, dtype=np.int8)
        self.board[0] = X  # X pieces
        self.board[-1] = O  # O pieces
        self.player = O  # O starts

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
        if O in self.board[0]:
            return O
        if X in self.board[-1]:
            return X
        if not np.any(self.board == O):
            return X
        if not np.any(self.board == X):
            return O
        return BLANK  # Draw
    
    def display_board(self):
        print("  " + " ".join(str(i) for i in range(self.n)))
        for r in range(self.n):
            print(f"{r} {' '.join('X' if cell == X else 'O' if cell == O else '.' for cell in self.board[r])}")

class HexapawnAgent:
    def __init__(self):
        self.q_table = {}

    def get_state_key(self, state):
        return (state.board.tobytes(), state.player)
        # return str(state.board.flatten().tolist()) + str(state.player)
    
    def get_q_value(self, state, move):
        state_key = self.get_state_key(state)
        return self.q_table.get(state_key, {}).get(move, 0.0)

    def update_q_value(self, state, move, new_q_value):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][move] = new_q_value

    def choose_move(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(state.get_possible_moves())
        else:
            return max(state.get_possible_moves(), key=lambda m: self.get_q_value(state, m))

def train(episodes, epsilon, alpha, gamma, n=N):
    agent = HexapawnAgent()
    
    for _ in trange(episodes):
        state = HexapawnState(n)
        while not state.is_terminal():
            move = agent.choose_move(state, epsilon)
            old_q = agent.get_q_value(state, move)
            
            state.make_move(move)
            
            if state.is_terminal():
                reward = 1 if state.get_winner() == O else -1
                new_q = old_q + alpha * (reward - old_q)
            else:
                best_next_move = max(state.get_possible_moves(), key=lambda m: agent.get_q_value(state, m))
                max_q_next = agent.get_q_value(state, best_next_move)
                new_q = old_q + alpha * (gamma * max_q_next - old_q)
            
            agent.update_q_value(state, move, new_q)
    
    return agent

def play_game(agent, human_player=X, n=N):
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
            move = agent.choose_move(state, epsilon=0)
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

if __name__ == "__main__":
    # Train the agent
    trained_agent = train(episodes=100_000, epsilon=0.1, alpha=0.1, gamma=0.9, n=N)

    # Play a game against the trained agent
    play_game(trained_agent)