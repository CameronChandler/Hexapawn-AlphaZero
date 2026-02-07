"""
Play script for AlphaZero Hexapawn
"""

import argparse
from hexapawn_core import (
    HexapawnState, AlphaZeroAgent, find_latest_model,
    X, O
)


def play_game(agent, human_player=X, n=3):
    """Play interactive game against trained agent"""
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
            move = agent.choose_move(state, temperature=0)
            print(f"Agent's move: From ({move[0]},{move[1]}) to ({move[2]},{move[3]})")

        state.make_move(move)
        print("Current board:")
        state.display_board()
        print()

    winner = state.get_winner()
    print(f"Player {'X' if winner == X else 'O'} wins!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play AlphaZero Hexapawn')
    parser.add_argument('--n', type=int, default=5,
                        help='Board size (default: 3)')
    parser.add_argument('--simulations', type=int, default=100,
                        help='MCTS simulations per move (default: 100)')
    parser.add_argument('--play-as', type=str, default='X', choices=['X', 'O'],
                        help='Play as X or O (default: X)')
    
    args = parser.parse_args()
    
    # Load trained model
    net = find_latest_model(args.n)
    
    if net is None:
        print(f"\nNo trained model found for board size {args.n}x{args.n}")
        print("Please train a model first:")
        print(f"  python train.py --n {args.n}")
        exit(1)
    
    agent = AlphaZeroAgent(net, simulations=args.simulations)
    
    human_player = X if args.play_as == 'X' else O
    
    print("\n" + "=" * 60)
    print(f"Playing against trained agent (board size: {args.n}x{args.n})")
    print(f"You are playing as: {args.play_as}")
    print(f"Agent using {args.simulations} simulations per move")
    print("=" * 60)
    print()
    
    play_game(agent, human_player=human_player, n=args.n)
