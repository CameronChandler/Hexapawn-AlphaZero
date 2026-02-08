"""
Training script for AlphaZero Hexapawn
"""

import torch
import torch.optim as optim
import random
from collections import deque
from tqdm import trange
import argparse
import json
import os
from datetime import datetime
import math

from hexapawn_core import (
    HexapawnState, HexapawnNet, AlphaZeroAgent,
    save_model, find_latest_model, X, O
)


def evaluate_agent(agent, num_games=50, n=3):
    """
    Evaluate agent by playing against a random player
    Agent plays both sides vs random.
    Returns win rate, loss rate
    """
    wins = 0
    
    for game_idx in range(num_games):
        agent_player = O if (game_idx % 2 == 0) else X
        state = HexapawnState(n)
        
        while not state.is_terminal():
            if state.player == agent_player:
                # Agent move
                move = agent.choose_move(state, temperature=0)
            else:
                # Random opponent move
                moves = state.get_possible_moves()
                move = random.choice(moves)
            
            state.make_move(move)
        
        winner = state.get_winner()
        if winner == agent_player:
            wins += 1
    
    return wins / num_games


def agent_vs_agent(agent1, agent2, num_games=50, n=3):
    """
    Play two agents against each other
    Returns: (agent1_wins, agent2_wins)
    """
    agent1_wins = 0
    agent2_wins = 0
    
    for _ in range(num_games):
        state = HexapawnState(n)
        
        while not state.is_terminal():
            if state.player == O:
                move = agent1.choose_move(state, temperature=0)
            else:
                move = agent2.choose_move(state, temperature=0)
            state.make_move(move)
        
        winner = state.get_winner()
        if winner == O:
            agent1_wins += 1
        elif winner == X:
            agent2_wins += 1
    
    return agent1_wins, agent2_wins


def save_training_stats(stats, n):
    """Save training statistics to JSON file"""
    os.makedirs("stats", exist_ok=True)
    filepath = os.path.join("stats", f"training_stats_n{n}.json")
    
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Training statistics saved to {filepath}")


def load_training_stats(n):
    """Load existing training statistics"""
    filepath = os.path.join("stats", f"training_stats_n{n}.json")
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            stats = json.load(f)
        return stats
    
    return {
        'iterations': [],
        'win_rates': [],
        'avg_losses': [],
        'policy_losses': [],
        'value_losses': [],
        'timestamp': datetime.now().isoformat()
    }


def self_play_game(agent, temperature=1.0, n=3, debug_value=False):
    """Play one game of self-play and collect training data"""
    state = HexapawnState(n)
    examples = []
    
    while not state.is_terminal():
        policy_target = agent.get_policy_target(state, temperature)
        examples.append((state.copy(), policy_target, None))
        move = agent.choose_move(state, temperature)
        # state.display_board()
        state.make_move(move)
    
    # Fill in game outcome for all examples
    winner = state.get_winner()
    training_examples = []
    start_value = None
    for i, (s, policy, _) in enumerate(examples):
        # Value from perspective of player at this state.
        value = 1 if winner == s.player else -1
        if value not in (-1, 1):
            print("Warning: non-binary value target produced.")
        if i == 0:
            start_value = value
        training_examples.append((s, policy, value))
    
    if debug_value:
        return training_examples, start_value, winner
    return training_examples


def train_network(net, examples, epochs=10, batch_size=32, n=3):
    """Train neural network on self-play data"""
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    total_avg_loss = 0
    total_policy_loss = 0
    total_value_loss = 0

    # Target diagnostics
    value_targets_all = [v for _, _, v in examples]
    if value_targets_all:
        pos = sum(1 for v in value_targets_all if v > 0)
        neg = sum(1 for v in value_targets_all if v < 0)
        print(f"Target value distribution: +1={pos}, -1={neg}")
    policy_nonzero = []
    policy_entropy = []
    for _, policy, _ in examples:
        if not policy:
            continue
        probs = list(policy.values())
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        nz = sum(1 for p in probs if p > 0)
        policy_nonzero.append(nz)
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        policy_entropy.append(entropy)
    if policy_nonzero:
        avg_nz = sum(policy_nonzero) / len(policy_nonzero)
        avg_ent = sum(policy_entropy) / len(policy_entropy)
        print(f"Target policy: avg_nonzero={avg_nz:.1f}, avg_entropy={avg_ent:.3f}")
    
    for epoch in range(epochs):
        random.shuffle(examples)
        total_loss = 0
        epoch_policy_loss = 0
        epoch_value_loss = 0
        batches = 0
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            states = torch.cat([s.to_tensor(canonical=True) for s, _, _ in batch])
            policy_targets = []
            value_targets = []
            
            for state, policy, value in batch:
                # Convert policy dict to tensor
                policy_tensor = torch.zeros(n * n * 3)
                for move, prob in policy.items():
                    idx = state.move_to_policy_index(move, canonical=True)
                    if idx is None:
                        continue
                    policy_tensor[idx] = float(prob)
                if torch.isnan(policy_tensor).any():
                    print("Warning: NaN in policy target tensor.")
                if policy_tensor.sum().item() == 0:
                    print("Warning: empty policy target tensor (sum=0).")
                policy_targets.append(policy_tensor)
                value_targets.append(value)
            
            policy_targets = torch.stack(policy_targets)
            value_targets = torch.FloatTensor(value_targets).unsqueeze(1)
            
            # Forward pass
            policy_out, value_out = net(states)
            
            # Loss
            policy_loss = -torch.sum(policy_targets * torch.log_softmax(policy_out, dim=1)) / len(batch)
            value_loss = torch.mean((value_out - value_targets) ** 2)
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            batches += 1
        
        avg_loss = total_loss / batches
        total_avg_loss += avg_loss
        total_policy_loss += epoch_policy_loss / batches
        total_value_loss += epoch_value_loss / batches
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f} (Policy: {epoch_policy_loss/batches:.4f}, Value: {epoch_value_loss/batches:.4f})")
    
    return total_avg_loss / epochs, total_policy_loss / epochs, total_value_loss / epochs


def train_alphazero(iterations=10, games_per_iteration=50, simulations=100, n=3, 
                    resume=True, eval_games=50, debug=False):
    '''
    If resume=True, load the latest saved model weights and continue training
    from the recorded iteration number. Optimizer state and replay buffer
    are NOT restored.
    '''
    
    # Load or create training statistics
    stats = load_training_stats(n) if resume else {
        'iterations': [],
        'win_rates': [],
        'avg_losses': [],
        'policy_losses': [],
        'value_losses': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Try to load existing model
    net = None
    start_iteration = 0
    baseline_agent = None
    
    if resume:
        net = find_latest_model(n)
        if net is not None:
            print(f"Resuming training from existing model for n={n}")
            latest_path = os.path.join("weights", f"hexapawn_n{n}_latest.pt")
            if os.path.exists(latest_path):
                checkpoint = torch.load(latest_path)
                start_iteration = checkpoint.get('iteration') or 0
                print(f"Starting from iteration {start_iteration}")
                
                # Keep a copy as baseline for comparison
                baseline_net = HexapawnNet(n)
                baseline_net.load_state_dict(net.state_dict())
                baseline_agent = AlphaZeroAgent(baseline_net, simulations=simulations)
    
    if net is None:
        print(f"Creating new model for n={n}")
        net = HexapawnNet(n)
    
    agent = AlphaZeroAgent(net, simulations=simulations)
    replay_buffer = deque(maxlen=10000)
    
    print("\nAlphaZero Training")
    print("=" * 60)
    print(f"Board size: {n}x{n}")
    print(f"Iterations: {iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"Simulations per move: {simulations}")
    print(f"Evaluation games: {eval_games}")
    print("=" * 60)
    
    for i in range(iterations):
        iteration = start_iteration + i + 1
        print(f"\nIteration {iteration}/{start_iteration + iterations}")
        print("-" * 60)
        
        # Self-play
        print("Generating self-play games...")
        start_values = []
        self_play_wins = {X: 0, O: 0}
        for game in trange(games_per_iteration):
            if debug:
                examples, start_value, winner = self_play_game(agent, temperature=1.0, n=n, debug_value=True)
                if start_value is not None:
                    start_values.append(start_value)
                if winner in self_play_wins:
                    self_play_wins[winner] += 1
            else:
                examples = self_play_game(agent, temperature=1.0, n=n)
            replay_buffer.extend(examples)
        
        # Train network
        print(f"Training on {len(replay_buffer)} examples...")
        avg_loss, policy_loss, value_loss = train_network(net, list(replay_buffer), epochs=10, n=n)
        
        # Evaluation against random player
        print("\nEvaluating against random player...")
        win_rate = evaluate_agent(agent, num_games=eval_games, n=n)
        
        print(f"Results vs Random:")
        print(f"  Win Rate:  {win_rate*100:.1f}%")

        if debug:
            print("\nMCTS diagnostics (start state):")
            start_state = HexapawnState(n)
            root = agent.search(start_state)
            visits = [c.visits for c in root.children]
            if visits:
                total_visits = sum(visits)
                probs = [v / total_visits for v in visits]
                entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                print(f"  Root children: {len(visits)}")
                print(f"  Visit entropy: {entropy:.3f}")
                print(f"  Visit range: min={min(visits)}, max={max(visits)}")
            else:
                print("  Warning: no root children found (no legal moves?)")

            with torch.no_grad():
                _, value_tensor = agent.net(start_state.to_tensor(canonical=True))
            print(f"  Net value(start): {value_tensor.item():+.3f}")
            if start_values:
                avg_start_value = sum(start_values) / len(start_values)
                print(f"  Self-play start value avg: {avg_start_value:+.3f}")
            total_self_play = self_play_wins[X] + self_play_wins[O]
            if total_self_play > 0:
                print(f"  Self-play winners: O={self_play_wins[O]}, X={self_play_wins[X]}")
        
        # Compare with baseline if available
        if baseline_agent is not None:
            print("\nComparing with baseline agent...")
            new_wins, baseline_wins = agent_vs_agent(
                agent, baseline_agent, num_games=20, n=n
            )
            print(f"New agent vs Baseline: {new_wins}-{baseline_wins} (W-L)")
            
            # Update baseline if new agent is better
            if new_wins > baseline_wins:
                print("New agent is stronger! Updating baseline.")
                baseline_net = HexapawnNet(n)
                baseline_net.load_state_dict(net.state_dict())
                baseline_agent = AlphaZeroAgent(baseline_net, simulations=simulations)
        
        # Save statistics
        stats['iterations'].append(iteration)
        stats['win_rates'].append(win_rate)
        stats['avg_losses'].append(avg_loss)
        stats['policy_losses'].append(policy_loss)
        stats['value_losses'].append(value_loss)
        
        save_training_stats(stats, n)
        
        # Save checkpoint every iteration
        save_model(net, n, iteration)
        save_model(net, n, iteration=None)
        
        print(f"\nIteration {iteration} complete.")
        print(f"  Buffer size: {len(replay_buffer)}")
        print(f"  Avg training loss: {avg_loss:.4f}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete - Summary")
    print("=" * 60)
    
    if len(stats['win_rates']) >= 2:
        initial_wr = stats['win_rates'][0] * 100
        final_wr = stats['win_rates'][-1] * 100
        improvement = final_wr - initial_wr
        
        print(f"Initial win rate vs random: {initial_wr:.1f}%")
        print(f"Final win rate vs random:   {final_wr:.1f}%")
        print(f"Improvement:                {improvement:+.1f}%")
    
    print(f"\nStatistics saved to: stats/training_stats_n{n}.json")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train AlphaZero Hexapawn')
    parser.add_argument('--n', type=int, default=4,
                        help='Board size (default: 4)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Training iterations (default: 10)')
    parser.add_argument('--games', type=int, default=50,
                        help='Games per iteration (default: 50)')
    parser.add_argument('--simulations', type=int, default=50,
                        help='MCTS simulations per move (default: 50)')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Number of evaluation games (default: 50)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Print MCTS diagnostics each iteration')
    
    args = parser.parse_args()
    
    train_alphazero(
        iterations=args.iterations,
        games_per_iteration=args.games,
        simulations=args.simulations,
        n=args.n,
        resume=args.resume,
        eval_games=args.eval_games,
        debug=args.debug
    )
