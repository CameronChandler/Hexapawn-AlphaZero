"""
Plot training statistics for AlphaZero Hexapawn
"""

import json
import os
import matplotlib.pyplot as plt
import argparse


def plot_training_stats(n=3):
    """Plot training statistics from JSON file"""
    filepath = os.path.join("stats", f"training_stats_n{n}.json")
    
    if not os.path.exists(filepath):
        print(f"No training stats found for board size {n}x{n}")
        print(f"Expected file: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        stats = json.load(f)
    
    iterations = stats['iterations']
    win_rates = [wr * 100 for wr in stats['win_rates']]
    draw_rates = [dr * 100 for dr in stats['draw_rates']]
    loss_rates = [lr * 100 for lr in stats['loss_rates']]
    avg_losses = stats['avg_losses']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'AlphaZero Hexapawn Training Progress ({n}x{n} Board)', fontsize=14, fontweight='bold')
    
    # Plot 1: Performance vs Random Player
    ax1.plot(iterations, win_rates,  'g-o', label='Win Rate',  linewidth=2, markersize=6)
    ax1.plot(iterations, draw_rates, 'b-s', label='Draw Rate', linewidth=2, markersize=6)
    ax1.plot(iterations, loss_rates, 'r-^', label='Loss Rate', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Rate (%)', fontsize=11)
    ax1.set_title('Performance vs Random Player', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Add horizontal line at 50% for reference
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    
    # Plot 2: Training Loss
    ax2.plot(iterations, avg_losses, color='C0', linewidth=2, marker='o', markersize=6)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Average Loss', fontsize=11)
    ax2.set_title('Training Loss Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f"training_progress_n{n}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    plt.show()


def print_summary(n=3):
    """Print text summary of training progress"""
    filepath = os.path.join("stats", f"training_stats_n{n}.json")
    
    if not os.path.exists(filepath):
        print(f"No training stats found for board size {n}x{n}")
        return
    
    with open(filepath, 'r') as f:
        stats = json.load(f)
    
    print("\n" + "=" * 60)
    print(f"Training Summary - {n}x{n} Board")
    print("=" * 60)
    
    if len(stats['iterations']) > 0:
        print(f"\nTotal iterations: {stats['iterations'][-1]}")
        print(f"Training started: {stats.get('timestamp', 'Unknown')}")
        
        print("\n--- Performance vs Random Player ---")
        print(f"Initial: Win={stats['win_rates'][0]*100:.1f}% Draw={stats['draw_rates'][0]*100:.1f}% Loss={stats['loss_rates'][0]*100:.1f}%")
        print(f"Final:   Win={stats['win_rates'][-1]*100:.1f}% Draw={stats['draw_rates'][-1]*100:.1f}% Loss={stats['loss_rates'][-1]*100:.1f}%")
        
        improvement = (stats['win_rates'][-1] - stats['win_rates'][0]) * 100
        print(f"\nWin rate improvement: {improvement:+.1f}%")
        
        print("\n--- Training Loss ---")
        print(f"Initial loss: {stats['avg_losses'][0]:.4f}")
        print(f"Final loss:   {stats['avg_losses'][-1]:.4f}")
        print(f"Best loss:    {min(stats['avg_losses']):.4f}")
        
        # Find iteration with best performance
        best_iter = stats['iterations'][stats['win_rates'].index(max(stats['win_rates']))]
        print(f"\nBest win rate achieved at iteration: {best_iter}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import numpy as np  # Import here for trend line
    
    parser = argparse.ArgumentParser(description='Plot AlphaZero training statistics')
    parser.add_argument('--n', type=int, default=5,
                        help='Board size (default: 5)')
    parser.add_argument('--summary-only', action='store_true',
                        help='Print summary without plotting')
    
    args = parser.parse_args()
    
    if args.summary_only:
        print_summary(args.n)
    else:
        print_summary(args.n)
        plot_training_stats(args.n)