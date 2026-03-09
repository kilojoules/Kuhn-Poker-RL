#!/usr/bin/env python3
"""Generate a gauntlet heatmap from Kuhn Poker self-play checkpoints.

Loads strategy snapshots from metrics.jsonl, computes the exact expected
P1 payoff for every (checkpoint_i, checkpoint_j) pair, and plots the
resulting gauntlet matrix.
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


DEALS = [(i, j) for i in range(3) for j in range(3) if i != j]


def expected_p1_value(p1_strat: dict, p2_strat: dict) -> float:
    """Exact expected P1 payoff when p1_strat plays P1 and p2_strat plays P2.

    Each strategy is a dict with keys: p1_bet, p1_call, p2_bet, p2_call,
    each a list/array of length 3 (one per card: J=0, Q=1, K=2).
    """
    b1 = np.asarray(p1_strat["p1_bet"])    # P1 bet prob
    c1 = np.asarray(p1_strat["p1_call"])   # P1 call prob (facing bet)
    b2 = np.asarray(p2_strat["p2_bet"])    # P2 bet prob (after P1 check)
    c2 = np.asarray(p2_strat["p2_call"])   # P2 call prob (after P1 bet)

    ev = 0.0
    for card1, card2 in DEALS:
        w = 1.0 / 6.0  # uniform deal probability
        sign = 1.0 if card1 > card2 else -1.0

        # P1 bets
        bet_ev = (
            c2[card2] * sign * 2.0          # P2 calls -> showdown pot=4
            + (1.0 - c2[card2]) * 1.0       # P2 folds -> P1 wins ante
        )

        # P1 checks
        # P2 checks -> showdown pot=2
        check_check_ev = sign * 1.0
        # P2 bets -> P1 decides
        check_bet_ev = (
            c1[card1] * sign * 2.0           # P1 calls -> showdown pot=4
            + (1.0 - c1[card1]) * (-1.0)     # P1 folds -> P1 loses ante
        )
        check_ev = (
            (1.0 - b2[card2]) * check_check_ev
            + b2[card2] * check_bet_ev
        )

        ev += w * (b1[card1] * bet_ev + (1.0 - b1[card1]) * check_ev)

    return ev


def load_strategies(metrics_path: Path) -> list:
    """Load agent and opponent strategies from a metrics.jsonl file.

    Deduplicates entries by timestep (keeps first occurrence).
    """
    entries = []
    seen_ts = set()
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                ts = entry["timesteps"]
                if ts not in seen_ts:
                    seen_ts.add(ts)
                    entries.append(entry)
    return entries


def build_gauntlet(entries: list) -> tuple:
    """Build gauntlet matrix from strategy checkpoints.

    Uses the agent's strategy as "protagonist" and the opponent's strategy
    as "adversary" for each checkpoint. Evaluates protagonist_i vs adversary_j
    using the analytical expected value.

    Returns (matrix, labels) where matrix[i,j] = normalized score of
    protagonist i vs adversary j.
    """
    n = len(entries)
    # Extract strategies
    agent_strats = [e["agent_strategy"] for e in entries]
    opp_strats = [e["opponent_strategy"] for e in entries]

    # Gauntlet: agent checkpoint i (as P1) vs opponent checkpoint j (as P2)
    # and vice versa, averaged for symmetry
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Agent i as P1 vs opponent j as P2
            v1 = expected_p1_value(agent_strats[i], opp_strats[j])
            # Agent i as P2 vs opponent j as P1
            # For this, agent_i plays P2 role and opp_j plays P1 role
            # P2 payoff = -P1 payoff, so we negate
            v2 = -expected_p1_value(opp_strats[j], agent_strats[i])
            matrix[i, j] = (v1 + v2) / 2.0

    # Build labels from timesteps
    labels = [f"{e['timesteps'] // 1000}k" for e in entries]

    return matrix, labels


def plot_gauntlet(matrix: np.ndarray, labels: list, output_path: Path,
                  title: str = "Kuhn Poker: Self-Play Gauntlet"):
    """Plot gauntlet heatmap."""
    n = len(labels)

    # Symmetric colorscale centered at 0
    vlim = max(abs(matrix.min()), abs(matrix.max()))

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="equal", vmin=-vlim, vmax=vlim)

    # Subsample labels if too many
    step = max(1, n // 12)
    tick_idx = list(range(0, n, step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=9)

    ax.set_xlabel("Opponent Checkpoint (training games)", fontsize=11)
    ax.set_ylabel("Agent Checkpoint (training games)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="Agent Expected Value", shrink=0.8)
    cbar.ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Kuhn Poker gauntlet heatmap")
    parser.add_argument("--metrics-dir", type=str,
                        default="experiments/results/selfplay",
                        help="Directory containing seed*/metrics.jsonl files")
    parser.add_argument("--output", type=str,
                        default="experiments/results/gauntlet.png",
                        help="Output image path")
    parser.add_argument("--title", type=str,
                        default="Kuhn Poker: Self-Play Gauntlet",
                        help="Plot title")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    seed_files = sorted(metrics_dir.glob("seed*/metrics.jsonl"))
    if not seed_files:
        # Fall back to single file
        single = metrics_dir / "metrics.jsonl"
        if single.exists():
            seed_files = [single]
        else:
            print(f"No metrics files found in {metrics_dir}")
            sys.exit(1)

    # Build gauntlet for each seed and average
    matrices = []
    labels = None
    for sf in seed_files:
        entries = load_strategies(sf)
        print(f"Loaded {len(entries)} checkpoints from {sf}")
        matrix, lbls = build_gauntlet(entries)
        matrices.append(matrix)
        if labels is None:
            labels = lbls

    avg_matrix = np.mean(matrices, axis=0)
    print(f"Averaged {len(matrices)} seeds")
    print(f"Gauntlet matrix shape: {avg_matrix.shape}")
    print(f"Value range: [{avg_matrix.min():.4f}, {avg_matrix.max():.4f}]")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_gauntlet(avg_matrix, labels, output_path, title=args.title)


if __name__ == "__main__":
    main()
