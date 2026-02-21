#!/usr/bin/env python3
"""Analyze sweep results and generate plots for Kuhn Poker experiments."""
import argparse
import json
import re
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


def load_metrics(results_dir: Path) -> dict:
    """Load all experiment metrics from directory tree."""
    data = {}
    for metrics_file in sorted(results_dir.rglob("metrics.jsonl")):
        rel = metrics_file.relative_to(results_dir)
        parts = list(rel.parts)

        # Parse experiment name from directory structure
        if len(parts) >= 2:
            exp_name = parts[0]
            seed_dir = parts[1]
        else:
            exp_name = "unknown"
            seed_dir = parts[0] if parts else "seed0"

        seed_match = re.search(r"seed(\d+)", seed_dir)
        seed = int(seed_match.group(1)) if seed_match else 0

        lines = []
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))

        if not lines:
            continue

        if exp_name not in data:
            data[exp_name] = {}
        data[exp_name][seed] = lines

    return data


def get_final_exploitability(runs: dict) -> list:
    """Get final exploitability from each seed's run."""
    values = []
    for seed, lines in runs.items():
        if lines:
            values.append(lines[-1]["agent_exploitability"])
    return values


def parse_experiment_name(name: str):
    """Parse algo, strategy, A value from experiment directory name.

    Format: {algo}_{strategy}_A{value}
    e.g. ppo_uniform_A0.50, buffered_thompson_loss_A0.90
    """
    m = re.match(r"^(ppo|buffered)_(.+)_A([\d.]+)$", name)
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return None, None, None


def plot_a_curve(data: dict, output_dir: Path):
    """Plot exploitability vs A for PPO and Buffered (uniform)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, color, label in [("ppo", "tab:blue", "PPO (memoryless)"),
                                ("buffered", "tab:red", "Buffered (replay buffer)")]:
        a_vals = []
        means = []
        stds = []
        for name, runs in sorted(data.items()):
            a, strat, a_val = parse_experiment_name(name)
            if a == algo and strat == "uniform" and a_val is not None:
                vals = get_final_exploitability(runs)
                if vals:
                    a_vals.append(a_val)
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))

        if a_vals:
            ax.errorbar(a_vals, means, yerr=stds, marker="o", label=label,
                       color=color, capsize=3)

    # Add self-play baseline
    if "selfplay" in data:
        sp_vals = get_final_exploitability(data["selfplay"])
        if sp_vals:
            ax.axhline(np.mean(sp_vals), color="gray", linestyle="--",
                       label=f"Self-play (A=0): {np.mean(sp_vals):.4f}")

    ax.set_xlabel("A (zoo sampling probability)")
    ax.set_ylabel("Exploitability")
    ax.set_title("Kuhn Poker: Exploitability vs Zoo Sampling (A)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "a_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'a_curve.png'}")


def plot_thompson_comparison(data: dict, output_dir: Path):
    """Plot uniform vs Thompson vs Thompson-Loss at each A value."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    strat_configs = [
        ("uniform", "tab:blue", "o", "Uniform"),
        ("thompson", "tab:orange", "s", "Thompson"),
        ("thompson_loss", "tab:green", "^", "Thompson (loss-seeking)"),
    ]

    for ax, algo, title in [(axes[0], "ppo", "PPO"),
                             (axes[1], "buffered", "Buffered")]:
        for strat, color, marker, label in strat_configs:
            a_vals, means, stds = [], [], []
            for name, runs in sorted(data.items()):
                a, s, a_val = parse_experiment_name(name)
                if a == algo and s == strat and a_val is not None:
                    vals = get_final_exploitability(runs)
                    if vals:
                        a_vals.append(a_val)
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
            if a_vals:
                ax.errorbar(a_vals, means, yerr=stds, marker=marker,
                           label=label, color=color, capsize=3)

        ax.set_xlabel("A")
        ax.set_ylabel("Exploitability")
        ax.set_title(f"{title}: Sampling Strategy Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "thompson_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'thompson_comparison.png'}")


def plot_timeseries(data: dict, output_dir: Path):
    """Plot exploitability over training for select conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pick a few representative conditions
    targets = [
        ("selfplay", "Self-play (A=0)", "gray"),
        ("ppo_uniform_A0.05", "PPO A=0.05", "tab:blue"),
        ("ppo_uniform_A0.50", "PPO A=0.50", "tab:green"),
        ("ppo_uniform_A0.90", "PPO A=0.90", "tab:red"),
        ("ppo_thompson_A0.50", "PPO Thompson A=0.50", "tab:orange"),
        ("ppo_thompson_loss_A0.50", "PPO Thompson-Loss A=0.50", "tab:purple"),
    ]

    for name, label, color in targets:
        if name not in data:
            continue
        all_ts = {}
        for seed, lines in data[name].items():
            for entry in lines:
                t = entry["timesteps"]
                if t not in all_ts:
                    all_ts[t] = []
                all_ts[t].append(entry["agent_exploitability"])

        if all_ts:
            ts = sorted(all_ts.keys())
            means = [np.mean(all_ts[t]) for t in ts]
            ax.plot(ts, means, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Games played")
    ax.set_ylabel("Exploitability")
    ax.set_title("Kuhn Poker: Exploitability Over Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'timeseries.png'}")


def plot_strategy_heatmap(data: dict, output_dir: Path):
    """Plot final strategy probabilities across conditions."""
    nash_alpha_third = {
        "p1_bet": [1/3, 0, 1], "p1_call": [0, 2/3, 1],
        "p2_bet": [1/3, 0, 1], "p2_call": [0, 1/3, 1],
    }

    conditions = []
    strategies = []
    for name in sorted(data.keys()):
        a, strat, a_val = parse_experiment_name(name)
        if a == "ppo" and strat == "uniform" and a_val is not None:
            all_strats = []
            for seed, lines in data[name].items():
                if lines and "agent_strategy" in lines[-1]:
                    s = lines[-1]["agent_strategy"]
                    vec = (s["p1_bet"] + s["p1_call"] + s["p2_bet"] + s["p2_call"])
                    all_strats.append(vec)
            if all_strats:
                conditions.append(f"A={a_val}")
                strategies.append(np.mean(all_strats, axis=0))

    if not conditions:
        return

    # Add Nash reference
    nash_vec = (list(nash_alpha_third["p1_bet"]) + list(nash_alpha_third["p1_call"]) +
                list(nash_alpha_third["p2_bet"]) + list(nash_alpha_third["p2_call"]))
    conditions.append("Nash")
    strategies.append(nash_vec)

    fig, ax = plt.subplots(figsize=(12, max(4, len(conditions) * 0.5 + 1)))
    mat = np.array(strategies)

    labels = ["P1 bet J", "P1 bet Q", "P1 bet K",
              "P1 call J", "P1 call Q", "P1 call K",
              "P2 bet J", "P2 bet Q", "P2 bet K",
              "P2 call J", "P2 call Q", "P2 call K"]

    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="P(aggressive action)")
    ax.set_title("Strategy Profiles: Learned vs Nash")
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'strategy_heatmap.png'}")


def print_summary_tables(data: dict):
    """Print summary tables to stdout."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    # Self-play baseline
    if "selfplay" in data:
        vals = get_final_exploitability(data["selfplay"])
        if vals:
            print(f"\nSelf-play (A=0): {np.mean(vals):.4f} +/- {np.std(vals):.4f} "
                  f"({len(vals)} seeds)")

    # PPO uniform table
    print("\nPPO (uniform sampling):")
    print(f"{'A':>6s} | {'Exploitability':>20s} | {'Seeds':>5s}")
    print("-" * 40)
    for name in sorted(data.keys()):
        a, strat, a_val = parse_experiment_name(name)
        if a == "ppo" and strat == "uniform" and a_val is not None:
            vals = get_final_exploitability(data[name])
            if vals:
                print(f"{a_val:>6.2f} | {np.mean(vals):>8.4f} +/- {np.std(vals):<8.4f} | {len(vals):>5d}")

    # Buffered uniform table
    print("\nBuffered (uniform sampling):")
    print(f"{'A':>6s} | {'Exploitability':>20s} | {'Seeds':>5s}")
    print("-" * 40)
    for name in sorted(data.keys()):
        a, strat, a_val = parse_experiment_name(name)
        if a == "buffered" and strat == "uniform" and a_val is not None:
            vals = get_final_exploitability(data[name])
            if vals:
                print(f"{a_val:>6.2f} | {np.mean(vals):>8.4f} +/- {np.std(vals):<8.4f} | {len(vals):>5d}")

    # 3-way Thompson comparison (PPO)
    print("\nSampling Strategy Comparison (PPO):")
    print(f"{'A':>6s} | {'Uniform':>20s} | {'Thompson':>20s} | {'Thompson-Loss':>20s}")
    print("-" * 80)
    for a_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        u_name = f"ppo_uniform_A{a_val:.2f}"
        t_name = f"ppo_thompson_A{a_val:.2f}"
        tl_name = f"ppo_thompson_loss_A{a_val:.2f}"
        parts = []
        for name in [u_name, t_name, tl_name]:
            if name in data:
                vals = get_final_exploitability(data[name])
                if vals:
                    parts.append(f"{np.mean(vals):>8.4f} +/- {np.std(vals):<8.4f}")
                else:
                    parts.append("       N/A          ")
            else:
                parts.append("       N/A          ")
        print(f"{a_val:>6.2f} | {parts[0]} | {parts[1]} | {parts[2]}")

    # 3-way Thompson comparison (Buffered)
    print("\nSampling Strategy Comparison (Buffered):")
    print(f"{'A':>6s} | {'Uniform':>20s} | {'Thompson':>20s} | {'Thompson-Loss':>20s}")
    print("-" * 80)
    for a_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        u_name = f"buffered_uniform_A{a_val:.2f}"
        t_name = f"buffered_thompson_A{a_val:.2f}"
        tl_name = f"buffered_thompson_loss_A{a_val:.2f}"
        parts = []
        for name in [u_name, t_name, tl_name]:
            if name in data:
                vals = get_final_exploitability(data[name])
                if vals:
                    parts.append(f"{np.mean(vals):>8.4f} +/- {np.std(vals):<8.4f}")
                else:
                    parts.append("       N/A          ")
            else:
                parts.append("       N/A          ")
        print(f"{a_val:>6.2f} | {parts[0]} | {parts[1]} | {parts[2]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Kuhn Poker experiments")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from {results_dir}...")
    data = load_metrics(results_dir)
    print(f"Found {len(data)} experiment conditions")

    if not data:
        print("No data found!")
        sys.exit(1)

    print_summary_tables(data)
    plot_a_curve(data, results_dir)
    plot_thompson_comparison(data, results_dir)
    plot_timeseries(data, results_dir)
    plot_strategy_heatmap(data, results_dir)


if __name__ == "__main__":
    main()
