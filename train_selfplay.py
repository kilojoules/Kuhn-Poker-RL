#!/usr/bin/env python3
"""Self-play baseline for Kuhn Poker (no zoo, A=0)."""
import argparse
import json
import numpy as np
from pathlib import Path

from kuhn_env import play_games, get_strategy, exploitability, action_entropy
from ppo import PPOAgent, PPOConfig


def train_selfplay(
    timesteps: int = 200_000,
    games_per_step: int = 512,
    log_interval: int = 50,
    output_dir: str = "experiments/results/selfplay",
    seed: int = 0,
    cfg: PPOConfig = None,
):
    np.random.seed(seed)

    if cfg is None:
        cfg = PPOConfig()
    agent = PPOAgent(cfg)
    opponent = PPOAgent(cfg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "metrics.jsonl"

    total_games = 0
    update_step = 0

    while total_games < timesteps:
        half = games_per_step // 2

        # Agent as P1
        p1_rec, p2_rec_opp, p1_rew, p2_rew = play_games(agent.act, opponent.act, half)
        # Agent as P2
        p1_rec_opp, p2_rec, p1_rew2, p2_rew2 = play_games(opponent.act, agent.act, half)

        # Collect agent decisions
        agent_obs, agent_acts, agent_logps, agent_rewards = [], [], [], []
        for obs, acts, logps, idx in p1_rec:
            agent_obs.append(obs); agent_acts.append(acts)
            agent_logps.append(logps); agent_rewards.append(p1_rew[idx])
        for obs, acts, logps, idx in p2_rec:
            agent_obs.append(obs); agent_acts.append(acts)
            agent_logps.append(logps); agent_rewards.append(p2_rew2[idx])

        # Collect opponent decisions
        opp_obs, opp_acts, opp_logps, opp_rewards = [], [], [], []
        for obs, acts, logps, idx in p2_rec_opp:
            opp_obs.append(obs); opp_acts.append(acts)
            opp_logps.append(logps); opp_rewards.append(p2_rew[idx])
        for obs, acts, logps, idx in p1_rec_opp:
            opp_obs.append(obs); opp_acts.append(acts)
            opp_logps.append(logps); opp_rewards.append(p1_rew2[idx])

        # Update agent
        a_obs = np.concatenate(agent_obs)
        a_acts = np.concatenate(agent_acts)
        a_logps = np.concatenate(agent_logps)
        a_rew = np.concatenate(agent_rewards)
        agent.update(a_obs, a_acts, a_rew, a_logps)

        # Update opponent
        o_obs = np.concatenate(opp_obs)
        o_acts = np.concatenate(opp_acts)
        o_logps = np.concatenate(opp_logps)
        o_rew = np.concatenate(opp_rewards)
        opponent.update(o_obs, o_acts, o_rew, o_logps)

        total_games += games_per_step
        update_step += 1

        if update_step % log_interval == 0:
            strat = get_strategy(agent.action_probs)
            opp_strat = get_strategy(opponent.action_probs)
            expl = exploitability(strat)
            opp_expl = exploitability(opp_strat)

            metrics = {
                "update": update_step,
                "timesteps": total_games,
                "agent_exploitability": expl,
                "opponent_exploitability": opp_expl,
                "agent_entropy": action_entropy(strat),
                "agent_strategy": {k: v.tolist() for k, v in strat.items()},
                "opponent_strategy": {k: v.tolist() for k, v in opp_strat.items()},
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            print(
                f"[{total_games:>8d}] expl={expl:.4f} opp_expl={opp_expl:.4f} "
                f"p1_bet={strat['p1_bet'].round(3)}"
            )

    print(f"\nDone. Metrics saved to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Kuhn Poker self-play")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--games-per-step", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="experiments/results/selfplay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--train-iters", type=int, default=4)
    args = parser.parse_args()

    cfg = PPOConfig(
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        hidden=args.hidden,
        clip_ratio=args.clip_ratio,
        train_iters=args.train_iters,
    )
    train_selfplay(
        timesteps=args.timesteps,
        games_per_step=args.games_per_step,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
