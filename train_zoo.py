#!/usr/bin/env python3
"""Zoo-based training for Kuhn Poker (PPO agent)."""
import argparse
import json
import numpy as np
from pathlib import Path

from kuhn_env import play_games, get_strategy, exploitability, action_entropy
from ppo import PPOAgent, PPOConfig
from zoo import OpponentZoo, a_schedule


def train_zoo(
    a_value: float = 0.1,
    timesteps: int = 200_000,
    games_per_step: int = 512,
    zoo_update_interval: int = 10,
    zoo_max_size: int = 50,
    log_interval: int = 50,
    output_dir: str = "experiments/results/zoo",
    seed: int = 0,
    cfg: PPOConfig = None,
    sampling_strategy: str = "uniform",
    competitiveness_threshold: float = 0.3,
    a_schedule_type: str = "constant",
    a_halflife: float = 0.25,
):
    np.random.seed(seed)

    if cfg is None:
        cfg = PPOConfig()
    agent = PPOAgent(cfg)
    latest_opponent = PPOAgent(cfg)

    opponent_zoo = OpponentZoo(
        cfg, max_size=zoo_max_size,
        sampling_strategy=sampling_strategy,
        competitiveness_threshold=competitiveness_threshold,
    )
    opponent_zoo.add(latest_opponent, update=0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "metrics.jsonl"

    total_games = 0
    update_step = 0

    while total_games < timesteps:
        # Current A value
        if a_schedule_type != "constant":
            current_a = a_schedule(total_games, timesteps, a_schedule_type, a_halflife)
        else:
            current_a = a_value

        # Pick opponent
        zoo_idx = None
        if len(opponent_zoo) > 0 and np.random.random() < current_a:
            current_opponent, zoo_idx = opponent_zoo.sample()
        else:
            current_opponent = latest_opponent

        half = games_per_step // 2

        # Play games: agent as P1 and as P2
        p1_rec, p2_rec_opp, p1_rew, p2_rew = play_games(
            agent.act, current_opponent.act, half
        )
        p1_rec_opp, p2_rec, p1_rew2, p2_rew2 = play_games(
            current_opponent.act, agent.act, half
        )

        # Thompson update
        if zoo_idx is not None:
            mean_rew = (p1_rew.mean() - p2_rew2.mean()) / 2.0
            opponent_zoo.update_outcome(zoo_idx, mean_rew)

        # Collect agent decisions
        agent_obs, agent_acts, agent_logps, agent_rewards = [], [], [], []
        for obs, acts, logps, idx in p1_rec:
            agent_obs.append(obs); agent_acts.append(acts)
            agent_logps.append(logps); agent_rewards.append(p1_rew[idx])
        for obs, acts, logps, idx in p2_rec:
            agent_obs.append(obs); agent_acts.append(acts)
            agent_logps.append(logps); agent_rewards.append(p2_rew2[idx])

        a_obs = np.concatenate(agent_obs)
        a_acts = np.concatenate(agent_acts)
        a_logps = np.concatenate(agent_logps)
        a_rew = np.concatenate(agent_rewards)
        agent.update(a_obs, a_acts, a_rew, a_logps)

        # Update latest opponent only when it was selected
        if current_opponent is latest_opponent:
            opp_obs, opp_acts, opp_logps, opp_rewards = [], [], [], []
            for obs, acts, logps, idx in p2_rec_opp:
                opp_obs.append(obs); opp_acts.append(acts)
                opp_logps.append(logps); opp_rewards.append(p2_rew[idx])
            for obs, acts, logps, idx in p1_rec_opp:
                opp_obs.append(obs); opp_acts.append(acts)
                opp_logps.append(logps); opp_rewards.append(p1_rew2[idx])

            o_obs = np.concatenate(opp_obs)
            o_acts = np.concatenate(opp_acts)
            o_logps = np.concatenate(opp_logps)
            o_rew = np.concatenate(opp_rewards)
            latest_opponent.update(o_obs, o_acts, o_rew, o_logps)

        total_games += games_per_step
        update_step += 1

        # Add to zoo periodically
        if update_step % zoo_update_interval == 0:
            opponent_zoo.add(latest_opponent, update=update_step)

        if update_step % log_interval == 0:
            strat = get_strategy(agent.action_probs)
            opp_strat = get_strategy(latest_opponent.action_probs)
            expl = exploitability(strat)

            metrics = {
                "update": update_step,
                "timesteps": total_games,
                "a_value": current_a,
                "a_schedule_type": a_schedule_type,
                "sampling_strategy": sampling_strategy,
                "zoo_size": len(opponent_zoo),
                "agent_exploitability": expl,
                "opponent_exploitability": exploitability(opp_strat),
                "agent_entropy": action_entropy(strat),
                "agent_strategy": {k: v.tolist() for k, v in strat.items()},
                "mean_reward": float(a_rew.mean()),
            }
            if sampling_strategy == "thompson":
                metrics.update(opponent_zoo.ts_diagnostics())
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            print(
                f"[{total_games:>8d}] A={current_a:.4f} zoo={len(opponent_zoo):>3d} "
                f"expl={expl:.4f} p1_bet={strat['p1_bet'].round(3)}"
            )

    print(f"\nDone. Metrics saved to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Kuhn Poker zoo training (PPO)")
    parser.add_argument("--a-value", "-A", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--games-per-step", type=int, default=512)
    parser.add_argument("--zoo-update-interval", type=int, default=10)
    parser.add_argument("--zoo-max-size", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="experiments/results/zoo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-strategy", type=str, default="uniform",
                        choices=["uniform", "thompson"])
    parser.add_argument("--competitiveness-threshold", type=float, default=0.3)
    parser.add_argument("--a-schedule", type=str, default="constant",
                        choices=["constant", "exponential", "linear", "sigmoid",
                                 "exponential_down", "linear_down", "sigmoid_down"])
    parser.add_argument("--a-halflife", type=float, default=0.25)
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
    train_zoo(
        a_value=args.a_value,
        timesteps=args.timesteps,
        games_per_step=args.games_per_step,
        zoo_update_interval=args.zoo_update_interval,
        zoo_max_size=args.zoo_max_size,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        cfg=cfg,
        sampling_strategy=args.sampling_strategy,
        competitiveness_threshold=args.competitiveness_threshold,
        a_schedule_type=args.a_schedule,
        a_halflife=args.a_halflife,
    )


if __name__ == "__main__":
    main()
