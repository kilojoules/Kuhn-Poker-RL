"""
Replay-buffer agent for Kuhn Poker with importance correction.

Supports both tabular and neural policies. Trains off-policy from a
FIFO replay buffer with truncated importance weights.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ppo import TabularPolicy, SoftmaxPolicy


@dataclass
class BufferedConfig:
    obs_dim: int = 5
    act_dim: int = 2
    hidden: int = 32
    lr: float = 0.1
    buffer_size: int = 10000
    batch_size: int = 256
    train_iters: int = 4
    entropy_coef: float = 0.005
    is_ratio_clip: float = 2.0
    policy_type: str = "tabular"


class ReplayBuffer:
    """FIFO replay buffer storing (obs, action, log_prob_behavior, reward)."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []

    def add(self, obs: np.ndarray, actions: np.ndarray,
            log_probs: np.ndarray, rewards: np.ndarray):
        for i in range(len(obs)):
            self.obs.append(obs[i])
            self.actions.append(int(actions[i]))
            self.log_probs.append(float(log_probs[i]))
            self.rewards.append(float(rewards[i]))

        while len(self.obs) > self.max_size:
            self.obs.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.rewards.pop(0)

    def sample(self, batch_size: int):
        n = len(self.obs)
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        return (
            np.array([self.obs[i] for i in idx]),
            np.array([self.actions[i] for i in idx]),
            np.array([self.log_probs[i] for i in idx]),
            np.array([self.rewards[i] for i in idx]),
        )

    def __len__(self):
        return len(self.obs)


class BufferedAgent:
    """Off-policy agent with replay buffer and importance correction."""

    def __init__(self, cfg: BufferedConfig):
        self.cfg = cfg
        if cfg.policy_type == "tabular":
            self.policy = TabularPolicy(cfg.entropy_coef)
        else:
            self.policy = SoftmaxPolicy(
                cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.entropy_coef
            )
        self.buffer = ReplayBuffer(max_size=cfg.buffer_size)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.policy.sample(obs)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.action_probs(obs)

    def store(self, obs: np.ndarray, actions: np.ndarray,
              log_probs: np.ndarray, rewards: np.ndarray):
        self.buffer.add(obs, actions, log_probs, rewards)

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return

        cfg = self.cfg
        for _ in range(cfg.train_iters):
            obs, actions, old_log_probs, rewards = self.buffer.sample(cfg.batch_size)

            advantages = rewards.copy().astype(np.float64)
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            probs, cache = self.policy.forward(obs)
            new_log_probs = np.log(
                probs[np.arange(len(actions)), actions] + 1e-10
            )

            ratio = np.exp(new_log_probs - old_log_probs)
            clipped_ratio = np.clip(ratio, 0.0, cfg.is_ratio_clip)
            weights = clipped_ratio * advantages

            if isinstance(self.policy, TabularPolicy):
                grad = self.policy.backward(cache, actions, weights)
                self.policy.logits += cfg.lr * grad
            else:
                grads = self.policy.backward(cache, actions, weights)
                self.policy.W1 += cfg.lr * grads["W1"]
                self.policy.b1 += cfg.lr * grads["b1"]
                self.policy.W2 += cfg.lr * grads["W2"]
                self.policy.b2 += cfg.lr * grads["b2"]

    def get_state(self) -> List[np.ndarray]:
        return [p.copy() for p in self.policy.get_params()]

    def load_state(self, state: List[np.ndarray]):
        self.policy.set_params(state)
