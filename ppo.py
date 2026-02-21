"""
PPO agent for Kuhn Poker with tabular + neural policy options.

Tabular policy: direct logit parameters per info set (12 total).
Neural policy: two-layer softmax network with analytic gradients.
On-policy / memoryless -- the key property for the A-parameter hypothesis.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PPOConfig:
    obs_dim: int = 5   # [card_J, card_Q, card_K, facing_bet, is_p1]
    act_dim: int = 2   # [passive, aggressive]
    hidden: int = 32
    lr: float = 0.1
    clip_ratio: float = 0.2
    train_iters: int = 4
    entropy_coef: float = 0.005
    policy_type: str = "tabular"  # "tabular" or "neural"


def _obs_to_infoset(obs: np.ndarray) -> np.ndarray:
    """Map observation batch to info set indices (0-11).

    Info sets:
      0-2: P1, not facing bet, card J/Q/K -> P(bet)
      3-5: P1, facing bet, card J/Q/K -> P(call)
      6-8: P2, not facing bet, card J/Q/K -> P(bet)
      9-11: P2, facing bet, card J/Q/K -> P(call)
    """
    card = np.argmax(obs[:, :3], axis=1)  # 0, 1, or 2
    facing = obs[:, 3].astype(int)         # 0 or 1
    is_p1 = obs[:, 4].astype(int)          # 0 or 1
    # P1: base 0, P2: base 6; facing_bet adds 3; card adds 0-2
    return (1 - is_p1) * 6 + facing * 3 + card


class TabularPolicy:
    """Tabular policy: 12 logit parameters, one per info set.

    Maps each info set to P(aggressive action) via sigmoid.
    """

    def __init__(self, entropy_coef: float = 0.005):
        self.num_infosets = 12
        self.entropy_coef = entropy_coef
        # Initialize logits near 0 (= 50% probability)
        self.logits = np.zeros(self.num_infosets, dtype=np.float64)

    def forward(self, obs: np.ndarray):
        """Return action probs and cache for backprop."""
        infosets = _obs_to_infoset(obs)
        logit_vals = self.logits[infosets]  # (batch,)

        # Softmax over 2 actions: [passive, aggressive]
        # logits = [0, logit_val] -> softmax
        batch = len(obs)
        full_logits = np.zeros((batch, 2), dtype=np.float64)
        full_logits[:, 1] = logit_vals

        logits_shifted = full_logits - full_logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits_shifted)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

        cache = {"infosets": infosets, "probs": probs}
        return probs, cache

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(obs)
        return probs

    def sample(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs, _ = self.forward(obs)
        actions = np.array([
            np.random.choice(2, p=np.clip(p, 1e-10, None) / np.clip(p, 1e-10, None).sum())
            for p in probs
        ])
        log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-10)
        return actions, log_probs

    def backward(self, cache: dict, actions: np.ndarray, weights: np.ndarray):
        """Compute gradients for the 12 logit parameters."""
        batch = len(actions)
        infosets = cache["infosets"]
        probs = cache["probs"]

        # d(log pi(a|s))/d(logit) for softmax over [0, logit]
        # For action 1 (aggressive): d/d(logit) = 1 - p_aggressive = p_passive
        # For action 0 (passive): d/d(logit) = -p_aggressive
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch), actions] = 1.0
        # Gradient of log pi w.r.t. logit for action 1:
        # d(log pi(a|s)) / d(logit_1) = (indicator(a=1) - probs[:, 1])
        d_logit = (one_hot[:, 1] - probs[:, 1]) * weights / batch

        # Entropy bonus: H = -p0*log(p0) - p1*log(p1)
        # dH/d(logit_1) = p0*p1*(log(p0) - log(p1)) = p0*p1*log(p0/p1)
        log_probs = np.log(probs + 1e-10)
        entropy_grad = probs[:, 0] * probs[:, 1] * (log_probs[:, 0] - log_probs[:, 1])
        d_logit += self.entropy_coef * entropy_grad / batch

        # Accumulate gradients per info set
        grad = np.zeros(self.num_infosets, dtype=np.float64)
        np.add.at(grad, infosets, d_logit)

        return grad

    def get_params(self) -> List[np.ndarray]:
        return [self.logits.copy()]

    def set_params(self, params: List[np.ndarray]):
        self.logits = params[0].copy()


class SoftmaxPolicy:
    """Two-layer ReLU network with softmax output and analytic gradients."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 32,
                 entropy_coef: float = 0.005):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.entropy_coef = entropy_coef

        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = np.random.randn(hidden, obs_dim).astype(np.float64) * scale1
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = np.random.randn(act_dim, hidden).astype(np.float64) * scale2
        self.b2 = np.zeros(act_dim, dtype=np.float64)

    def forward(self, obs: np.ndarray):
        obs = obs.astype(np.float64)
        pre_h = obs @ self.W1.T + self.b1
        h = np.maximum(0, pre_h)
        logits = h @ self.W2.T + self.b2

        logits_shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        cache = {"obs": obs, "pre_h": pre_h, "h": h, "probs": probs}
        return probs, cache

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(obs)
        return probs

    def sample(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs, _ = self.forward(obs)
        actions = np.array([
            np.random.choice(len(p), p=np.clip(p, 1e-10, None) / np.clip(p, 1e-10, None).sum())
            for p in probs
        ])
        log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-10)
        return actions, log_probs

    def backward(self, cache: dict, actions: np.ndarray, weights: np.ndarray):
        batch = len(actions)
        obs = cache["obs"]
        pre_h = cache["pre_h"]
        h = cache["h"]
        probs = cache["probs"]

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch), actions] = 1.0
        d_logits = (one_hot - probs) * weights[:, None] / batch

        log_probs = np.log(probs + 1e-10)
        entropy_grad = (
            -(log_probs + 1) * probs
            + probs * np.sum((log_probs + 1) * probs, axis=-1, keepdims=True)
        )
        d_logits += self.entropy_coef * entropy_grad / batch

        dW2 = d_logits.T @ h
        db2 = d_logits.sum(axis=0)
        d_h = d_logits @ self.W2
        d_pre_h = d_h * (pre_h > 0).astype(np.float64)
        dW1 = d_pre_h.T @ obs
        db1 = d_pre_h.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: List[np.ndarray]):
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]


class PPOAgent:
    """PPO with clipped surrogate and analytic gradients."""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        if cfg.policy_type == "tabular":
            self.policy = TabularPolicy(cfg.entropy_coef)
        else:
            self.policy = SoftmaxPolicy(
                cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.entropy_coef
            )

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.policy.sample(obs)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.action_probs(obs)

    def update(self, obs: np.ndarray, actions: np.ndarray,
               rewards: np.ndarray, old_log_probs: np.ndarray):
        cfg = self.cfg

        advantages = rewards.copy().astype(np.float64)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(cfg.train_iters):
            probs, cache = self.policy.forward(obs)
            new_log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-10)

            ratio = np.exp(new_log_probs - old_log_probs)
            clipped = (ratio < 1 - cfg.clip_ratio) | (ratio > 1 + cfg.clip_ratio)
            weights = np.where(clipped, 0.0, advantages)

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
