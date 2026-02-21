"""
Kuhn Poker environment for adversarial self-play experiments.

3-card poker: Jack (0) < Queen (1) < King (2).
Each player antes 1 chip, gets one card. Bets are 1 chip.

Game tree (P1 acts first):
  P1: check or bet
    P1 checks -> P2: check or bet
      P2 checks -> showdown (pot=2, winner gets +1)
      P2 bets -> P1: fold or call
        P1 folds -> P2 wins (P1 loses ante, -1)
        P1 calls -> showdown (pot=4, winner gets +2)
    P1 bets -> P2: fold or call
      P2 folds -> P1 wins (P2 loses ante, P1 gets +1)
      P2 calls -> showdown (pot=4, winner gets +2)

Observation: [card_J, card_Q, card_K, facing_bet, is_player_1] (5 dims)
Actions: 0 = passive (check/fold), 1 = aggressive (bet/call)
"""
import numpy as np
from typing import Callable, Dict, List, Tuple

JACK, QUEEN, KING = 0, 1, 2
NUM_CARDS = 3
OBS_DIM = 5
ACT_DIM = 2

# All possible deals (no replacement): 6 total
DEALS = np.array([(i, j) for i in range(3) for j in range(3) if i != j])


def make_obs(cards: np.ndarray, facing_bet: np.ndarray, is_p1: np.ndarray) -> np.ndarray:
    """Encode observations for a batch of decision points.

    Args:
        cards: (n,) int array of card indices (0=J, 1=Q, 2=K)
        facing_bet: (n,) bool/float array
        is_p1: (n,) bool/float array

    Returns:
        obs: (n, 5) float array
    """
    n = len(cards)
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    obs[np.arange(n), cards] = 1.0
    obs[:, 3] = np.asarray(facing_bet, dtype=np.float32)
    obs[:, 4] = np.asarray(is_p1, dtype=np.float32)
    return obs


def play_games(
    p1_fn: Callable, p2_fn: Callable, num_games: int
) -> Tuple[List, List, np.ndarray, np.ndarray]:
    """Play vectorized Kuhn Poker games.

    Args:
        p1_fn: callable(obs_batch) -> (actions, log_probs)
        p2_fn: callable(obs_batch) -> (actions, log_probs)
        num_games: number of games to play

    Returns:
        p1_records: list of (obs, actions, log_probs, game_indices) tuples
        p2_records: list of (obs, actions, log_probs, game_indices) tuples
        p1_rewards: (num_games,) final P1 rewards
        p2_rewards: (num_games,) final P2 rewards
    """
    # Deal cards
    deal_idx = np.random.randint(0, len(DEALS), size=num_games)
    cards = DEALS[deal_idx]
    p1_cards = cards[:, 0]
    p2_cards = cards[:, 1]

    p1_records = []
    p2_records = []
    p1_rewards = np.zeros(num_games, dtype=np.float32)

    all_indices = np.arange(num_games)

    # --- Step 1: P1 first decision (all games) ---
    p1_obs1 = make_obs(p1_cards, np.zeros(num_games), np.ones(num_games))
    p1_act1, p1_logp1 = p1_fn(p1_obs1)
    p1_records.append((p1_obs1, p1_act1, p1_logp1, all_indices.copy()))

    bet_mask = p1_act1 == 1
    check_mask = p1_act1 == 0

    # --- Step 2: P2 decision (all games) ---
    # P2 faces bet if P1 bet; otherwise P2 acts after P1 check
    p2_obs = make_obs(p2_cards, bet_mask.astype(np.float32), np.zeros(num_games))
    p2_act, p2_logp = p2_fn(p2_obs)
    p2_records.append((p2_obs, p2_act, p2_logp, all_indices.copy()))

    # --- Resolve: P1 bet branch ---
    # P2 folds -> P1 wins ante (+1)
    p2_fold_after_bet = bet_mask & (p2_act == 0)
    p1_rewards[p2_fold_after_bet] = 1.0

    # P2 calls -> showdown (pot=4, winner gets +2)
    p2_call_after_bet = bet_mask & (p2_act == 1)
    if p2_call_after_bet.any():
        wins = p1_cards[p2_call_after_bet] > p2_cards[p2_call_after_bet]
        p1_rewards[p2_call_after_bet] = np.where(wins, 2.0, -2.0)

    # --- Resolve: P1 check branch ---
    # P2 checks -> showdown (pot=2, winner gets +1)
    both_check = check_mask & (p2_act == 0)
    if both_check.any():
        wins = p1_cards[both_check] > p2_cards[both_check]
        p1_rewards[both_check] = np.where(wins, 1.0, -1.0)

    # P2 bets after P1 check -> P1 faces bet (Step 3)
    p2_bet_after_check = check_mask & (p2_act == 1)
    if p2_bet_after_check.any():
        facing_idx = np.where(p2_bet_after_check)[0]
        n_facing = len(facing_idx)

        p1_obs2 = make_obs(
            p1_cards[facing_idx],
            np.ones(n_facing),
            np.ones(n_facing),
        )
        p1_act2, p1_logp2 = p1_fn(p1_obs2)
        p1_records.append((p1_obs2, p1_act2, p1_logp2, facing_idx))

        # P1 folds -> P2 wins (+1 for P2, -1 for P1)
        p1_fold = p1_act2 == 0
        p1_rewards[facing_idx[p1_fold]] = -1.0

        # P1 calls -> showdown (pot=4)
        p1_call = p1_act2 == 1
        call_games = facing_idx[p1_call]
        if len(call_games) > 0:
            wins = p1_cards[call_games] > p2_cards[call_games]
            p1_rewards[call_games] = np.where(wins, 2.0, -2.0)

    p2_rewards = -p1_rewards  # zero-sum
    return p1_records, p2_records, p1_rewards, p2_rewards


def get_strategy(policy_fn: Callable) -> Dict[str, np.ndarray]:
    """Extract strategy probabilities at all 12 info sets.

    Returns dict with:
        p1_bet: (3,) P(bet) for P1 with each card (not facing bet)
        p1_call: (3,) P(call) for P1 with each card (facing bet)
        p2_bet: (3,) P(bet) for P2 with each card (not facing bet, i.e. after P1 check)
        p2_call: (3,) P(call) for P2 with each card (facing bet, i.e. after P1 bet)
    """
    cards = np.array([0, 1, 2])

    # P1, not facing bet
    obs_p1_nf = make_obs(cards, np.zeros(3), np.ones(3))
    probs_p1_nf = policy_fn(obs_p1_nf)  # (3, 2) -> P(aggressive)
    p1_bet = probs_p1_nf[:, 1]

    # P1, facing bet
    obs_p1_f = make_obs(cards, np.ones(3), np.ones(3))
    probs_p1_f = policy_fn(obs_p1_f)
    p1_call = probs_p1_f[:, 1]

    # P2, not facing bet (after P1 check)
    obs_p2_nf = make_obs(cards, np.zeros(3), np.zeros(3))
    probs_p2_nf = policy_fn(obs_p2_nf)
    p2_bet = probs_p2_nf[:, 1]

    # P2, facing bet (after P1 bet)
    obs_p2_f = make_obs(cards, np.ones(3), np.zeros(3))
    probs_p2_f = policy_fn(obs_p2_f)
    p2_call = probs_p2_f[:, 1]

    return {
        "p1_bet": p1_bet,
        "p1_call": p1_call,
        "p2_bet": p2_bet,
        "p2_call": p2_call,
    }


def exploitability(strategy: Dict[str, np.ndarray]) -> float:
    """Compute exploitability of a strategy profile.

    exploitability = BR_value_P1(opponent_P2_strategy) + BR_value_P2(opponent_P1_strategy)

    At Nash equilibrium, exploitability = 0.

    Args:
        strategy: dict from get_strategy() with p1_bet, p1_call, p2_bet, p2_call

    Returns:
        Exploitability (non-negative float). 0 = Nash equilibrium.
    """
    b1 = strategy["p1_bet"]    # P1 bet probs (3,)
    c1 = strategy["p1_call"]   # P1 call probs (3,)
    b2 = strategy["p2_bet"]    # P2 bet probs after P1 check (3,)
    c2 = strategy["p2_call"]   # P2 call probs after P1 bet (3,)

    br_p1 = _best_response_p1_value(b2, c2)
    br_p2 = _best_response_p2_value(b1, c1)
    return br_p1 + br_p2


def _best_response_p1_value(b2: np.ndarray, c2: np.ndarray) -> float:
    """Compute P1's best response value against P2's strategy.

    P2's strategy: b2[c2] = P(bet after P1 check), c2[c2] = P(call after P1 bet)
    """
    total = 0.0
    for c1 in range(3):
        # First compute BR at facing_bet info set (c1, facing)
        # Reached when P1 checked and P2 bet
        call_val = 0.0
        fold_val = 0.0
        for c2_card in range(3):
            if c2_card == c1:
                continue
            w = (1.0 / 6.0) * b2[c2_card]  # reach weight
            sgn = 1.0 if c1 > c2_card else -1.0
            call_val += w * sgn * 2.0
            fold_val += w * (-1.0)

        # BR at facing_bet: call or fold
        br_facing_val = max(call_val, fold_val)
        br_facing_call = 1.0 if call_val >= fold_val else 0.0

        # Now compute first action: bet vs check
        bet_val = 0.0
        check_val = 0.0
        for c2_card in range(3):
            if c2_card == c1:
                continue
            w = 1.0 / 6.0
            sgn = 1.0 if c1 > c2_card else -1.0

            # Bet: P2 calls with c2[c2_card], folds with 1-c2[c2_card]
            bet_val += w * (c2[c2_card] * sgn * 2.0 + (1.0 - c2[c2_card]) * 1.0)

            # Check: P2 checks with 1-b2[c2_card], bets with b2[c2_card]
            check_payoff_p2_checks = sgn * 1.0
            check_payoff_p2_bets = br_facing_call * sgn * 2.0 + (1.0 - br_facing_call) * (-1.0)
            check_val += w * (
                (1.0 - b2[c2_card]) * check_payoff_p2_checks
                + b2[c2_card] * check_payoff_p2_bets
            )

        total += max(bet_val, check_val)

    return total


def _best_response_p2_value(b1: np.ndarray, c1: np.ndarray) -> float:
    """Compute P2's best response value against P1's strategy.

    P1's strategy: b1[c1] = P(bet first action), c1[c1] = P(call when facing bet)
    """
    total = 0.0
    for c2 in range(3):
        # Info set (c2, facing_bet=True): after P1 bet
        call_val = 0.0
        fold_val = 0.0
        for c1_card in range(3):
            if c1_card == c2:
                continue
            w = (1.0 / 6.0) * b1[c1_card]
            sgn = 1.0 if c2 > c1_card else -1.0
            call_val += w * sgn * 2.0
            fold_val += w * (-1.0)
        facing_contribution = max(call_val, fold_val)

        # Info set (c2, not_facing): after P1 check
        check_val = 0.0
        bet_val = 0.0
        for c1_card in range(3):
            if c1_card == c2:
                continue
            w = (1.0 / 6.0) * (1.0 - b1[c1_card])
            sgn = 1.0 if c2 > c1_card else -1.0
            check_val += w * sgn * 1.0
            # If P2 bets: P1 calls with c1[c1_card], folds with 1-c1[c1_card]
            bet_val += w * (c1[c1_card] * sgn * 2.0 + (1.0 - c1[c1_card]) * 1.0)
        not_facing_contribution = max(check_val, bet_val)

        total += facing_contribution + not_facing_contribution

    return total


def action_entropy(strategy: Dict[str, np.ndarray]) -> float:
    """Mean entropy across all 12 info sets."""
    all_probs = []
    for key in ["p1_bet", "p1_call", "p2_bet", "p2_call"]:
        for p in strategy[key]:
            all_probs.append([1 - p, p])
    total_entropy = 0.0
    for probs in all_probs:
        for p in probs:
            if p > 1e-10:
                total_entropy -= p * np.log(p)
    return total_entropy / len(all_probs)


# --- Nash equilibrium reference ---
def nash_strategy(alpha: float = 1.0 / 3.0) -> Dict[str, np.ndarray]:
    """Return the Nash equilibrium strategy parameterized by alpha in [0, 1/3].

    P1: J bet alpha, Q check (call alpha+1/3 facing bet), K bet 3*alpha (call always)
    P2: J bet 1/3 after check (fold to bet), Q check (call 1/3), K always bet/call
    """
    return {
        "p1_bet": np.array([alpha, 0.0, 3.0 * alpha]),
        "p1_call": np.array([0.0, alpha + 1.0 / 3.0, 1.0]),
        "p2_bet": np.array([1.0 / 3.0, 0.0, 1.0]),
        "p2_call": np.array([0.0, 1.0 / 3.0, 1.0]),
    }


def game_value() -> float:
    """Expected value for P1 at Nash equilibrium: -1/18."""
    return -1.0 / 18.0


def test_exploitability():
    """Verify exploitability computation against known Nash equilibria."""
    # Test at alpha=1/3
    nash = nash_strategy(1.0 / 3.0)
    expl = exploitability(nash)
    assert abs(expl) < 1e-10, f"Nash (alpha=1/3) exploitability should be 0, got {expl}"

    # Test at alpha=0
    nash0 = nash_strategy(0.0)
    expl0 = exploitability(nash0)
    assert abs(expl0) < 1e-10, f"Nash (alpha=0) exploitability should be 0, got {expl0}"

    # Test that a non-Nash strategy has positive exploitability
    bad = {
        "p1_bet": np.array([0.5, 0.5, 0.5]),
        "p1_call": np.array([0.5, 0.5, 0.5]),
        "p2_bet": np.array([0.5, 0.5, 0.5]),
        "p2_call": np.array([0.5, 0.5, 0.5]),
    }
    expl_bad = exploitability(bad)
    assert expl_bad > 0, f"Random strategy should have positive exploitability, got {expl_bad}"

    # Test pure strategy
    pure = {
        "p1_bet": np.array([1.0, 1.0, 1.0]),
        "p1_call": np.array([1.0, 1.0, 1.0]),
        "p2_bet": np.array([1.0, 1.0, 1.0]),
        "p2_call": np.array([1.0, 1.0, 1.0]),
    }
    expl_pure = exploitability(pure)
    assert expl_pure > 0, f"Pure strategy should have positive exploitability, got {expl_pure}"

    print(f"Nash (alpha=1/3) exploitability: {exploitability(nash):.10f}")
    print(f"Nash (alpha=0) exploitability: {exploitability(nash0):.10f}")
    print(f"Random (all 0.5) exploitability: {expl_bad:.4f}")
    print(f"Pure aggressive exploitability: {expl_pure:.4f}")


if __name__ == "__main__":
    test_exploitability()
