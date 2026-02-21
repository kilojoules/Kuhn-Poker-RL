# Kuhn-Poker-RL

A-parameter hypothesis testbed using Kuhn Poker — a sequel to [RPS_RL](https://github.com/kilojoules/RPS_RL) that tests whether the same findings hold in a game with information asymmetry, sequential decisions, and non-trivial Nash equilibrium structure.

**Key finding: every result from RPS reverses.** Zoo sampling hurts convergence, the replay buffer provides no advantage, and Thompson Sampling makes things worse. The reason: in Kuhn Poker, the best response to diverse/random opponents is an exploitative strategy, not Nash.

## The A Parameter

A = probability of sampling an opponent from the historical zoo (vs. playing the latest opponent):
- **A=0**: Self-play — always play latest opponent, no zoo.
- **A in (0, 1)**: Mix of latest opponent + zoo sampling.
- **A near 1**: Almost always sample from zoo.

## The Game

Kuhn Poker is a simplified poker game with a known Nash equilibrium:

- **3-card deck**: Jack (J) < Queen (Q) < King (K)
- **Each player** antes 1 chip, receives one card
- **Player 1** acts first: check or bet (1 chip)
  - If P1 checks → P2: check (showdown) or bet
    - If P2 bets → P1: fold or call
  - If P1 bets → P2: fold or call
- **Showdown**: higher card wins the pot

**Nash equilibrium** (parameterized by α ∈ [0, 1/3]):

| Info Set | Nash (α=1/3) | Description |
|----------|-------------|-------------|
| P1 bet J | 0.33 | Bluff 1/3 of the time |
| P1 bet Q | 0.00 | Never bet Queen |
| P1 bet K | 1.00 | Always value-bet King |
| P1 call J | 0.00 | Always fold Jack to bet |
| P1 call Q | 0.67 | Call 2/3 with Queen |
| P1 call K | 1.00 | Always call with King |
| P2 bet J | 0.33 | Bluff 1/3 after check |
| P2 bet Q | 0.00 | Never bet Queen |
| P2 bet K | 1.00 | Always bet King |
| P2 call J | 0.00 | Always fold Jack |
| P2 call Q | 0.33 | Call 1/3 with Queen |
| P2 call K | 1.00 | Always call with King |

Game value: −1/18 for Player 1. Exploitability at Nash = 0.

**Why Kuhn Poker instead of RPS?** RPS has a degenerate property: Nash equilibrium (uniform 1/3, 1/3, 1/3) is the best response to *any* mixture of opponents. Training against random opponents naturally converges to Nash. Kuhn Poker breaks this — the best response to random opponents involves aggressive betting that exploits their mistakes, which is far from Nash. This makes the zoo's effect on convergence qualitatively different.

## Results

We ran 145 experiments: 7 A values × 2 algorithms (PPO, Buffered) × 2 sampling strategies (Uniform, Thompson) × 5 seeds, plus 5 self-play baselines. All at 500k games with tabular policies (12 logit parameters per agent).

### The A Curve Inverts

In RPS, more zoo = lower exploitability. In Kuhn Poker, **more zoo = higher exploitability**, monotonically:

![A curve](experiments/results/a_curve.png)

**PPO (uniform sampling) — 500k games, 5 seeds:**

| Condition | Exploitability (mean ± std) |
|-----------|----------------------------|
| Self-play (A=0) | **0.2268 ± 0.0034** |
| A=0.05 | 0.2342 ± 0.0049 |
| A=0.10 | 0.2335 ± 0.0082 |
| A=0.20 | 0.2561 ± 0.0048 |
| A=0.30 | 0.2736 ± 0.0074 |
| A=0.50 | 0.3087 ± 0.0052 |
| A=0.70 | 0.4682 ± 0.0178 |
| A=0.90 | 0.5896 ± 0.0061 |

**Buffered (uniform sampling) — 500k games, 5 seeds:**

| Condition | Exploitability (mean ± std) |
|-----------|----------------------------|
| A=0.05 | 0.2360 ± 0.0044 |
| A=0.10 | 0.2426 ± 0.0083 |
| A=0.20 | 0.2625 ± 0.0056 |
| A=0.30 | 0.2763 ± 0.0043 |
| A=0.50 | 0.3067 ± 0.0111 |
| A=0.70 | 0.4671 ± 0.0067 |
| A=0.90 | 0.5908 ± 0.0056 |

Self-play (A=0) produces the lowest exploitability at **0.2268** — better than any zoo condition. The A curve rises monotonically: A=0.9 is 2.6× worse than self-play.

### PPO and Buffered Are Identical

Unlike RPS where PPO's curve was steeper than Buffered's, the two algorithms produce **indistinguishable results** in Kuhn Poker. The curves overlap within error bars at every A value.

**Why?** In RPS, the replay buffer's "memory" of past opponents provided diversity that PPO lacked. In Kuhn Poker, that same memory is counterproductive — it reinforces exploitative strategies learned against weak historical opponents. The buffer remembers the *wrong things*.

### Thompson Sampling Hurts

In RPS, Thompson Sampling improved exploitability by 30–69% at high A. In Kuhn Poker, Thompson **increases** exploitability by up to 18.5%:

![Thompson comparison](experiments/results/thompson_comparison.png)

**PPO — Uniform vs Thompson (500k games, 5 seeds):**

| A | Uniform | Thompson | Change |
|---|---------|----------|--------|
| 0.05 | 0.2342 | 0.2330 | −0.5% |
| 0.10 | 0.2335 | 0.2420 | +3.6% |
| 0.20 | 0.2561 | 0.2678 | +4.6% |
| 0.30 | 0.2736 | 0.2838 | +3.7% |
| 0.50 | 0.3087 | 0.3658 | **+18.5%** |
| 0.70 | 0.4682 | 0.5242 | **+12.0%** |
| 0.90 | 0.5896 | 0.5990 | +1.6% |

**Why?** Thompson selects "competitive" opponents — those producing close matches. In Kuhn Poker, an agent that has learned exploitative strategies will have close matches against opponents near Nash (since both do reasonably well). Thompson then over-samples these near-Nash opponents, while under-sampling the weak opponents that the agent has learned to crush. This concentrates training on opponents that don't expose the agent's exploitative weaknesses.

### What the Agent Actually Learns

The strategy heatmap reveals exactly how zoo sampling corrupts learning:

![Strategy heatmap](experiments/results/strategy_heatmap.png)

The critical failure mode is **P1/P2 bet Q** (Queen betting probability):

| Condition | P1 bet Q | P2 bet Q | Nash |
|-----------|----------|----------|------|
| A=0.05 | 0.38 | 0.60 | **0.00** |
| A=0.50 | 0.67 | 0.69 | **0.00** |
| A=0.90 | 0.92 | 0.82 | **0.00** |

At A=0.9, the agent bets with Queen 92% of the time — it should *never* bet with Queen. Betting with Queen is optimal against weak opponents who fold too much, but catastrophic against competent opponents who call with King.

The zoo is filled with historical checkpoints from early training, when opponents played randomly. Against random opponents, aggressive betting with all cards is profitable. So the agent learns "always bet" — which works against the zoo but is maximally exploitable.

### Training Dynamics

![Timeseries](experiments/results/timeseries.png)

## Why Everything Reverses

The reversal traces to a single structural difference between RPS and Kuhn Poker:

| Property | RPS | Kuhn Poker |
|----------|-----|------------|
| Nash is best response to random? | **Yes** — uniform play is optimal against any opponent mix | **No** — best response to random is exploitative betting |
| Zoo diversity helps? | Yes — diverse opponents = diverse gradients → Nash | No — diverse weak opponents → exploitative strategies |
| Replay buffer helps? | Yes — "remembers" diverse past opponents | No — "remembers" rewards from exploiting weak opponents |
| Thompson helps? | Yes — selects informative opponents | No — selects opponents that don't expose weaknesses |

In game theory terms: RPS is a **symmetric** zero-sum game where the minimax strategy coincides with the maximin strategy against any opponent distribution. Kuhn Poker is an **asymmetric information** game where the optimal strategy depends critically on the opponent's skill level. Training against a distribution of weak opponents teaches exploitation, not equilibrium play.

## Implications for the A-Parameter Hypothesis

**The A-parameter hypothesis — that memoryless algorithms need more zoo sampling — does not hold in Kuhn Poker.** Specifically:

1. **A\* = 0 (self-play) for both algorithms.** The optimal zoo sampling ratio is zero. Any amount of zoo sampling hurts.
2. **The replay buffer provides no advantage.** PPO and Buffered produce identical results at every A value, contradicting the hypothesis that "memory capacity" determines sensitivity to zoo sampling.
3. **The benefit of zoo sampling is game-dependent.** It helps in games where Nash is robust to opponent distribution (like RPS) and hurts in games where Nash requires precise adaptation to the opponent (like poker).
4. **Thompson Sampling's value depends on what "competitive" means.** In RPS, competitive matches signal informative opponents. In Kuhn Poker, competitive matches signal opponents the agent already handles well — the opposite of what you want for learning.

**What transfers from RPS to Kuhn Poker:** The qualitative insight that zoo diversity can be destabilizing (observed at high A in RPS's long-horizon experiments) is the *dominant* effect in Kuhn Poker, present at all timescales and all A values.

**What doesn't transfer:** The specific finding that moderate zoo sampling accelerates convergence, that PPO benefits more than buffered agents, and that Thompson Sampling helps at high A.

## Quick Start

```bash
# Install with pixi
pixi install

# Run tests
pixi run test-env

# Self-play baseline
pixi run selfplay

# Single zoo experiment
python train_zoo.py -A 0.1 --timesteps 500000

# Single buffered experiment
python train_zoo_buffered.py -A 0.5 --timesteps 500000 --sampling-strategy thompson

# Full sweep (145 experiments, ~50 min)
pixi run sweep

# Analyze results
pixi run analyze
```

## Architecture

- **Policy**: Tabular — 12 logit parameters (one per info set), mapped to action probabilities via softmax. No neural network approximation error.
- **PPO**: On-policy, clipped surrogate, analytic gradients. Memoryless — only trains on current batch.
- **Buffered**: Off-policy with FIFO replay buffer (10k transitions) and truncated importance weights. Provides "memory" of past experience.
- **Zoo**: Stores up to 50 opponent checkpoints. Supports uniform and Thompson sampling.
- **Exploitability**: Exact best-response computation over the full game tree. Verified against known Nash equilibria (exploitability = 0 at α=0 and α=1/3).

## Project Structure

```
Kuhn-Poker-RL/
├── kuhn_env.py            # Game environment + exact exploitability
├── ppo.py                 # Tabular/neural policy + PPO agent
├── buffered_agent.py      # Buffered agent with importance correction
├── zoo.py                 # Opponent zoo + Thompson sampling + schedules
├── train_selfplay.py      # Self-play baseline (A=0)
├── train_zoo.py           # Zoo training for PPO
├── train_zoo_buffered.py  # Zoo training for Buffered
├── run_sweep.py           # Experiment sweep orchestration
├── analyze.py             # Analysis and plotting
├── pyproject.toml         # Pixi project configuration
└── experiments/results/   # Sweep results and plots
```
