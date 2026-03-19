# Prioritized Experience Replay

| Field | Details |
|-------|---------|
| **Authors** | Tom Schaul, John Quan, Ioannis Antonoglou, David Silver |
| **Year** | 2016 |
| **Venue** | ICLR 2016 |
| **Institution** | Google DeepMind |

---

## Key Contributions

- Proposed **prioritized experience replay**, which replays transitions with high temporal-difference (TD) error more frequently, replacing uniform random sampling
- Introduced two prioritization strategies: **proportional** and **rank-based** prioritization
- Incorporated **importance sampling corrections** to counteract the bias introduced by non-uniform sampling
- Achieved significant improvements over uniform replay on 41 out of 49 Atari games
- Provided an efficient implementation using a sum-tree data structure for $O(\log N)$ sampling and updating

---

## Background & Motivation

Experience replay (Lin, 1992) stores agent transitions $(s_t, a_t, r_t, s_{t+1})$ in a buffer and samples mini-batches uniformly at random for training. This breaks temporal correlations and improves data efficiency. However, **uniform sampling treats all transitions equally**, regardless of their learning potential.

Key insight: Some transitions are more "surprising" or "informative" than others. A transition where the agent's prediction is very wrong (high TD error) likely carries more useful learning signal than one the agent already predicts well. By replaying these high-error transitions more often, learning can be made significantly faster.

This idea is inspired by **prioritized sweeping** in model-based RL, where states are updated in order of their expected value change.

---

## Method

### TD Error as Priority

The most natural measure of a transition's importance is the magnitude of its TD error:

$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) - Q(s_t, a_t; \theta)$$

Transitions with large $|\delta_t|$ are far from the current Q-estimate and thus carry the most learning signal.

### Proportional Prioritization

The probability of sampling transition $i$ is proportional to its priority:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

where $p_i = |\delta_i| + \epsilon$ is the priority of transition $i$ ($\epsilon > 0$ is a small constant to ensure every transition has a nonzero probability of being sampled).

The exponent $\alpha$ controls how much prioritization is used:
- $\alpha = 0$: uniform sampling (standard experience replay)
- $\alpha = 1$: full prioritization

### Rank-Based Prioritization

An alternative scheme based on the rank of the transition when sorted by $|\delta_i|$:

$$p_i = \frac{1}{\text{rank}(i)}$$

where $\text{rank}(i)$ is the rank of transition $i$ when all transitions are sorted by $|\delta_i|$ in descending order. This leads to a probability distribution:

$$P(i) = \frac{(1 / \text{rank}(i))^\alpha}{\sum_k (1 / \text{rank}(k))^\alpha}$$

Rank-based prioritization is more robust to outliers since it depends only on the ordering, not the magnitude, of TD errors.

### Importance Sampling Correction

Non-uniform sampling introduces **bias** into the gradient estimates. If left uncorrected, this changes what the algorithm converges to. Importance sampling weights correct for this:

$$w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta$$

where $N$ is the replay buffer size and $\beta$ controls the degree of correction:
- $\beta = 0$: no correction (biased)
- $\beta = 1$: full correction (unbiased)

In practice, $\beta$ is annealed from an initial value $\beta_0 < 1$ to $1$ over the course of training. This reflects the fact that bias matters most near convergence.

The weights are normalized by the maximum weight in the mini-batch:

$$\hat{w}_i = \frac{w_i}{\max_j w_j}$$

The TD update becomes:

$$\Delta\theta = \hat{w}_i \cdot \delta_i \cdot \nabla_\theta Q(s_i, a_i; \theta)$$

### Efficient Implementation

Naive prioritized sampling would be $O(N)$ per sample. The paper uses a **sum-tree** data structure:

| Operation | Uniform Replay | Prioritized (Sum-Tree) |
|-----------|---------------|----------------------|
| Insert | $O(1)$ | $O(\log N)$ |
| Sample | $O(1)$ | $O(\log N)$ |
| Update priority | N/A | $O(\log N)$ |

A sum-tree is a binary tree where each leaf stores a priority and each internal node stores the sum of its children. Sampling proportional to priority reduces to traversing the tree — a single $O(\log N)$ operation.

### Stale Priorities

When a transition's priority is set based on its TD error, the error can become stale as the Q-network updates. The paper addresses this by:
1. Setting the priority of a newly added transition to the maximum priority in the buffer (ensuring it is sampled at least once)
2. Updating the priority only when a transition is replayed (lazy updates)

### Algorithm Summary

```
Initialize replay buffer R with sum-tree, priority exponent α, IS exponent β₀
for each training step:
    Sample mini-batch of transitions i ~ P(i) using sum-tree
    Compute TD errors δ_i for each sampled transition
    Compute IS weights: w_i = (N · P(i))^(-β) / max_j(w_j)
    Update Q-network with weighted TD errors: Δθ = Σ w_i · δ_i · ∇Q
    Update priorities: p_i = |δ_i| + ε
    Anneal β toward 1
```

---

## Key Results

### Atari 2600 Benchmark

| Algorithm | Mean Score (normalized) | Median Score (normalized) | Games Improved |
|-----------|------------------------|--------------------------|----------------|
| DQN (uniform replay) | 100% (baseline) | 100% (baseline) | -- |
| DQN + Proportional PER | ~138% | ~124% | 41 / 49 |
| DQN + Rank-based PER | ~142% | ~128% | 41 / 49 |
| Double DQN + Rank-based PER | **~160%** | **~140%** | -- |

### Learning Speed

Prioritized replay achieved equivalent performance to uniform replay in approximately **2x fewer training steps** in most games, demonstrating significantly improved sample efficiency.

### Sensitivity to Hyperparameters

| Parameter | Best Range | Effect |
|-----------|-----------|--------|
| $\alpha$ (priority exponent) | 0.5 -- 0.7 | Higher = more aggressive prioritization |
| $\beta_0$ (initial IS exponent) | 0.4 -- 0.5 | Annealed to 1.0 by end of training |
| $\epsilon$ (priority floor) | $10^{-6}$ | Prevents zero-probability transitions |

Both proportional and rank-based variants showed similar overall performance, with rank-based being slightly more robust to outlier TD errors.

---

## Impact & Legacy

- **Core component of modern RL**: Prioritized experience replay is now a standard ingredient in DQN-based systems and was incorporated into Rainbow DQN (2017) as one of six key improvements
- **Influenced sample-efficient RL**: The idea of non-uniform data sampling based on learning signal inspired many subsequent works in curriculum learning and active learning for RL
- **Hindsight Experience Replay (HER)**: Built on similar ideas of selectively replaying informative transitions, applied to goal-conditioned RL
- **Distributed RL systems**: Ape-X and R2D2 used prioritized replay at scale with distributed actors and a centralized prioritized buffer
- **Theory development**: Motivated further study of bias-variance tradeoffs in off-policy learning and the convergence properties of prioritized methods

---

## Key Takeaways

1. **Not all experiences are equally valuable**: Transitions with high TD error contain more learning signal, and replaying them more often accelerates learning
2. **Importance sampling is essential**: Without IS correction, prioritized sampling introduces bias that can degrade final performance; annealing $\beta$ from a low value to 1 balances learning speed and correctness
3. **Rank-based prioritization is robust**: Using ranks instead of raw TD errors avoids sensitivity to outliers and scale
4. **Efficient data structures matter**: The sum-tree enables practical prioritized sampling with only logarithmic overhead
5. **Prioritization is orthogonal to other improvements**: It composes well with Double DQN, dueling networks, and other DQN enhancements for cumulative gains
