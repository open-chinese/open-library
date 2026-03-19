# Continuous Control with Deep Reinforcement Learning (DDPG)

| Field | Details |
|-------|---------|
| **Authors** | Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra |
| **Year** | 2016 (published at ICLR 2016; arXiv 2015) |
| **Venue** | ICLR 2016 |
| **Institution** | Google DeepMind |

---

## Key Contributions

- Introduced **Deep Deterministic Policy Gradient (DDPG)**, an actor-critic algorithm that learns continuous control policies end-to-end
- Extended DQN-style techniques (experience replay, target networks) to the continuous action domain where DQN cannot operate
- Demonstrated that a single algorithm with fixed hyperparameters could solve over 20 continuous control tasks in simulation
- Showed that policies could be learned directly from raw pixel observations in some tasks
- Provided a practical, model-free approach to continuous control that combined the stability of DQN with the expressiveness of policy gradient methods

---

## Background & Motivation

DQN achieved impressive results in discrete-action domains (e.g., Atari), but it fundamentally requires finding $\arg\max_a Q(s, a)$ at each step. In continuous action spaces, this maximization becomes intractable since the action space is uncountably infinite.

Prior approaches to continuous RL had significant limitations:

- **Discretizing actions**: Leads to combinatorial explosion (e.g., a 7-DOF robot arm with 3 bins per joint = $3^7 = 2187$ actions)
- **NAF (Normalized Advantage Functions)**: Restricted Q-functions to be quadratic in actions
- **Stochastic policy gradients**: High variance, requiring many samples

DDPG addresses these by learning a **deterministic policy** $\mu(s)$ that directly outputs the continuous action, while using an off-policy actor-critic framework with the stabilizing techniques from DQN.

---

## Method / Architecture

### Overview

DDPG maintains four networks:

| Network | Symbol | Role |
|---------|--------|------|
| Actor (policy) | $\mu(s \mid \theta^\mu)$ | Maps states to actions |
| Critic (Q-function) | $Q(s, a \mid \theta^Q)$ | Evaluates state-action pairs |
| Target actor | $\mu'(s \mid \theta^{\mu'})$ | Stabilizes training |
| Target critic | $Q'(s, a \mid \theta^{Q'})$ | Stabilizes training |

### Critic Update

The critic is trained to minimize the Bellman error using transitions $(s_t, a_t, r_t, s_{t+1})$ sampled from a replay buffer $\mathcal{R}$:

$$L(\theta^Q) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathcal{R}} \left[ \left( Q(s_t, a_t \mid \theta^Q) - y_t \right)^2 \right]$$

where the target value $y_t$ uses the **target networks**:

$$y_t = r_t + \gamma \, Q'(s_{t+1}, \mu'(s_{t+1} \mid \theta^{\mu'}) \mid \theta^{Q'})$$

### Actor Update

The actor is updated using the **deterministic policy gradient theorem** (Silver et al., 2014). The objective is to maximize the expected Q-value:

$$J(\theta^\mu) = \mathbb{E}_{s \sim \mathcal{R}} \left[ Q(s, \mu(s \mid \theta^\mu) \mid \theta^Q) \right]$$

The gradient is computed via the chain rule:

$$\nabla_{\theta^\mu} J \approx \mathbb{E}_{s \sim \mathcal{R}} \left[ \nabla_a Q(s, a \mid \theta^Q) \big|_{a=\mu(s)} \cdot \nabla_{\theta^\mu} \mu(s \mid \theta^\mu) \right]$$

This is the key insight: the actor is optimized by backpropagating through the critic, which provides a differentiable signal for how to improve the action.

### Soft Target Network Updates

Unlike DQN (which copies the target network periodically), DDPG uses **Polyak averaging** to slowly track the learned networks:

$$\theta^{Q'} \leftarrow \tau \theta^Q + (1 - \tau) \theta^{Q'}$$
$$\theta^{\mu'} \leftarrow \tau \theta^\mu + (1 - \tau) \theta^{\mu'}$$

with $\tau \ll 1$ (typically $\tau = 0.001$), providing smooth and stable target updates.

### Exploration

Since DDPG uses a deterministic policy, exploration is achieved by adding noise to the action:

$$a_t = \mu(s_t \mid \theta^\mu) + \mathcal{N}_t$$

The paper uses an **Ornstein-Uhlenbeck (OU) process** for temporally correlated exploration noise, which is beneficial for physical control tasks with inertia:

$$d\mathcal{N}_t = -\theta_{\text{OU}} \, \mathcal{N}_t \, dt + \sigma_{\text{OU}} \, dW_t$$

### Batch Normalization

Each layer's inputs are normalized using batch normalization to handle different physical units across state dimensions and to generalize across tasks with different observation scales.

### Algorithm Summary

```
Initialize actor μ(s|θ^μ), critic Q(s,a|θ^Q), target networks θ^{μ'}, θ^{Q'}
Initialize replay buffer R
for episode = 1 to M do:
    Initialize OU noise process N
    Observe initial state s_1
    for t = 1 to T do:
        Select action a_t = μ(s_t|θ^μ) + N_t
        Execute a_t, observe r_t, s_{t+1}
        Store (s_t, a_t, r_t, s_{t+1}) in R
        Sample minibatch from R
        Update critic by minimizing Bellman error
        Update actor using deterministic policy gradient
        Soft-update target networks
    end for
end for
```

---

## Key Results

### MuJoCo Continuous Control Tasks

| Task | DDPG Return | Planners (with model access) |
|------|-------------|------------------------------|
| Cartpole Swing-up | ~820 | ~860 (iLQG) |
| Pendulum | ~830 | -- |
| Reacher (7-DOF) | ~-4 | ~-4.5 (iLQG) |
| Cheetah | ~3500 | ~3200 (iLQG) |
| Gripper | ~45 | ~52 (MPC) |
| Walker2D | ~1600 | -- |

DDPG matched or exceeded model-based planners on several tasks -- despite being model-free and learning from scratch.

### Learning from Pixels

DDPG successfully learned to solve several tasks directly from 64x64 RGB pixel inputs (using convolutional layers), including:
- Cartpole Swing-up
- Reaching tasks
- Puck-sliding tasks

Performance from pixels was within ~80-90% of performance from low-dimensional state features.

### Comparison to Baselines

| Method | Tasks Solved (out of 20+) |
|--------|--------------------------|
| DDPG | 18+ |
| DPG (original, with tile coding) | <5 |
| NAF | ~10 |
| Stochastic Actor-Critic | <5 |

---

## Impact & Legacy

- **Standard continuous RL algorithm**: DDPG became the default algorithm for continuous control tasks and a fundamental baseline in the RL literature
- **Enabled robotics RL**: Its ability to handle continuous actions made it directly applicable to robotic manipulation and locomotion
- **Spawned improved variants**: Led to TD3 (Twin Delayed DDPG), SAC (Soft Actor-Critic), and other algorithms that addressed DDPG's overestimation and instability issues
- **Influenced multi-agent RL**: MADDPG extended DDPG to multi-agent settings
- **Bridged DQN and policy gradients**: Showed how to combine the off-policy stability of DQN (replay, targets) with the continuous-action capability of policy gradient methods

---

## Key Takeaways

1. **Deterministic policies enable efficient continuous control**: By outputting a single action (instead of a distribution), the policy gradient simplifies to a differentiable chain rule computation through the critic
2. **DQN tricks transfer to actor-critic**: Experience replay and target networks are equally important for stabilizing continuous-action learning
3. **Soft target updates outperform hard copies**: Polyak averaging ($\tau = 0.001$) provides smoother, more stable training than periodic target network replacement
4. **Exploration noise design matters**: Temporally correlated (OU) noise is more effective than i.i.d. Gaussian noise for physical control tasks
5. **A single set of hyperparameters can generalize**: The same algorithm configuration solved a diverse suite of control problems, suggesting good robustness
