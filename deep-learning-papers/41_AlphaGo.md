# Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo)

| Field | Details |
|-------|---------|
| **Authors** | David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, Demis Hassabis |
| **Year** | 2016 |
| **Venue** | Nature |
| **Institution** | Google DeepMind |

---

## Key Contributions

- First computer program to defeat a professional human Go player on a full-sized 19x19 board
- Combined deep convolutional neural networks with Monte Carlo Tree Search (MCTS) in a novel architecture
- Introduced a training pipeline that blends supervised learning from expert games with reinforcement learning from self-play
- Demonstrated that deep learning could tackle problems with search spaces vastly exceeding those of chess ($\sim 10^{170}$ legal positions)
- Defeated the European Go champion Fan Hui 5--0 and later beat world champion Lee Sedol 4--1

---

## Background & Motivation

Go had long been considered the "grand challenge" of AI in board games. Unlike chess, Go has:

- An enormous branching factor (~250 moves per position vs. ~35 in chess)
- A search space of approximately $10^{170}$ legal board positions
- Difficulty in constructing reliable position-evaluation heuristics

Traditional game-playing AI relied on alpha-beta search with handcrafted evaluation functions, which proved insufficient for Go. Prior Go programs using MCTS achieved only amateur-level play. AlphaGo addressed this by replacing handcrafted heuristics with deep neural networks trained through a combination of supervised and reinforcement learning.

---

## Method / Architecture

AlphaGo uses two neural networks -- a **policy network** and a **value network** -- integrated into a Monte Carlo Tree Search framework.

### 1. Supervised Learning (SL) Policy Network

A 13-layer deep convolutional neural network trained on 30 million positions from expert games on the KGS Go Server. The network takes the board state $s$ as input and outputs a probability distribution over legal moves $a$:

$$p_\sigma(a \mid s)$$

The network is trained to maximize the log-likelihood of the expert move $a$:

$$\Delta\sigma \propto \frac{\partial \log p_\sigma(a \mid s)}{\partial \sigma}$$

This achieved a move-prediction accuracy of 57.0% on a held-out test set, significantly surpassing prior state-of-the-art (44.4%).

### 2. Reinforcement Learning (RL) Policy Network

The SL policy network is further improved via self-play reinforcement learning. The RL policy network $p_\rho$ has an identical architecture but is trained using policy gradient methods (REINFORCE). Games are played between the current policy and a randomly selected previous iteration to prevent overfitting:

$$\Delta\rho \propto \frac{\partial \log p_\rho(a_t \mid s_t)}{\partial \rho} z_t$$

where $z_t \in \{-1, +1\}$ is the outcome of the game (win/loss) from the perspective of the current player at time step $t$.

### 3. Value Network

A separate convolutional neural network $v_\theta(s)$ is trained to predict the expected outcome (probability of winning) from position $s$ under self-play with the RL policy:

$$v_\theta(s) \approx v^{p_\rho}(s) = \mathbb{E}[z_t \mid s_t = s, a_{t \ldots T} \sim p_\rho]$$

The value network is trained by regression on self-play outcomes, minimizing mean squared error:

$$\Delta\theta \propto \frac{\partial v_\theta(s)}{\partial \theta}(z - v_\theta(s))$$

To avoid overfitting (since consecutive positions in a game are highly correlated), training data is generated from 30 million distinct positions, each sampled from a separate game.

### 4. Monte Carlo Tree Search (MCTS) Integration

During gameplay, MCTS simulations are guided by the policy and value networks. Each simulation traverses the tree by selecting actions that maximize an upper confidence bound:

$$a_t = \arg\max_a \left( Q(s_t, a) + u(s_t, a) \right)$$

where:

$$u(s, a) \propto \frac{P(s, a)}{1 + N(s, a)}$$

- $Q(s, a)$ is the action-value (mean evaluation across simulations)
- $P(s, a)$ is the prior probability from the SL policy network
- $N(s, a)$ is the visit count

Leaf nodes are evaluated by combining the value network output with a rollout using a fast (but weaker) rollout policy $p_\pi$:

$$V(s_L) = (1 - \lambda)\, v_\theta(s_L) + \lambda\, z_L$$

where $\lambda$ is a mixing parameter (set to 0.5) and $z_L$ is the rollout result.

### Network Architecture Summary

| Component | Architecture | Training Data | Purpose |
|-----------|-------------|---------------|---------|
| SL Policy Network | 13-layer CNN | 30M expert moves | Move prediction prior |
| Fast Rollout Policy | Linear softmax | 30M expert moves | Fast MCTS rollouts |
| RL Policy Network | 13-layer CNN | Self-play games | Improved move selection |
| Value Network | 13-layer CNN | 30M self-play positions | Position evaluation |

---

## Key Results

### Against Other Go Programs

| Opponent | AlphaGo Win Rate |
|----------|-----------------|
| Crazy Stone | 77% |
| Zen | 86% |
| Pachi (100k rollouts) | 99% |
| Fuego | 100% |
| GnuGo | 100% |

### Against Human Players

| Match | Opponent | Result |
|-------|----------|--------|
| October 2015 | Fan Hui (European Champion, 2 dan pro) | 5--0 |
| March 2016 | Lee Sedol (18-time World Champion, 9 dan pro) | 4--1 |

### Ablation Study: Component Contributions

| Configuration | Elo Rating |
|--------------|------------|
| AlphaGo (full system) | 3168 |
| Without value network | 2890 |
| Without rollouts | 3078 |
| Without RL policy (SL only) | 2665 |
| SL policy network alone (no search) | 1517 |

The value network proved more critical than rollouts, and RL training provided roughly 500 Elo points of improvement over supervised learning alone.

---

## Impact & Legacy

- **Landmark AI achievement**: AlphaGo's victory over Lee Sedol was featured on the front page of Nature and covered by global media, becoming one of the most well-known AI accomplishments
- **Catalyzed AlphaGo Zero and AlphaZero**: Subsequent work removed the need for human expert data entirely, learning solely from self-play and achieving even stronger performance
- **Broader RL adoption**: Demonstrated that deep RL combined with search could solve problems previously considered intractable, inspiring applications in robotics, drug discovery, and scientific research
- **MuZero and beyond**: The principles extended to learning environment dynamics without explicit game rules, enabling generalization to Atari, chess, shogi, and Go with a single algorithm
- **Cultural impact**: Changed public perception of AI capabilities and reignited interest in Go worldwide

---

## Key Takeaways

1. **Combining learning and search is powerful**: Neither deep networks alone nor tree search alone achieved top performance; their integration was essential
2. **Supervised learning provides a strong initialization**: Expert data bootstrapped the policy, which RL then refined beyond human-level play
3. **Value functions reduce search depth**: The value network allowed accurate evaluation without rolling out to terminal states, dramatically improving efficiency
4. **Self-play generates unlimited training data**: RL through self-play overcame the limitations of fixed expert datasets
5. **Scale matters**: 1,202 CPUs and 176 GPUs for distributed MCTS during matches -- engineering scale was critical to real-time play at superhuman level
