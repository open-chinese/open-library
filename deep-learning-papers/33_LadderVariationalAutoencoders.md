# Ladder Variational Autoencoders

**Authors:** Casper Kaae Sonderby, Tapani Raiko, Lars Maaloe, Soren Kaae Sonderby, Ole Winther
**Year:** 2016
**Venue:** NeurIPS 2016
**Link:** https://arxiv.org/abs/1602.02282

---

## Key Contributions

- Introduced the **Ladder VAE (LVAE)**, which combines ideas from ladder networks with hierarchical variational autoencoders
- Proposed a **top-down inference pathway** that recursively corrects the approximate posterior using information from higher layers
- Addressed the critical problem of **posterior collapse** in deep hierarchical VAEs, where higher stochastic layers tend to be ignored
- Demonstrated that sharing information between the generative and inference models enables training of **deeper hierarchical latent variable models**
- Achieved state-of-the-art density estimation results on standard benchmarks

---

## Background & Motivation

Hierarchical VAEs stack multiple layers of latent variables to capture multi-scale structure:

$$p_\theta(x, z_1, \ldots, z_L) = p(z_L) \prod_{l=1}^{L-1} p_\theta(z_l | z_{l+1}) \cdot p_\theta(x | z_1)$$

However, training deep hierarchical VAEs is notoriously difficult:

1. **Posterior collapse:** Higher layers of latent variables tend to become inactive, with $q_\phi(z_l|x) \approx p(z_l)$, meaning the model ignores those layers
2. **Optimization difficulty:** The approximate posterior for each layer must be accurate enough to provide useful gradients
3. **Information gap:** Bottom-up inference (encoder) and top-down generation (decoder) use completely separate pathways, leading to poor coordination

The **ladder network** (Rasmus et al., 2015) showed that combining bottom-up and top-down information improves semi-supervised learning. The Ladder VAE applies this principle to variational inference.

---

## Method

### Architecture Overview

The Ladder VAE modifies the standard hierarchical VAE by introducing a **top-down inference pathway** that merges bottom-up (data-driven) information with top-down (prior-driven) information.

### Bottom-Up Pass (Deterministic)

A standard feedforward encoder computes deterministic features at each layer:

$$d_l = f_l^{\text{enc}}(d_{l-1}), \quad l = 1, \ldots, L$$

where $d_0 = x$ and $f_l^{\text{enc}}$ are neural network layers. From these, compute proposal parameters:

$$\hat{\mu}_l, \hat{\sigma}_l^2 = g_l(d_l)$$

These are **not** the final approximate posterior parameters -- they are proposals that will be refined.

### Top-Down Pass (Stochastic)

Starting from the top layer, the model recursively refines the approximate posterior by combining bottom-up proposals with top-down prior information.

At each layer $l$ (from $L$ down to $1$):

1. Compute the **prior** conditioned on the layer above:

$$p_\theta(z_l | z_{l+1}) = \mathcal{N}(z_l; \mu_l^{\text{prior}}, (\sigma_l^{\text{prior}})^2)$$

where $\mu_l^{\text{prior}}, \sigma_l^{\text{prior}}$ are functions of $z_{l+1}$.

2. **Merge** the bottom-up proposal with the top-down prior using precision-weighted combination:

$$\sigma_l^2 = \frac{1}{\frac{1}{\hat{\sigma}_l^2} + \frac{1}{(\sigma_l^{\text{prior}})^2}}$$

$$\mu_l = \sigma_l^2 \left( \frac{\hat{\mu}_l}{\hat{\sigma}_l^2} + \frac{\mu_l^{\text{prior}}}{(\sigma_l^{\text{prior}})^2} \right)$$

3. Sample $z_l \sim \mathcal{N}(\mu_l, \sigma_l^2)$

This precision-weighted merging is the **key innovation** -- it acts like a Kalman filter, combining two sources of information optimally under Gaussian assumptions.

### The ELBO for Ladder VAE

The training objective is the standard hierarchical ELBO:

$$\mathcal{L} = \mathbb{E}_{q_\phi(z_{1:L}|x)}\left[\log p_\theta(x|z_1)\right] - \sum_{l=1}^{L} \mathbb{E}_{q_\phi(z_{>l}|x)}\left[D_{KL}(q_\phi(z_l|z_{>l}, x) \| p_\theta(z_l|z_{l+1}))\right]$$

Because both the approximate posterior and the prior at each layer are Gaussian, the KL terms have **closed-form solutions**:

$$D_{KL}(q_\phi(z_l | \cdot) \| p_\theta(z_l | z_{l+1})) = \frac{1}{2}\sum_j \left[\frac{\sigma_{l,j}^2 + (\mu_{l,j} - \mu_{l,j}^{\text{prior}})^2}{(\sigma_{l,j}^{\text{prior}})^2} - 1 - \log \frac{\sigma_{l,j}^2}{(\sigma_{l,j}^{\text{prior}})^2}\right]$$

### Warm-Up Strategy

To further combat posterior collapse, the paper employs **KL warm-up** (also called KL annealing), where the KL term is scaled by a factor $\beta$ that is gradually increased from 0 to 1 during training:

$$\mathcal{L}_{\text{warm-up}} = \mathbb{E}[\log p_\theta(x|z_1)] - \beta \sum_l D_{KL}(\cdot)$$

---

## Comparison of Inference Approaches

| Approach | Inference Path | Posterior Collapse | Posterior Quality |
|----------|---------------|-------------------|-------------------|
| Standard Hierarchical VAE | Purely bottom-up | Severe in deep models | Limited by encoder capacity |
| Ladder VAE | Bottom-up + top-down merge | Significantly reduced | Improved via precision weighting |
| Normalizing Flow VAE | Bottom-up + flow transform | Moderate | Improved via flexible distributions |

---

## Key Results

### MNIST (NLL in nats, importance sampled with $k=5000$)

| Model | Test NLL |
|-------|----------|
| VAE (1 layer) | $\leq 86.5$ |
| IWAE ($k=50$) | $\leq 84.8$ |
| DRAW | $\leq 80.97$ |
| Ladder VAE (5 layers) | $\leq 81.74$ |
| Ladder VAE (5 layers) + warm-up | $\leq 81.02$ |

### OMNIGLOT

| Model | Test NLL |
|-------|----------|
| VAE | $\leq 107.62$ |
| IWAE ($k=50$) | $\leq 103.38$ |
| Ladder VAE (5 layers) | $\leq 102.11$ |

### Active Units Across Layers

| Layer | Standard Hierarchical VAE | Ladder VAE |
|-------|--------------------------|------------|
| $z_1$ (bottom) | Active | Active |
| $z_2$ | Often inactive | Active |
| $z_3$ | Almost always inactive | Active |
| $z_4$ | Inactive | Active |
| $z_5$ (top) | Inactive | Active |

The Ladder VAE successfully keeps all layers active, demonstrating that the top-down inference path resolves posterior collapse.

---

## Impact & Legacy

- **Solved posterior collapse for hierarchical VAEs:** Demonstrated that the combination of bottom-up and top-down inference is critical for deep latent variable models
- **Influenced NVAE and VDVAE:** Modern very deep VAEs (Vahdat & Kautz, 2020; Child, 2021) adopt the same top-down inference strategy pioneered by Ladder VAE
- **Precision-weighted merging:** The idea of combining prior and data-driven information via precision weighting became a standard technique in hierarchical VAE design
- **Connection to neuroscience:** The top-down/bottom-up merging resembles predictive coding theories of brain function, inspiring cross-disciplinary research
- **KL warm-up:** While not originated here, the paper helped popularize this training strategy, which became standard practice

---

## Key Takeaways

1. The core innovation is **merging bottom-up encoder features with top-down generative model information** using precision-weighted Gaussian combination at each layer
2. This approach effectively addresses **posterior collapse** in deep hierarchical VAEs, keeping all latent layers active and informative
3. The architecture acts like a **recursive refinement** process: starting from a coarse top-level representation and progressively adding detail at each lower layer
4. **KL warm-up** remains important even with the ladder architecture, suggesting that optimization dynamics still play a role
5. The top-down inference paradigm introduced here became the **standard approach** for all modern deep hierarchical VAEs
