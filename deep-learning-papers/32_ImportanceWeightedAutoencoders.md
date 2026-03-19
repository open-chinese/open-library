# Importance Weighted Autoencoders

**Authors:** Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
**Year:** 2016
**Venue:** ICLR 2016
**Link:** https://arxiv.org/abs/1509.00519

---

## Key Contributions

- Proposed the **Importance Weighted Autoencoder (IWAE)**, which uses multiple samples from the approximate posterior to form a tighter variational bound
- Proved that the IWAE bound **monotonically improves** as the number of importance samples $k$ increases
- Demonstrated that tighter bounds lead to better generative models with richer latent representations
- Showed that the standard VAE is a special case of IWAE with $k=1$
- Provided both theoretical analysis and empirical evidence that multi-sample objectives are superior to single-sample ELBO

---

## Background & Motivation

The standard VAE optimizes the Evidence Lower Bound (ELBO):

$$\mathcal{L}_1 = \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right] \leq \log p_\theta(x)$$

This bound is loose when the approximate posterior $q_\phi(z|x)$ is far from the true posterior $p_\theta(z|x)$. A loose bound means:

1. The generative model $p_\theta$ receives a biased training signal
2. The learned latent representations may be suboptimal
3. The model underestimates the true log-likelihood

The question: **Can we get a tighter bound without changing the family of approximate posteriors?**

---

## Method

### Importance Weighted ELBO

Instead of using a single sample from $q_\phi(z|x)$, draw $k$ samples $z_1, \ldots, z_k \sim q_\phi(z|x)$ and compute:

$$\mathcal{L}_k = \mathbb{E}_{z_1, \ldots, z_k \sim q_\phi(z|x)} \left[ \log \frac{1}{k} \sum_{i=1}^{k} \frac{p_\theta(x, z_i)}{q_\phi(z_i|x)} \right]$$

This is an importance-weighted estimate of $\log p(x)$ using $q_\phi(z|x)$ as the proposal distribution.

### Key Theoretical Results

**Theorem 1 (Monotonic improvement):**

$$\mathcal{L}_1 \leq \mathcal{L}_2 \leq \cdots \leq \mathcal{L}_k \leq \cdots \leq \log p_\theta(x)$$

The bound gets strictly tighter with more samples.

**Theorem 2 (Convergence):**

$$\lim_{k \to \infty} \mathcal{L}_k = \log p_\theta(x)$$

In the limit, the bound converges to the true marginal log-likelihood.

### Connection to Importance Sampling

The importance weights are defined as:

$$w_i = \frac{p_\theta(x, z_i)}{q_\phi(z_i|x)}$$

The IWAE objective can be rewritten as:

$$\mathcal{L}_k = \mathbb{E}\left[ \log \frac{1}{k} \sum_{i=1}^{k} w_i \right]$$

By Jensen's inequality applied in the opposite direction from the standard ELBO derivation, this provides a tighter bound.

### Gradient Estimator

The gradient with respect to $\theta$ is straightforward:

$$\nabla_\theta \mathcal{L}_k = \mathbb{E}\left[ \sum_{i=1}^{k} \tilde{w}_i \nabla_\theta \log p_\theta(x, z_i) \right]$$

where the **normalized importance weights** are:

$$\tilde{w}_i = \frac{w_i}{\sum_{j=1}^{k} w_j}$$

For $\phi$, the gradient uses the reparameterization trick on each $z_i$:

$$\nabla_\phi \mathcal{L}_k = \mathbb{E}_{\epsilon_1, \ldots, \epsilon_k}\left[ \nabla_\phi \log \frac{1}{k} \sum_{i=1}^{k} w_i \right]$$

### Effective Sample Size

The paper analyzes how increasing $k$ affects the encoder and decoder differently:

| Component | Effect of increasing $k$ |
|-----------|--------------------------|
| Decoder $p_\theta$ | Receives better training signal from tighter bound |
| Encoder $q_\phi$ | Gradient signal can become dominated by few high-weight samples |

This observation later led to research on the "signal-to-noise ratio" problem with IWAE gradients.

---

## Architecture

The IWAE uses the same encoder-decoder architecture as a VAE, with the only difference being in the objective computation:

| Aspect | VAE | IWAE |
|--------|-----|------|
| Samples per datapoint | 1 | $k$ (typically 5--50) |
| Objective | $\mathcal{L}_1$ (ELBO) | $\mathcal{L}_k$ (IW-ELBO) |
| Computational cost | $O(1)$ per datapoint | $O(k)$ per datapoint |
| Bound tightness | Loose | Tighter |

---

## Key Results

### MNIST Log-Likelihood (NLL, nats)

| Model | $k=1$ | $k=5$ | $k=50$ |
|-------|-------|-------|--------|
| VAE (1 stochastic layer) | $-86.47$ | -- | -- |
| IWAE (1 stochastic layer) | $-86.47$ | $-85.54$ | $-84.78$ |
| VAE (2 stochastic layers) | $-85.33$ | -- | -- |
| IWAE (2 stochastic layers) | $-85.33$ | $-84.30$ | $-83.89$ |

### Key Observations

- Increasing $k$ consistently improves the log-likelihood bound
- The improvement is more pronounced for models with more stochastic layers
- IWAE with $k=50$ achieves state-of-the-art results among VAE-based models
- The gap between $\mathcal{L}_k$ and $\mathcal{L}_1$ quantifies how suboptimal the standard ELBO is

### Active Units Analysis

| Model | Number of Active Units |
|-------|----------------------|
| VAE ($k=1$) | 23 |
| IWAE ($k=5$) | 35 |
| IWAE ($k=50$) | 48 |

IWAE uses more latent dimensions effectively, avoiding the "posterior collapse" problem common in VAEs.

---

## Impact & Legacy

- **Tighter bounds became standard:** The idea of using multi-sample bounds influenced subsequent work on variational inference
- **Identified a fundamental tradeoff:** Later work (Rainforth et al., 2018) showed that while the bound improves with $k$, the gradient signal for the encoder degrades -- leading to methods like DReG and STL estimators
- **Influenced flow-based methods:** Motivating research on normalizing flows to improve $q_\phi$ directly rather than relying on importance weighting
- **Reweighted Wake-Sleep:** Inspired connections between importance weighting and the wake-sleep algorithm
- **Foundation for modern methods:** VQ-VAE, hierarchical VAEs, and diffusion models all build on insights from IWAE about the importance of bound tightness

---

## Key Takeaways

1. Using $k$ samples from the approximate posterior and forming an importance-weighted estimate yields a **provably tighter bound** than the standard ELBO
2. The standard VAE objective ($k=1$) is a **special case** of the IWAE objective, meaning IWAE is a strict generalization
3. Tighter bounds lead to **better generative models** and more expressive use of the latent space (more active units)
4. The computational cost scales linearly with $k$, but the improvement has **diminishing returns** -- most of the gain comes from relatively small $k$ (5--50)
5. A subtle tension exists: while the decoder benefits from tighter bounds, the encoder's gradient signal can degrade with large $k$, an insight that sparked significant follow-up research
