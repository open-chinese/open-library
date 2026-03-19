# Improved Techniques for Training GANs

**Authors:** Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
**Year:** 2016
**Venue:** NeurIPS 2016
**ArXiv:** 1606.03498

---

## Key Contributions

- Introduced several practical techniques to stabilize GAN training, addressing the notoriously difficult optimization dynamics of adversarial networks.
- Proposed **feature matching**, **minibatch discrimination**, **historical averaging**, **one-sided label smoothing**, and **virtual batch normalization**.
- Introduced the **Inception Score (IS)** as a quantitative metric for evaluating the quality of generated images.
- Achieved state-of-the-art semi-supervised classification results on MNIST, CIFAR-10, and SVHN using GANs.

---

## Background & Motivation

Generative Adversarial Networks, introduced by Goodfellow et al. (2014), frame generative modeling as a two-player minimax game between a generator $G$ and a discriminator $D$:

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

In theory, this game has a Nash equilibrium where $G$ perfectly recovers the data distribution. In practice, however, GAN training is plagued by:

- **Mode collapse:** The generator produces only a small subset of modes from the true distribution.
- **Training instability:** The generator and discriminator oscillate rather than converge.
- **No reliable evaluation metric:** It was unclear how to quantitatively compare different GAN models.

This paper directly addresses all three problems with a suite of practical heuristics and a new evaluation protocol.

---

## Method

### 1. Feature Matching

Instead of training the generator to maximize the discriminator's output directly, feature matching trains $G$ to match the expected value of features on an intermediate layer of the discriminator:

$$\| \mathbb{E}_{x \sim p_{\text{data}}} f(x) - \mathbb{E}_{z \sim p_z(z)} f(G(z)) \|_2^2$$

where $f(x)$ denotes activations on an intermediate layer of the discriminator. This prevents the generator from "overtraining" against the current discriminator.

### 2. Minibatch Discrimination

Mode collapse occurs partly because the discriminator processes each example independently. Minibatch discrimination lets the discriminator look at combinations of examples in a minibatch, enabling it to detect lack of diversity.

Given input $f(x_i) \in \mathbb{R}^A$, multiply by a tensor $T \in \mathbb{R}^{A \times B \times C}$ to get a matrix $M_i \in \mathbb{R}^{B \times C}$. Then compute:

$$c_b(x_i, x_j) = \exp\left(-\| M_{i,b} - M_{j,b} \|_{L_1}\right)$$

$$o(x_i)_b = \sum_{j=1}^{n} c_b(x_i, x_j)$$

The output $o(x_i) \in \mathbb{R}^B$ is concatenated with $f(x_i)$ and fed into the next layer of the discriminator.

### 3. Historical Averaging

Add a penalty term to each player's cost:

$$\left\| \theta - \frac{1}{t} \sum_{i=1}^{t} \theta[i] \right\|^2$$

where $\theta[i]$ is the parameter value at past time $i$. This is inspired by fictitious play in game theory and discourages parameters from oscillating.

### 4. One-Sided Label Smoothing

Replace the target for real examples from $1$ to $0.9$. This prevents the discriminator from giving extremely large gradients to the generator. Importantly, only smooth the positive (real) labels; smoothing the negative (fake) labels can cause problems.

### 5. Virtual Batch Normalization (VBN)

Standard batch normalization causes each sample in a minibatch to depend on every other sample. VBN normalizes each sample using statistics collected from a fixed reference batch chosen at the start of training:

$$\hat{x} = \frac{x - \mu_{\text{ref}}}{\sigma_{\text{ref}}}$$

This eliminates intra-batch dependencies while retaining the benefits of normalization.

### 6. Inception Score

To evaluate sample quality, the authors propose:

$$\text{IS} = \exp\left(\mathbb{E}_x \left[ D_{KL}(p(y|x) \| p(y)) \right]\right)$$

where $p(y|x)$ is the conditional label distribution from a pretrained Inception network, and $p(y) = \int p(y|x = G(z)) \, dz$ is the marginal. A high IS means:
- Each generated image is confidently classified (low entropy of $p(y|x)$).
- The set of images is diverse (high entropy of $p(y)$).

---

## Key Results

### Semi-Supervised Classification (Error Rate %)

| Dataset   | Baseline (Fully Supervised) | GAN-based Semi-Supervised | Number of Labels Used |
|-----------|-----------------------------|---------------------------|-----------------------|
| MNIST     | 0.93                        | **0.93** (±0.065)         | 100                   |
| CIFAR-10  | ~20                         | **18.63** (±2.32)         | 4000                  |
| SVHN      | ~25                         | **8.11** (±1.3)           | 1000                  |

### Inception Scores on CIFAR-10

| Method                          | Inception Score |
|---------------------------------|-----------------|
| Real Data                       | 11.24 ± 0.12   |
| **This paper (minibatch disc.)** | **8.09 ± 0.07** |
| DCGAN                           | 6.40 ± 0.05    |

---

## Impact & Legacy

- The **Inception Score** became one of the most widely used metrics for evaluating GANs (later supplemented by FID).
- **Feature matching** and **minibatch discrimination** became standard tools in the GAN training toolkit.
- The semi-supervised learning results catalyzed a large body of follow-up work on using GANs for learning with limited labels.
- Many subsequent GAN papers (WGAN, Progressive GAN, StyleGAN) built upon the stabilization insights from this work.
- The paper demonstrated that practical engineering heuristics can matter as much as theoretical advances.

---

## Key Takeaways

1. GAN training is fundamentally a game-theoretic optimization problem; standard gradient descent assumptions do not directly apply.
2. Feature matching provides a more stable training signal than directly optimizing generator output against the discriminator.
3. Minibatch discrimination is an effective way to combat mode collapse by allowing the discriminator to reason about sample diversity.
4. The Inception Score was the first widely adopted quantitative metric for GAN evaluation, despite its known limitations (insensitivity to intra-class mode dropping, dependence on the Inception network).
5. GANs are powerful semi-supervised learners: the discriminator learns useful representations of the data even with very few labels.
