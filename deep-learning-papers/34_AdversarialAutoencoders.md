# Adversarial Autoencoders

**Authors:** Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey
**Year:** 2016
**Venue:** ICLR 2016 Workshop (arXiv 2015)
**Link:** https://arxiv.org/abs/1511.05644

---

## Key Contributions

- Introduced the **Adversarial Autoencoder (AAE)**, which replaces the KL divergence regularization in VAEs with adversarial training to match the aggregated posterior to an arbitrary prior
- Enabled the use of **any prior distribution** (not just simple Gaussians) without requiring a tractable KL divergence
- Demonstrated versatile applications: unsupervised clustering, semi-supervised classification, and dimensionality reduction
- Showed that adversarial regularization produces **sharper reconstructions** than KL-based VAEs while maintaining a structured latent space
- Bridged the gap between VAEs and GANs by combining reconstruction-based and adversarial objectives

---

## Background & Motivation

In a standard VAE, the latent space is regularized via the KL divergence:

$$D_{KL}(q_\phi(z|x) \| p(z))$$

This has two main limitations:

1. **Restricted prior choices:** The KL divergence must be tractable, limiting the prior $p(z)$ to simple distributions (typically $\mathcal{N}(0, I)$)
2. **Blurry outputs:** The KL penalty can over-regularize, causing the decoder to produce blurry reconstructions

GANs can match arbitrary distributions without computing likelihoods, but they lack an encoder for inference. AAE combines the best of both worlds.

---

## Method

### Architecture

An AAE consists of three components:

| Component | Role | Parameters |
|-----------|------|------------|
| Encoder $q_\phi(z\|x)$ | Maps data to latent codes | $\phi$ |
| Decoder $p_\theta(x\|z)$ | Reconstructs data from latent codes | $\theta$ |
| Discriminator $D_\omega(z)$ | Distinguishes between $q_\phi(z)$ and $p(z)$ | $\omega$ |

### Two-Phase Training

Training alternates between two phases in each iteration:

**Phase 1: Reconstruction Phase**

Update the encoder and decoder to minimize the reconstruction loss:

$$\mathcal{L}_{\text{rec}} = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \| x - \text{Dec}(\text{Enc}(x)) \|^2 \right]$$

or equivalently, for binary data:

$$\mathcal{L}_{\text{rec}} = -\mathbb{E}_{x \sim p_{\text{data}}} \left[ \sum_i x_i \log \hat{x}_i + (1 - x_i)\log(1 - \hat{x}_i) \right]$$

**Phase 2: Regularization Phase (Adversarial)**

Update the discriminator and encoder using the adversarial objective. The discriminator learns to distinguish between:
- **Real samples:** $z \sim p(z)$ (drawn from the desired prior)
- **Fake samples:** $z = \text{Enc}(x)$, $x \sim p_{\text{data}}$ (the aggregated posterior)

The adversarial loss is:

$$\min_\phi \max_\omega \; \mathbb{E}_{z \sim p(z)}[\log D_\omega(z)] + \mathbb{E}_{x \sim p_{\text{data}}}[\log(1 - D_\omega(\text{Enc}(x)))]$$

### Aggregated Posterior Matching

The key insight is that instead of matching $q_\phi(z|x)$ to $p(z)$ for each $x$ (as the VAE does with KL), the AAE matches the **aggregated posterior**:

$$q_\phi(z) = \int q_\phi(z|x) \, p_{\text{data}}(x) \, dx$$

to the prior $p(z)$. This is a weaker constraint that still ensures meaningful generation from $p(z)$ while giving the encoder more freedom per datapoint.

### Deterministic vs. Stochastic Encoder

The encoder can be either:

| Type | Output | Aggregated Posterior |
|------|--------|---------------------|
| **Deterministic** | $z = f_\phi(x)$ | $q_\phi(z) = \int \delta(z - f_\phi(x)) p_{\text{data}}(x) dx$ |
| **Stochastic** | $z \sim q_\phi(z\|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ | $q_\phi(z) = \int q_\phi(z\|x) p_{\text{data}}(x) dx$ |

The deterministic variant works well in practice and is simpler to implement.

### Flexible Priors

Since the adversarial objective only requires samples from $p(z)$, the prior can be:

- **Gaussian mixture:** $p(z) = \sum_k \pi_k \mathcal{N}(\mu_k, \Sigma_k)$ for clustering
- **Swiss roll or other manifolds:** For specific latent space geometries
- **Categorical + continuous:** For disentangled representations
- **Any distribution** from which we can sample

---

## Semi-Supervised Learning with AAE

For semi-supervised classification, the latent code is split into two parts:

$$z = [y, s]$$

where $y$ is a categorical variable (class label) and $s$ is a continuous style variable.

The training incorporates:

1. **Reconstruction:** $\mathcal{L}_{\text{rec}}(x, \text{Dec}(y, s))$
2. **Adversarial on $s$:** Match $q(s)$ to $\mathcal{N}(0, I)$
3. **Adversarial on $y$:** Match $q(y)$ to $\text{Cat}(1/K, \ldots, 1/K)$
4. **Supervised:** Cross-entropy on labeled examples: $\mathcal{L}_{\text{sup}} = -\mathbb{E}_{(x,y) \sim \text{labeled}}[\log q(y|x)]$

---

## Key Results

### MNIST Generation Quality

| Model | Reconstruction Quality | Latent Space Structure |
|-------|----------------------|----------------------|
| VAE | Blurry | Smooth, structured |
| AAE (Gaussian prior) | Sharper | Smooth, structured |
| AAE (Mixture of 10 Gaussians) | Sharp | Clustered by digit |

### Semi-Supervised Classification on MNIST

| Model | 100 labels | 1000 labels |
|-------|-----------|-------------|
| M1+M2 (Kingma et al., 2014) | $3.33\%$ | -- |
| AAE (1000 clusters) | $4.10\%$ | $1.90\%$ |
| AAE (16 clusters) | $\mathbf{1.90\%}$ | $\mathbf{1.60\%}$ |

### Clustering on MNIST (Unsupervised)

| Model | Clustering Accuracy |
|-------|-------------------|
| K-means | $53.3\%$ |
| AAE (16 clusters) | $\sim 90\%$ |
| AAE (30 clusters) | $\sim 95\%$ |

### Comparison of Regularization Approaches

| Method | KL Tractability Required | Arbitrary Priors | Reconstruction Quality |
|--------|-------------------------|------------------|----------------------|
| VAE (KL) | Yes | No | Blurry |
| WAE (MMD) | No | Yes | Moderate |
| AAE (Adversarial) | No | Yes | Sharp |

---

## Impact & Legacy

- **Flexible priors:** Showed that VAE-style models are not limited to simple Gaussian priors, opening up a wide design space
- **Hybrid VAE-GAN models:** Pioneered the combination of reconstruction and adversarial objectives that became popular (VAE-GAN, ALI/BiGAN, etc.)
- **Clustering applications:** The ability to impose mixture priors made AAEs popular for unsupervised and semi-supervised clustering
- **Wasserstein Autoencoder (WAE):** Theoretically grounded the aggregated posterior matching approach that AAE introduced
- **Practical impact:** Used in drug discovery, anomaly detection, and representation learning where specific latent space structures are desired

---

## Key Takeaways

1. Replacing KL divergence with **adversarial training** for latent space regularization removes the requirement for tractable KL computation and enables arbitrary prior distributions
2. The AAE matches the **aggregated posterior** $q(z)$ to $p(z)$, which is a weaker but more practical constraint than per-datapoint KL matching
3. The **two-phase training** (reconstruction + adversarial regularization) is simple to implement and stable in practice
4. AAEs are remarkably versatile -- the same framework handles **unsupervised generation, clustering, semi-supervised learning, and disentanglement** by simply changing the prior structure
5. The work demonstrated that **the best of VAEs (inference, reconstruction) and GANs (flexible distribution matching) can be combined** in a single framework
