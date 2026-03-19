# Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks

**Authors:** Tim Salimans, Diederik P. Kingma
**Year:** 2016
**Venue:** NeurIPS 2016
**Link:** https://arxiv.org/abs/1602.07868

---

## Key Contributions

- Introduced **Weight Normalization**, a reparameterization of neural network weights that decouples the magnitude and direction of weight vectors
- Proposed a technique that is **independent of batch statistics**, unlike Batch Normalization, making it applicable to RNNs, reinforcement learning, and generative models
- Demonstrated faster convergence than unnormalized networks with minimal computational overhead
- Introduced a complementary **data-dependent initialization** scheme that provides many benefits of full Batch Normalization
- Provided theoretical analysis showing that weight normalization improves the conditioning of the optimization problem

---

## Background & Motivation

Batch Normalization accelerates training but has drawbacks:

| Limitation of BatchNorm | Consequence |
|------------------------|-------------|
| Depends on mini-batch statistics | Adds noise, doesn't work with batch size 1 |
| Different behavior train vs. test | Running averages can cause discrepancy |
| Difficult to apply to RNNs | Requires per-timestep statistics |
| Not suitable for noise-sensitive tasks | Generative models, RL need deterministic behavior |

Weight Normalization provides similar optimization benefits by directly reparameterizing the weights, with no batch dependence and negligible overhead.

---

## Method

### Weight Normalization Reparameterization

For a neural network layer with weight vector $\mathbf{w}$ and input $\mathbf{x}$:

$$y = \phi(\mathbf{w} \cdot \mathbf{x} + b)$$

Weight Normalization reparameterizes $\mathbf{w}$ as:

$$\mathbf{w} = \frac{g}{\|\mathbf{v}\|} \mathbf{v}$$

where:
- $\mathbf{v}$ is an unnormalized weight vector (same dimensionality as $\mathbf{w}$)
- $g$ is a scalar magnitude parameter
- $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$ is the Euclidean norm of $\mathbf{v}$

This ensures that $\|\mathbf{w}\| = g$, regardless of the values of $\mathbf{v}$.

### Effect on Optimization

The key insight is that this decoupling separates two aspects of learning:

| Parameter | Controls | Degrees of Freedom |
|-----------|----------|-------------------|
| $g$ | **Magnitude** (length) of the weight vector | 1 scalar |
| $\mathbf{v}$ | **Direction** of the weight vector | $k-1$ (on the unit sphere) |

### Gradients

The gradients with respect to the new parameters:

$$\nabla_g \mathcal{L} = \frac{\nabla_\mathbf{w} \mathcal{L} \cdot \mathbf{v}}{\|\mathbf{v}\|}$$

$$\nabla_\mathbf{v} \mathcal{L} = \frac{g}{\|\mathbf{v}\|} \nabla_\mathbf{w} \mathcal{L} - \frac{g \, \nabla_g \mathcal{L}}{\|\mathbf{v}\|^2} \mathbf{v}$$

The gradient for $\mathbf{v}$ can be rewritten more intuitively:

$$\nabla_\mathbf{v} \mathcal{L} = \frac{g}{\|\mathbf{v}\|} M_\mathbf{w} \nabla_\mathbf{w} \mathcal{L}$$

where $M_\mathbf{w} = I - \frac{\mathbf{w}\mathbf{w}^T}{\|\mathbf{w}\|^2}$ is the **projection matrix** that projects away the component of the gradient in the direction of $\mathbf{w}$.

### Geometric Interpretation

The gradient update for the direction $\mathbf{v}$ only uses the component of $\nabla_\mathbf{w}\mathcal{L}$ that is **perpendicular** to $\mathbf{w}$. This means:

- The direction and magnitude are updated **independently**
- The effective learning rate for the direction is proportional to $\frac{g}{\|\mathbf{v}\|}$, which provides **automatic scaling**
- As $\|\mathbf{v}\|$ grows during training, the effective learning rate for direction updates decreases, providing implicit stabilization

### Comparison: Effective Learning Rate

For a learning rate $\eta$, the effective step size for the weight direction:

$$\|\Delta \hat{\mathbf{w}}\| \approx \frac{\eta g}{\|\mathbf{v}\|} \|\nabla_\mathbf{w}\mathcal{L}\|$$

where $\hat{\mathbf{w}} = \mathbf{w}/\|\mathbf{w}\|$ is the weight direction. This means the effective learning rate is $\frac{\eta g}{\|\mathbf{v}\|}$, which:

1. Decreases as $\|\mathbf{v}\|$ grows, providing **automatic annealing**
2. Is independent of the magnitude $g$, preventing large weights from causing large steps

### Data-Dependent Initialization

To further improve convergence, the paper proposes initializing $g$ and $b$ using a single mini-batch:

$$g \leftarrow \frac{1}{\sigma_{t}}$$

$$b \leftarrow \frac{-\mu_{t}}{\sigma_{t}}$$

where $\mu_t$ and $\sigma_t$ are the mean and standard deviation of the pre-activation $t = \mathbf{v} \cdot \mathbf{x} / \|\mathbf{v}\|$ computed on the initial mini-batch. This ensures that all features initially have zero mean and unit variance, similar to the initialization effect of Batch Normalization.

### Mean-Only Batch Normalization

The paper also proposes a lightweight combination: Weight Normalization + mean-only BN:

$$t = \mathbf{w} \cdot \mathbf{x}$$

$$\tilde{t} = t - \mu_\mathcal{B}[t] + b$$

This subtracts only the batch mean (not dividing by variance), adding minimal noise while fixing the mean shift.

---

## Application to Different Architectures

| Architecture | Weight Norm Applicability |
|-------------|--------------------------|
| Fully connected | Applied to each weight matrix |
| Convolutional | Applied per-filter: each filter $\mathbf{v}_k$ gets its own scalar $g_k$ |
| LSTM / RNN | Applied to each weight matrix without timestep-dependent statistics |
| Generative models | Works with batch size 1, no noise from batch statistics |

For convolutional layers with $K$ filters of shape $(C, H, W)$:

$$\mathbf{w}_k = \frac{g_k}{\|\mathbf{v}_k\|} \mathbf{v}_k, \quad k = 1, \ldots, K$$

Each filter is independently normalized.

---

## Key Results

### CIFAR-10 Classification

| Method | Test Error | Convergence Speed |
|--------|-----------|-------------------|
| No normalization | $8.2\%$ | Baseline |
| Batch Normalization | $7.3\%$ | Fast |
| Weight Normalization | $7.6\%$ | Fast |
| Weight Norm + Mean-Only BN | $\mathbf{7.1\%}$ | Fast |

### Convolutional VAE on MNIST

| Method | NLL (nats) |
|--------|-----------|
| No normalization | $\leq 85.5$ |
| Batch Normalization | $\leq 85.3$ |
| Weight Normalization | $\leq \mathbf{84.3}$ |

Weight Normalization outperformed BatchNorm for generative models, where batch-dependent noise is harmful.

### DRAW (Attention-based Generative Model)

| Method | NLL (nats) |
|--------|-----------|
| Baseline DRAW | $\leq 80.97$ |
| DRAW + Weight Norm | $\leq \mathbf{79.59}$ |

### Reinforcement Learning (DQN on Atari)

| Method | Average Score Improvement |
|--------|--------------------------|
| No normalization | Baseline |
| Batch Normalization | Harmful (added noise destabilizes RL) |
| Weight Normalization | **Consistent improvement** across games |

---

## Impact & Legacy

- **Complementary to other normalizations:** Weight Normalization can be used alongside Layer Normalization or other techniques
- **Generative model training:** Became popular for training GANs and VAEs where batch-dependent normalization adds unwanted noise
- **Spectral Normalization connection:** Later work on Spectral Normalization (Miyato et al., 2018) extended the idea of constraining weight matrices for GAN training
- **Theoretical insights:** The analysis of optimization geometry influenced understanding of how normalization helps training in general
- **Practical simplicity:** Easy to implement (a few lines of code), no running statistics to maintain, deterministic behavior

---

## Key Takeaways

1. **Decouple magnitude and direction:** Reparameterizing $\mathbf{w} = \frac{g}{\|\mathbf{v}\|}\mathbf{v}$ separates "how large" from "which direction," improving optimization conditioning
2. **No batch dependence:** Unlike BatchNorm, Weight Normalization is purely a weight reparameterization with no dependency on mini-batch statistics, making it suitable for RNNs, small batches, and noise-sensitive applications
3. **Automatic learning rate adaptation:** The effective learning rate for the weight direction scales as $1/\|\mathbf{v}\|$, providing implicit stabilization without explicit scheduling
4. **Data-dependent initialization** provides the initialization benefits of BatchNorm without ongoing batch dependence
5. **Best suited for applications where BatchNorm fails:** Generative models, RL, RNNs, and online learning benefit most from Weight Normalization
