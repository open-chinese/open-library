# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Authors:** Sergey Ioffe, Christian Szegedy
**Year:** 2015 (peak adoption 2016)
**Venue:** ICML 2015
**Link:** https://arxiv.org/abs/1502.03167

---

## Key Contributions

- Introduced **Batch Normalization (BatchNorm)**, a technique that normalizes layer inputs using mini-batch statistics, dramatically improving training speed and stability
- Identified and addressed **internal covariate shift** -- the change in the distribution of layer inputs caused by parameter updates in preceding layers
- Enabled training with **much higher learning rates** without divergence
- Reduced the need for careful weight initialization and other regularization techniques like Dropout
- Helped train significantly deeper networks and became a **default component** in virtually all deep learning architectures by 2016

---

## Background & Motivation

Training deep networks is complicated by the fact that each layer's input distribution changes as the parameters of all preceding layers are updated. This phenomenon, termed **internal covariate shift**, has several consequences:

1. **Slow training:** Each layer must continuously adapt to new input distributions
2. **Saturation:** Inputs to activation functions can drift into saturated regions (e.g., large values for sigmoid), causing vanishing gradients
3. **Sensitivity to initialization:** Networks require carefully tuned initial weights to avoid early divergence
4. **Low learning rates:** High learning rates amplify the covariate shift problem, leading to instability

Prior work on **whitening** (decorrelating and normalizing inputs) was computationally expensive and not differentiable in a way suitable for end-to-end training.

---

## Method

### Core Idea

Normalize each feature across the mini-batch, then apply a learned affine transformation to restore representational power.

### Batch Normalization Transform

For a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$ of values for a particular feature (or activation):

**Step 1: Compute mini-batch statistics**

$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**Step 2: Normalize**

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

where $\epsilon$ is a small constant for numerical stability (typically $10^{-5}$).

**Step 3: Scale and shift (learnable parameters)**

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma$ (scale) and $\beta$ (shift) are **learned parameters** per feature dimension.

### Why the Learnable Parameters?

Without $\gamma$ and $\beta$, normalization would constrain the network's representational capacity. By learning $\gamma = \sigma_\mathcal{B}$ and $\beta = \mu_\mathcal{B}$, the network can recover the original unnormalized activation -- meaning BatchNorm is an identity transform in the worst case.

### Placement in the Network

BatchNorm is typically applied **before the activation function**:

$$h = \text{Activation}(\text{BN}(Wx + b))$$

Note: Since BN includes a learned bias $\beta$, the bias $b$ in the linear layer is redundant and can be removed.

### Training vs. Inference

| Phase | Mean | Variance |
|-------|------|----------|
| Training | Mini-batch mean $\mu_\mathcal{B}$ | Mini-batch variance $\sigma_\mathcal{B}^2$ |
| Inference | Running mean $\mathbb{E}[\mu_\mathcal{B}]$ | Running variance $\mathbb{E}[\sigma_\mathcal{B}^2]$ |

During training, running averages are maintained via exponential moving average:

$$\mu_{\text{running}} \leftarrow (1 - \alpha) \cdot \mu_{\text{running}} + \alpha \cdot \mu_\mathcal{B}$$

$$\sigma^2_{\text{running}} \leftarrow (1 - \alpha) \cdot \sigma^2_{\text{running}} + \alpha \cdot \sigma^2_\mathcal{B}$$

At inference, normalization uses these population statistics:

$$y = \gamma \cdot \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}} + \beta$$

This can be folded into a single affine transform for efficient inference.

### Gradient Computation

The gradients through BatchNorm are fully differentiable. For loss $\ell$:

$$\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma$$

$$\frac{\partial \ell}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \frac{-1}{2}(\sigma_\mathcal{B}^2 + \epsilon)^{-3/2}$$

$$\frac{\partial \ell}{\partial \mu_\mathcal{B}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \ell}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

---

## Batch Normalization for Convolutional Networks

For convolutional layers, normalization is applied **per channel** across all spatial locations and batch elements:

| Dimension | Normalization Scope |
|-----------|-------------------|
| Fully connected | Per feature: normalize across batch dimension |
| Convolutional | Per channel: normalize across batch, height, and width |

For a convolutional feature map of shape $(N, C, H, W)$, each channel $c$ has a single $\mu_c$, $\sigma_c^2$, $\gamma_c$, and $\beta_c$, computed across all $N \times H \times W$ values.

---

## Key Results

### ImageNet Classification (Inception Network)

| Configuration | Steps to 72.2% accuracy | Max accuracy |
|---------------|------------------------|-------------|
| Inception (baseline) | $31.0 \times 10^6$ | $72.2\%$ |
| Inception + BN | $13.3 \times 10^6$ | $73.0\%$ |
| Inception + BN (higher LR) | $6.0 \times 10^6$ | $74.0\%$ |
| Inception + BN + modifications | -- | $\mathbf{74.8\%}$ |
| Ensemble (BN-Inception) | -- | $\mathbf{4.82\%}$ top-5 error |

### Training Speed Improvements

| Aspect | Without BN | With BN |
|--------|-----------|---------|
| Learning rate | 0.0015 | 0.045 (30x higher) |
| Convergence speed | Baseline | **14x fewer steps** to reach same accuracy |
| Dropout needed | Yes | No (BN provides regularization) |
| Careful initialization | Required | Less critical |

### Effect on Gradient Flow

BatchNorm keeps activations in the linear regime of activation functions, preventing:
- Vanishing gradients from saturated activations
- Exploding gradients from unconstrained activation growth

---

## Impact & Legacy

Batch Normalization became arguably the **single most impactful practical technique** in deep learning:

- **Universal adoption:** By 2016, virtually every successful deep network used BatchNorm (ResNet, Inception, VGG variants, etc.)
- **Enabled deeper networks:** Critical for training 100+ layer networks like ResNet
- **Training recipe simplification:** Reduced the importance of careful learning rate scheduling, weight initialization, and Dropout
- **Spawned normalization family:** Inspired Layer Normalization, Instance Normalization, Group Normalization, and others
- **Ongoing theoretical debate:** The original "internal covariate shift" explanation has been questioned; alternative explanations include loss landscape smoothing (Santurkar et al., 2018) and implicit regularization
- **Known limitations:** Does not work well with small batch sizes, RNNs, or when training and test batch statistics differ significantly -- motivating the alternative normalization methods

---

## Key Takeaways

1. **Normalize, then rescale:** The two-step process of normalizing activations and then applying learnable $\gamma$ and $\beta$ is the essential recipe -- normalization improves optimization while learnable parameters preserve expressiveness
2. **Higher learning rates are the main practical benefit:** BatchNorm enables 10-30x higher learning rates, which is the primary source of faster training
3. **Batch dependence is both a strength and weakness:** Mini-batch statistics provide regularization (like noise injection) but cause issues at small batch sizes and during inference
4. **The exact mechanism is still debated:** While "reducing internal covariate shift" was the original motivation, later research suggests the main benefit is **smoothing the loss landscape**
5. **Placement matters:** BatchNorm before activation functions is most common, though post-activation placement also works and remains debated
