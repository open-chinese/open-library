# Deep Networks with Stochastic Depth

**Authors:** Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Q. Weinberger
**Year:** 2016
**Venue:** ECCV 2016
**Link:** https://arxiv.org/abs/1603.09382

---

## Key Contributions

- Introduced **Stochastic Depth**, a training procedure that randomly drops entire layers (residual blocks) during training while using the full network at test time
- Enabled training of **very deep residual networks** (up to 1202 layers) that were previously difficult to optimize
- Demonstrated that stochastic depth acts as an effective **regularizer** and substantially reduces training time
- Provided insights into the **effective depth** of residual networks -- suggesting that not all layers are needed for every input
- Established a conceptual bridge between Dropout (which drops neurons) and this approach (which drops entire layers)

---

## Background & Motivation

ResNets demonstrated that very deep networks (100+ layers) could be trained via skip connections. However, problems remained:

| Problem | Description |
|---------|-------------|
| **Vanishing gradients** | Even with skip connections, very deep networks (>100 layers) suffer gradient degradation |
| **Diminishing returns** | Going from 100 to 1000 layers yields marginal or no improvement |
| **Training cost** | Deeper networks require proportionally more computation |
| **Feature reuse** | Many layers may learn near-identity mappings, doing little useful computation |

The motivation: **Do we really need all layers for every training example?** If not, randomly skipping layers during training could provide both regularization and computational savings.

---

## Method

### Residual Block Review

A standard residual block computes:

$$H_l = \text{ReLU}(f_l(H_{l-1}) + H_{l-1})$$

where $f_l$ is the residual function (typically two or three convolutional layers with batch normalization) and $H_{l-1}$ is the input from the previous block.

### Stochastic Depth

During training, each residual block $l$ is kept with probability $p_l$ and skipped (replaced by identity) with probability $1 - p_l$:

$$H_l = \begin{cases} \text{ReLU}(f_l(H_{l-1}) + H_{l-1}) & \text{with probability } p_l \\ H_{l-1} & \text{with probability } 1 - p_l \end{cases}$$

This can be written compactly using a Bernoulli random variable $b_l \sim \text{Bernoulli}(p_l)$:

$$H_l = \text{ReLU}(b_l \cdot f_l(H_{l-1}) + H_{l-1})$$

When $b_l = 0$, the block reduces to a pure identity mapping, and the input passes straight through.

### Survival Probability Schedule

The paper proposes a **linear decay** schedule for survival probabilities:

$$p_l = 1 - \frac{l}{L}(1 - p_L)$$

where:
- $L$ is the total number of residual blocks
- $p_L$ is the survival probability of the last layer (a hyperparameter, typically $0.5$)
- $l$ ranges from $1$ to $L$

This means:
- **Early layers** (small $l$): $p_l \approx 1$, almost always kept (they extract basic features)
- **Later layers** (large $l$): $p_l \to p_L$, frequently dropped (they tend to learn more refined, redundant features)

| Layer position | $p_l$ (with $p_L = 0.5$) |
|---------------|--------------------------|
| $l = 1$ (first) | $\approx 1.0$ |
| $l = L/4$ | $\approx 0.875$ |
| $l = L/2$ | $\approx 0.75$ |
| $l = 3L/4$ | $\approx 0.625$ |
| $l = L$ (last) | $0.5$ |

### Expected Network Depth

The expected depth during training is:

$$\mathbb{E}[\text{depth}] = \sum_{l=1}^{L} p_l$$

With the linear schedule ($p_L = 0.5$):

$$\mathbb{E}[\text{depth}] = \sum_{l=1}^{L}\left(1 - \frac{l}{L}\cdot 0.5\right) = L - \frac{0.5}{L}\cdot\frac{L(L+1)}{2} \approx \frac{3L}{4}$$

So a 110-layer network effectively trains as a ~82-layer network on average.

### Test-Time Adjustment

At test time, **all layers are used** (no stochastic dropping), but each residual function is scaled by its survival probability to compensate for the training-time expectation:

$$H_l^{\text{test}} = \text{ReLU}(p_l \cdot f_l(H_{l-1}) + H_{l-1})$$

This is analogous to how standard Dropout scales activations at test time.

### Training Algorithm

For each mini-batch:
1. For each residual block $l = 1, \ldots, L$:
   - Sample $b_l \sim \text{Bernoulli}(p_l)$
   - If $b_l = 1$: compute full residual block
   - If $b_l = 0$: pass input directly (identity)
2. Compute loss and backpropagate (gradients only flow through active blocks)
3. Update parameters of active blocks only

---

## Connection to Other Methods

| Method | What is dropped | Granularity | Applied to |
|--------|----------------|-------------|------------|
| Dropout | Individual neurons | Fine | Any layer |
| DropConnect | Individual weights | Very fine | Any layer |
| **Stochastic Depth** | **Entire residual blocks** | **Coarse** | **ResNets** |
| DropPath (later) | Entire paths | Coarse | Multi-branch networks |

Stochastic Depth can be seen as a **structured, layer-level Dropout** that leverages the residual network's identity shortcuts.

---

## Key Results

### CIFAR-10

| Model | Depth | Test Error |
|-------|-------|-----------|
| ResNet (constant depth) | 110 | $6.41\%$ |
| ResNet + Stochastic Depth | 110 | $\mathbf{5.23\%}$ |
| ResNet (constant depth) | 1202 | $6.72\%$ (worse than 110!) |
| ResNet + Stochastic Depth | 1202 | $\mathbf{4.91\%}$ |

The 1202-layer ResNet **without** stochastic depth performs worse than the 110-layer version due to overfitting, but **with** stochastic depth it achieves the best result.

### CIFAR-100

| Model | Depth | Test Error |
|-------|-------|-----------|
| ResNet | 110 | $27.22\%$ |
| ResNet + Stochastic Depth | 110 | $\mathbf{24.58\%}$ |
| ResNet | 1202 | $27.55\%$ |
| ResNet + Stochastic Depth | 1202 | $\mathbf{23.73\%}$ |

### SVHN

| Model | Test Error |
|-------|-----------|
| ResNet-110 | $2.01\%$ |
| ResNet-110 + Stochastic Depth | $\mathbf{1.75\%}$ |

### Training Speedup

| Depth | Training Time (standard) | Training Time (stochastic depth) | Speedup |
|-------|-------------------------|--------------------------------|---------|
| 110 layers | Baseline | $\sim 25\%$ faster | $\sim 1.33\times$ |
| 1202 layers | Baseline | $\sim 25\%$ faster | $\sim 1.33\times$ |

The speedup comes from skipping the forward and backward pass through dropped blocks.

### Sensitivity to $p_L$

| $p_L$ | CIFAR-10 Error (110 layers) |
|--------|---------------------------|
| $1.0$ (no dropping) | $6.41\%$ |
| $0.8$ | $5.60\%$ |
| $0.5$ | $\mathbf{5.23\%}$ |
| $0.2$ | $5.85\%$ |

$p_L = 0.5$ is the sweet spot -- too little dropping provides insufficient regularization, too much dropping undermines learning.

---

## Insights on Effective Depth

The paper provides evidence that:

1. **Not all layers are needed:** Randomly dropping 25% of layers on average actually improves performance, suggesting redundancy in deep networks
2. **Early layers are more important:** The linear decay schedule reflects the empirical finding that early layers extract critical low-level features
3. **Implicit ensemble:** During training, the network samples from $2^L$ possible sub-networks of different depths, creating an implicit ensemble
4. **Gradient flow:** Shorter effective paths during training alleviate vanishing gradient problems

---

## Impact & Legacy

- **Directly influenced DenseNet:** Gao Huang, the lead author, went on to develop DenseNet, which maximizes feature reuse across layers
- **DropPath in NAS:** The technique evolved into DropPath, widely used in Neural Architecture Search and modern architectures (EfficientNet, Vision Transformers)
- **Training very deep networks:** Showed that the training problems of very deep networks can be overcome with stochastic regularization
- **Implicit ensemble interpretation:** Contributed to the understanding of ResNets as ensembles of shallow networks (Veit et al., 2016)
- **Practical adoption:** Stochastic depth / DropPath is a standard component in modern vision architectures including Swin Transformer and ConvNeXt

---

## Key Takeaways

1. **Randomly skipping entire residual blocks** during training provides significant regularization and training speedup, while using all blocks at test time (with probability-scaled outputs)
2. **Linear decay schedule** for survival probabilities -- keeping early layers with high probability and later layers with lower probability -- works best, reflecting the relative importance of layers
3. Stochastic depth enabled training **1202-layer ResNets** that outperform shallower variants, breaking through the depth barrier where standard training degrades
4. The approach suggests deep networks have **significant redundancy**, and different subsets of layers suffice for different inputs
5. The technique evolved into **DropPath**, which became a standard regularizer in modern architectures including Vision Transformers
