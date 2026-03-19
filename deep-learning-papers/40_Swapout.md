# Swapout: Learning an Ensemble of Deep Architectures

**Authors:** Saurabh Singh, Derek Hoiem, David Forsyth
**Year:** 2016
**Venue:** NeurIPS 2016
**Link:** https://arxiv.org/abs/1605.06465

---

## Key Contributions

- Introduced **Swapout**, a stochastic training scheme that generalizes both Dropout and Stochastic Depth within a unified framework for residual networks
- Showed that independently sampling whether to include the skip connection and the residual function for each layer creates a **richer ensemble of architectures** during training
- Demonstrated that the optimal stochastic strategy differs from both Dropout and Stochastic Depth, suggesting the design space is richer than previously thought
- Provided systematic experimental analysis comparing different stochastic training strategies for residual networks
- Achieved competitive state-of-the-art results on CIFAR-10 and CIFAR-100

---

## Background & Motivation

In a standard residual network, the output of block $l$ is:

$$H_l = H_{l-1} + f_l(H_{l-1})$$

Several stochastic training methods modify this computation:

| Method | Formulation | What varies |
|--------|------------|-------------|
| Standard ResNet | $H_l = H_{l-1} + f_l(H_{l-1})$ | Nothing |
| Dropout | $H_l = H_{l-1} + f_l(\text{drop}(H_{l-1}))$ | Neurons within $f_l$ |
| Stochastic Depth | $H_l = H_{l-1} + b_l \cdot f_l(H_{l-1})$ | Whether $f_l$ is active |
| **Swapout** | $H_l = \Theta_l^{(1)} \odot H_{l-1} + \Theta_l^{(2)} \odot f_l(H_{l-1})$ | Both terms independently |

The key insight: **why couple the skip connection and residual function?** What if we independently decide for each unit whether to include the identity, the residual, both, or neither?

---

## Method

### Swapout Formulation

For residual block $l$, the output is:

$$H_l = \Theta_l^{(1)} \odot H_{l-1} + \Theta_l^{(2)} \odot f_l(H_{l-1})$$

where:
- $\Theta_l^{(1)} \in \{0, 1\}^d$ is a binary mask for the **skip connection** (identity path)
- $\Theta_l^{(2)} \in \{0, 1\}^d$ is a binary mask for the **residual function**
- $\odot$ denotes element-wise multiplication
- Each element is sampled independently: $\Theta_{l,j}^{(1)} \sim \text{Bernoulli}(\theta_1)$ and $\Theta_{l,j}^{(2)} \sim \text{Bernoulli}(\theta_2)$

### Four Possible States Per Unit

For each hidden unit $j$ in layer $l$, there are four possible states:

| $\Theta_{l,j}^{(1)}$ | $\Theta_{l,j}^{(2)}$ | Output $H_{l,j}$ | Interpretation |
|---|---|---|---|
| 0 | 0 | $0$ | Unit is dropped entirely |
| 1 | 0 | $H_{l-1,j}$ | Pure skip connection |
| 0 | 1 | $f_l(H_{l-1})_j$ | Pure feedforward (no skip) |
| 1 | 1 | $H_{l-1,j} + f_l(H_{l-1})_j$ | Standard residual connection |

### Special Cases

Swapout unifies several existing methods:

| $\theta_1$ | $\theta_2$ | Method |
|-----------|-----------|--------|
| $1$ | $1$ | Standard ResNet (deterministic) |
| $1$ | $b_l \in \{0,1\}$ (same for all units) | Stochastic Depth |
| $1$ | Element-wise Bernoulli | Skip-Forward (partial Swapout) |
| Element-wise Bernoulli | Element-wise Bernoulli | **Full Swapout** |

### Stochastic Depth as a Special Case

In Stochastic Depth, a single Bernoulli variable $b_l$ controls the entire residual function:

$$H_l = H_{l-1} + b_l \cdot f_l(H_{l-1}), \quad b_l \sim \text{Bernoulli}(p_l)$$

In Swapout, this is generalized so that each element of the skip and residual can be independently toggled.

### Dropout Connection

Standard Dropout applied to the residual block can be seen as:

$$H_l = H_{l-1} + f_l(\text{diag}(\mathbf{z}) \cdot H_{l-1})$$

where $\mathbf{z} \sim \text{Bernoulli}(p)$. Swapout instead applies stochasticity to the **output** of both paths, which the authors argue provides a richer and more effective form of regularization.

### Inference Strategies

At test time, several strategies can be used:

**1. Deterministic Scaling (Expected Value)**

$$H_l^{\text{test}} = \theta_1 \cdot H_{l-1} + \theta_2 \cdot f_l(H_{l-1})$$

This uses the expected value of the masks, analogous to Dropout scaling.

**2. Stochastic Inference (MC Averaging)**

Run $T$ forward passes with random masks and average:

$$\hat{y} = \frac{1}{T}\sum_{t=1}^{T} f^{(\Theta^{(t)})}(x)$$

**3. Parameter Search**

Replace $\theta_1$ and $\theta_2$ with tunable parameters $\alpha_1$ and $\alpha_2$:

$$H_l^{\text{test}} = \alpha_1 \cdot H_{l-1} + \alpha_2 \cdot f_l(H_{l-1})$$

and optimize $\alpha_1, \alpha_2$ on the validation set.

---

## Architecture Details

The experiments use a standard ResNet architecture:

| Component | Configuration |
|-----------|--------------|
| Base architecture | ResNet (pre-activation) |
| Block type | BasicBlock (two $3 \times 3$ conv layers) |
| Depth | 20, 32, or custom |
| Width | Standard or widened (WRN-style) |
| Swapout applied to | Output of each residual block |

### Stochastic Schedule

Like Stochastic Depth, the probabilities can follow a schedule across layers:

| Schedule | $\theta_1^{(l)}$ | $\theta_2^{(l)}$ |
|----------|-----------------|-----------------|
| Constant | Same for all $l$ | Same for all $l$ |
| Linear decay | $1 - \frac{l}{L}(1-\theta_1^{(L)})$ | $1 - \frac{l}{L}(1-\theta_2^{(L)})$ |

---

## Key Results

### CIFAR-10

| Model | Depth | Test Error |
|-------|-------|-----------|
| ResNet | 20 | $8.75\%$ |
| ResNet + Dropout | 20 | $8.52\%$ |
| ResNet + Stochastic Depth | 20 | $8.32\%$ |
| ResNet + Swapout (deterministic) | 20 | $7.95\%$ |
| ResNet + Swapout (stochastic) | 20 | $\mathbf{7.68\%}$ |

| Model | Depth | Test Error |
|-------|-------|-----------|
| ResNet | 32 | $7.51\%$ |
| ResNet + Swapout (deterministic) | 32 | $6.89\%$ |
| ResNet + Swapout (stochastic) | 32 | $\mathbf{6.58\%}$ |

### CIFAR-100

| Model | Depth | Test Error |
|-------|-------|-----------|
| ResNet | 32 | $30.40\%$ |
| ResNet + Stochastic Depth | 32 | $28.92\%$ |
| ResNet + Swapout (deterministic) | 32 | $27.57\%$ |
| ResNet + Swapout (stochastic) | 32 | $\mathbf{26.86\%}$ |

### Inference Strategy Comparison (CIFAR-10, ResNet-32)

| Inference Method | Test Error |
|-----------------|-----------|
| Deterministic ($\theta_1, \theta_2$) | $6.89\%$ |
| Stochastic (10 samples) | $6.72\%$ |
| Stochastic (50 samples) | $6.61\%$ |
| Stochastic (100 samples) | $\mathbf{6.58\%}$ |

### Optimal Probabilities

The paper found that the best configurations are **not** the special cases:

| Dataset | Best $\theta_1$ | Best $\theta_2$ | Interpretation |
|---------|-----------------|-----------------|----------------|
| CIFAR-10 | $\sim 0.9$ | $\sim 0.5$ | Keep skip mostly, drop residual often |
| CIFAR-100 | $\sim 0.9$ | $\sim 0.5$ | Similar pattern |

This suggests that:
- The skip connection should be kept with high probability (identity paths are important)
- The residual function benefits from significant stochastic dropping (regularization)
- The optimal point is neither standard ResNet ($\theta_1=\theta_2=1$) nor any existing method

---

## Analysis and Insights

### Ensemble Interpretation

A network with $L$ Swapout layers and $d$ hidden units per layer trains over an implicit ensemble of architectures. The number of possible sub-architectures is:

$$4^{L \cdot d}$$

since each unit at each layer has 4 possible states. This is a much larger ensemble than Stochastic Depth ($2^L$ sub-networks) or standard Dropout.

### Gradient Flow Analysis

- When $\Theta^{(1)} = 1, \Theta^{(2)} = 0$: gradients flow directly through identity (no vanishing)
- When $\Theta^{(1)} = 0, \Theta^{(2)} = 1$: network behaves like a feedforward network (no skip)
- The mixture of these paths creates diverse gradient pathways, improving optimization

### Connection to Implicit Ensembles

The paper draws on the insight from Veit et al. (2016) that ResNets can be viewed as ensembles of paths of different lengths. Swapout enriches this ensemble by allowing **partial paths** where some units use the skip and others use the residual within the same layer.

---

## Impact & Legacy

- **Unified framework:** Demonstrated that Dropout, Stochastic Depth, and other stochastic regularizers are points in a larger design space
- **Influenced DropPath:** The idea of independently dropping paths contributed to the development of DropPath, now standard in Vision Transformers
- **Architecture search insight:** Showed that the optimal stochastic configuration is not obvious and should be tuned, presaging Neural Architecture Search
- **Ensemble understanding:** Deepened understanding of how residual networks implicitly form ensembles and how stochastic training enriches these ensembles
- **Stochastic inference gains:** Demonstrated that MC averaging at test time (similar to MC Dropout) provides additional accuracy gains

---

## Key Takeaways

1. **Swapout generalizes both Dropout and Stochastic Depth** by independently sampling binary masks for the skip connection and residual function at the level of individual units
2. The optimal configuration has **high skip probability ($\sim 0.9$) and moderate residual probability ($\sim 0.5$)**, which does not correspond to any previously proposed method
3. **Stochastic inference** (averaging over multiple forward passes with random masks) consistently outperforms deterministic scaling, confirming the ensemble interpretation
4. The framework reveals that residual networks benefit from a **very large implicit ensemble** of sub-architectures ($4^{L \cdot d}$ possible configurations)
5. The key practical lesson: **the identity path in ResNets is critical** (should rarely be dropped), while the residual function can tolerate aggressive stochastic dropping for regularization benefit
