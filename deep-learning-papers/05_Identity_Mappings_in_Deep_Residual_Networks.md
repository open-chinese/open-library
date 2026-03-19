# Identity Mappings in Deep Residual Networks

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Year:** 2016
**Venue:** ECCV 2016
**ArXiv:** 1603.05027

---

## Key Contributions

- Provided a thorough analysis of residual network design, showing that **clean identity shortcuts** (without any gating, scaling, or convolutions on the skip path) are essential for effective information propagation.
- Proposed the **pre-activation** residual block where Batch Normalization (BN) and ReLU are placed *before* the convolution (BN-ReLU-Conv), as opposed to the original post-activation design (Conv-BN-ReLU).
- Demonstrated both theoretically and empirically that pre-activation ResNets train more easily, generalize better, and avoid optimization difficulties that arise in very deep networks.
- Achieved improved results on CIFAR-10, CIFAR-100, and ImageNet compared to the original ResNet design.

---

## Background & Motivation

The original ResNet paper demonstrated that skip connections enable training of very deep networks. However, the precise design of the residual block was not thoroughly explored. This paper asks two fundamental questions:

1. **What should the skip connection look like?** Should it be a pure identity, or could gating/scaling/projection improve results?
2. **Where should BN and ReLU be placed?** The original block uses $\text{Conv} \rightarrow \text{BN} \rightarrow \text{ReLU}$; is this optimal?

---

## Method

### General Formulation

A residual unit is expressed as:

$$\mathbf{y}_l = h(\mathbf{x}_l) + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$

$$\mathbf{x}_{l+1} = f(\mathbf{y}_l)$$

where:
- $\mathbf{x}_l$ is the input to the $l$-th residual unit
- $h(\mathbf{x}_l)$ is the skip connection
- $\mathcal{F}$ is the residual function
- $f$ is the post-addition activation

### Part 1: Analysis of Skip Connection Design

The authors systematically compare different choices for $h(\mathbf{x}_l)$:

| Shortcut Type             | Formula                                      | CIFAR-10 Error (%) |
|---------------------------|----------------------------------------------|---------------------|
| **Identity (original)**   | $h(\mathbf{x}) = \mathbf{x}$                | **6.61**            |
| Constant scaling          | $h(\mathbf{x}) = \lambda \mathbf{x}$        | 12.35 ($\lambda=0.5$)|
| Exclusive gating          | $h(\mathbf{x}) = (1 - g(\mathbf{x})) \odot \mathbf{x}$ | 8.70      |
| Shortcut-only gating      | $h(\mathbf{x}) = g(\mathbf{x}) \odot \mathbf{x}$ | 12.86          |
| Convolutional shortcut    | $h(\mathbf{x}) = W_s \mathbf{x}$ (1x1 conv) | 6.91               |
| Dropout shortcut          | Random zero-out of shortcut                  | 7.17               |

**Conclusion:** The pure identity shortcut consistently performs best. Any modification to the skip path impedes information flow and hurts optimization.

### Theoretical Justification for Clean Identity

If $h(\mathbf{x}_l) = \mathbf{x}_l$ (identity) and $f$ is also identity, then by unrolling the recursion:

$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$$

This means that the feature at any deeper layer $L$ is the feature at layer $l$ plus a sum of residuals. The gradient with respect to the loss $\mathcal{E}$ is:

$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \left(1 + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)\right)$$

The critical term is the **1** in the parentheses: it guarantees that the gradient can flow directly from any layer $L$ back to any layer $l$ without attenuation. If $h$ involves scaling by $\lambda < 1$, this term becomes $\lambda^{L-l}$, which vanishes exponentially for deep networks.

### Part 2: Analysis of Activation Function Placement

The authors compare different orderings of BN, ReLU, Conv, and the addition:

| Configuration         | Order                                              | CIFAR-10 Error (%) |
|-----------------------|----------------------------------------------------|---------------------|
| Original (post-act)   | Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU   | 6.61               |
| BN after addition     | Conv -> BN -> ReLU -> Conv -> Add -> BN -> ReLU   | 8.17               |
| ReLU before addition  | Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Add   | 7.84               |
| ReLU-only pre-act     | ReLU -> Conv -> ReLU -> Conv -> Add               | 7.71               |
| **Full pre-activation** | **BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Add** | **4.92** (ResNet-1001) |

### The Pre-Activation Residual Block

The recommended design places BN and ReLU *before* each convolution:

```
Input x_l
  |
  +--> BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) --> F(x_l)
  |                                                              |
  +---------------------- identity shortcut --------------------(+)--> x_{l+1}
```

This design has two important properties:

1. **The addition sees clean, unnormalized, un-activated features on both paths.** The post-addition signal feeds directly into the next block's BN-ReLU, ensuring the skip path is a true identity.

2. **BN and ReLU serve as pre-activation for the weight layers**, acting as a "normalization + nonlinearity" preprocessor. This improves regularization, since BN now normalizes the signal *before* the convolution processes it.

---

## Key Results

### CIFAR-10/CIFAR-100

| Model                    | CIFAR-10 Error (%) | CIFAR-100 Error (%) |
|--------------------------|---------------------|----------------------|
| ResNet-164 (original)    | 5.46                | 24.33                |
| **ResNet-164 (pre-act)** | **5.46**            | **22.71**            |
| ResNet-1001 (original)   | 7.61                | 33.47 (divergent early training) |
| **ResNet-1001 (pre-act)**| **4.92**            | **22.68**            |

The pre-activation design particularly shines at extreme depth: the original ResNet-1001 exhibits optimization problems, while the pre-activation variant trains smoothly and achieves the best results.

### ImageNet

| Model                        | Top-1 Error (%) | Top-5 Error (%) |
|------------------------------|-----------------|-----------------|
| ResNet-200 (original)        | Difficult training (overfitting) | -- |
| **ResNet-200 (pre-act)**     | **20.7**        | **5.3**         |

The original ResNet-200 showed signs of overfitting on ImageNet; the pre-activation variant resolved this and achieved lower error than the original ResNet-152.

---

## Impact & Legacy

- The **pre-activation residual block** (sometimes called "ResNet v2") became the recommended default in many implementations and subsequent works.
- The theoretical analysis of gradient flow through identity shortcuts influenced the design of many subsequent architectures, including DenseNet (which takes the idea further with dense connections).
- Demonstrated that architectural choices that preserve clean gradient paths are more important than adding complexity to the shortcut or activation structure.
- The insight that BN before convolution provides better regularization influenced subsequent normalization research (Group Normalization, Layer Normalization placement in Transformers).

---

## Key Takeaways

1. **Clean identity shortcuts are critical.** Any modification (scaling, gating, 1x1 convolutions) to the skip path degrades performance, especially in very deep networks, because it disrupts the direct gradient path.
2. The unrolled representation $\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}_i$ shows that ResNets behave like an ensemble of paths of different lengths, and the identity skip ensures that the gradient term always contains $1$ plus a residual term.
3. **Pre-activation** (BN-ReLU before Conv) is strictly better than post-activation for deep ResNets because it preserves the clean identity path through the addition.
4. The pre-activation design becomes increasingly important as network depth increases; it resolves optimization instabilities observed in 1,000+ layer networks.
5. Good architectural design can substitute for or complement regularization: the pre-activation ResNet-200 avoids the overfitting issues of the original ResNet-200 without any additional regularization.
