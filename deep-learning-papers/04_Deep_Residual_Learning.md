# Deep Residual Learning for Image Recognition

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Year:** 2016
**Venue:** CVPR 2016 (Best Paper Award); originally arXiv December 2015
**ArXiv:** 1512.03385

---

## Key Contributions

- Introduced **Residual Networks (ResNets)** with shortcut (skip) connections that enable training of networks with over 100 layers, and demonstrated networks up to 1,202 layers deep.
- Won 1st place on ILSVRC 2015 classification (3.57% top-5 error), ILSVRC 2015 detection, ILSVRC 2015 localization, and COCO 2015 detection and segmentation.
- Showed that deeper plain networks (without skip connections) suffer from a **degradation problem** where training accuracy gets worse with increasing depth, and that residual connections solve this.
- Provided a simple, general-purpose architectural principle that has become the default design choice in virtually all subsequent deep learning architectures.

---

## Background & Motivation

By 2015, the trend was clear: deeper networks learn richer representations and achieve better accuracy. However, simply stacking more layers hit a wall:

- **Vanishing/exploding gradients** were partially addressed by batch normalization and careful initialization, but a more fundamental problem remained.
- **Degradation problem:** Adding more layers to an already "good" network caused *higher training error* (not just overfitting). A 56-layer plain network performs worse than a 20-layer one on both training and test sets.

This is counterintuitive. A deeper network should be at least as expressive as a shallower one: the extra layers could, in principle, learn identity mappings. The degradation problem suggests that optimizers struggle to find these identity mappings in practice.

**Key Insight:** If identity mappings are hard to learn with standard layers, make them the default by reformulating the layer as learning a *residual* on top of the identity.

---

## Method

### Residual Learning Framework

Instead of hoping each stack of layers learns an underlying mapping $\mathcal{H}(\mathbf{x})$, explicitly let the layers learn the residual:

$$\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$$

The output of the block then becomes:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

If the optimal transformation is close to identity, it is easier for the network to push the residual $\mathcal{F}$ toward zero than to learn an identity mapping from scratch with a stack of nonlinear layers.

### Residual Block

A basic residual block consists of:

$$\mathbf{y} = \sigma(W_2 \cdot \sigma(W_1 \cdot \mathbf{x} + b_1) + b_2 + \mathbf{x})$$

where $\sigma$ is the ReLU activation. In practice with batch normalization:

```
x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> (+x) -> ReLU
```

### Handling Dimension Mismatch

When the dimensions of $\mathbf{x}$ and $\mathcal{F}(\mathbf{x})$ differ (e.g., when changing the number of feature maps or spatial resolution), a linear projection is applied to the shortcut:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$

where $W_s$ is a $1 \times 1$ convolution that matches dimensions. The authors found that using projections only when dimensions change (and identity shortcuts otherwise) works well and is parameter-efficient.

### Bottleneck Block

For deeper networks (ResNet-50 and above), a **bottleneck** design reduces computation:

```
x -> Conv(1x1, reduce) -> BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> Conv(1x1, expand) -> BN -> (+x) -> ReLU
```

The $1 \times 1$ convolutions reduce and then restore the channel dimension. For example, a 256-channel input is reduced to 64 channels, processed by a $3 \times 3$ convolution, and then expanded back to 256 channels.

### Network Architectures

| Model      | Layers | Parameters | Top-5 Error (%) |
|------------|--------|------------|-----------------|
| VGG-19     | 19     | 144M       | 7.3             |
| ResNet-18  | 18     | 11.7M      | 27.88 (top-1)   |
| ResNet-34  | 34     | 21.8M      | 25.03 (top-1)   |
| ResNet-50  | 50     | 25.6M      | 22.85 (top-1)   |
| ResNet-101 | 101    | 44.5M      | 21.75 (top-1)   |
| ResNet-152 | 152    | 60.2M      | 21.43 (top-1)   |

### Training Details

- **Data augmentation:** Random crop from resized image, horizontal flip, per-pixel mean subtraction, color augmentation.
- **Optimization:** SGD with momentum 0.9, weight decay $10^{-4}$, mini-batch size 256.
- **Learning rate:** Starting at 0.1, divided by 10 when the error plateaus.
- **Batch normalization** after every convolution, before ReLU.
- **No dropout** is needed (BN provides regularization).

---

## Key Results

### ImageNet Classification (Single Model, Single Crop)

| Model            | Top-1 Error (%) | Top-5 Error (%) |
|------------------|-----------------|-----------------|
| VGG-16           | 28.07           | 9.33            |
| GoogLeNet        | --              | 6.67            |
| **ResNet-34**    | 24.19           | 7.40            |
| **ResNet-152**   | **21.43**       | **5.71**        |

### ImageNet Classification (Ensemble)

| Model                | Top-5 Error (%) |
|----------------------|-----------------|
| VGG ensemble         | 6.8             |
| GoogLeNet ensemble   | 6.7             |
| **ResNet ensemble**  | **3.57**        |

### Degradation Problem Demonstration

| Network Depth (Plain) | Training Error | Test Error |
|----------------------|----------------|------------|
| 20 layers            | Lower          | Lower      |
| 56 layers            | **Higher**     | **Higher** |

| Network Depth (ResNet) | Training Error | Test Error |
|------------------------|----------------|------------|
| 20 layers              | Higher         | Higher     |
| 56 layers              | **Lower**      | **Lower**  |

This table demonstrates the key finding: plain networks degrade with depth, while ResNets improve with depth.

### CIFAR-10 Results

- ResNet-110: 6.43% error
- ResNet-1202: 7.93% error (slight overfitting at extreme depth without additional regularization, but no degradation in training error)

---

## Impact & Legacy

- **ResNet is arguably the single most influential deep learning architecture paper.** As of 2024, it has over 200,000 citations.
- Skip connections became a universal design principle used in virtually every subsequent architecture: DenseNet, ResNeXt, Transformer (residual connections around attention and FFN), U-Net variants, and more.
- Enabled the practical training of networks with hundreds or thousands of layers, fundamentally changing what "deep" means in deep learning.
- The ResNet-50 backbone remains one of the most commonly used feature extractors in computer vision, even a decade later.
- Inspired the theoretical study of why skip connections help optimization, including connections to ensemble-like behavior, shattered gradients, and loss landscape smoothing.

---

## Key Takeaways

1. The **degradation problem** is not overfitting but an optimization difficulty: deep plain networks cannot easily learn identity mappings.
2. Residual connections make identity mapping the default, so the network only needs to learn the deviation from identity. This is a much easier optimization target.
3. Depth matters: with residual connections, performance consistently improves as more layers are added (up to a point).
4. Bottleneck blocks ($1 \times 1$ reduce, $3 \times 3$, $1 \times 1$ expand) make deep networks computationally feasible.
5. The simplicity and generality of the residual connection idea is its greatest strength. It requires no new hyperparameters, no new loss functions, and no changes to the training procedure.
