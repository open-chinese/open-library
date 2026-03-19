# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

**Authors:** Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
**Year:** 2016 (DeepLabv2, arXiv 2016; builds on DeepLabv1, ICLR 2015)
**Venue:** TPAMI 2017 (arXiv June 2016)
**Paper:** [arXiv:1606.00915](https://arxiv.org/abs/1606.00915)

---

## Key Contributions

- Introduced **atrous (dilated) convolution** as a core tool for semantic segmentation, allowing enlarged receptive fields without increasing parameters or reducing resolution.
- Proposed **Atrous Spatial Pyramid Pooling (ASPP)**, which captures multi-scale context by applying atrous convolutions at multiple dilation rates in parallel.
- Combined deep CNN features with a **Dense (Fully Connected) Conditional Random Field (CRF)** as a post-processing step to refine object boundaries.
- Achieved state-of-the-art results on PASCAL VOC 2012 segmentation benchmark.
- Provided a thorough analysis of the three main challenges in applying DCNNs to segmentation: reduced resolution, multi-scale objects, and localization imprecision.

---

## Background & Motivation

Three key challenges when applying classification CNNs to semantic segmentation:

1. **Reduced spatial resolution**: Repeated pooling and striding in deep networks reduce the output resolution by a factor of 32. Naive upsampling produces coarse predictions.
2. **Multi-scale objects**: Objects appear at varying scales. A single receptive field size cannot capture both local detail and global context.
3. **Imprecise localization**: Classification networks are invariant to spatial transformations, but segmentation requires precise pixel-level localization.

DeepLab addresses these with: (1) atrous convolution, (2) ASPP, and (3) CRF post-processing, respectively.

---

## Method / Architecture

### Atrous (Dilated) Convolution

Standard convolution with kernel size $k$ has a receptive field of $k$. Atrous convolution inserts $r - 1$ zeros between filter weights, effectively increasing the receptive field to $k + (k - 1)(r - 1)$ without increasing the number of parameters.

For a 1D input signal $x[i]$ and filter $w[k]$ of length $K$:

$$y[i] = \sum_{k=1}^{K} x[i + r \cdot k] \, w[k]$$

where $r$ is the **dilation rate** (atrous rate). Standard convolution is the special case $r = 1$.

For a 2D feature map with atrous rate $r$, a $3 \times 3$ kernel effectively covers a $(2r + 1) \times (2r + 1)$ region while using only 9 parameters.

### Adapting VGG / ResNet for Dense Prediction

Instead of using the full stride-32 network and upsampling:
1. Remove the last pooling and striding layers (pool4 and pool5 in VGG, or the last stride-2 convolutions in ResNet).
2. Replace subsequent convolutions with **atrous convolutions** (rate $r = 2$ for the layer after the removed pool4, $r = 4$ for layers after the removed pool5).
3. This produces an output at stride 8 instead of stride 32, preserving 16x more spatial resolution.

The output feature map is then bilinearly upsampled by $8\times$ to match the original image resolution.

### Atrous Spatial Pyramid Pooling (ASPP)

To capture objects at multiple scales, ASPP applies several parallel atrous convolutions with different rates to the same feature map:

$$\text{ASPP}(F) = [\text{Conv}_{r=6}(F) \,;\, \text{Conv}_{r=12}(F) \,;\, \text{Conv}_{r=18}(F) \,;\, \text{Conv}_{r=24}(F)]$$

where $[\cdot \,;\, \cdot]$ denotes concatenation (or the scores are fused). Each branch uses $3 \times 3$ atrous convolutions at the specified rate, capturing context at different scales.

| ASPP Branch | Rate $r$ | Effective Receptive Field |
|---|---|---|
| Branch 1 | 6 | $13 \times 13$ |
| Branch 2 | 12 | $25 \times 25$ |
| Branch 3 | 18 | $37 \times 37$ |
| Branch 4 | 24 | $49 \times 49$ |

The outputs of all branches are combined (concatenated or summed) and passed through a $1 \times 1$ convolution to produce the final class scores.

### Fully Connected CRF Post-Processing

The CNN output provides good semantic predictions but with imprecise boundaries. A fully connected (dense) CRF refines the segmentation by considering pairwise relationships between all pixels.

The CRF energy function:

$$E(\mathbf{x}) = \sum_i \theta_i(x_i) + \sum_{i < j} \theta_{ij}(x_i, x_j)$$

**Unary potential** $\theta_i$: Negative log-probability from the CNN output:

$$\theta_i(x_i) = -\log P(x_i)$$

**Pairwise potential** $\theta_{ij}$: Penalizes different labels at nearby pixels with similar appearance:

$$\theta_{ij}(x_i, x_j) = \mu(x_i, x_j) \left[ w_1 \exp\left(-\frac{|p_i - p_j|^2}{2\sigma_\alpha^2} - \frac{|I_i - I_j|^2}{2\sigma_\beta^2}\right) + w_2 \exp\left(-\frac{|p_i - p_j|^2}{2\sigma_\gamma^2}\right) \right]$$

where:
- $\mu(x_i, x_j) = 1$ if $x_i \neq x_j$ (Potts model)
- $p_i, p_j$ are pixel positions
- $I_i, I_j$ are pixel colors (RGB)
- The first kernel is an **appearance kernel** (bilateral), encouraging nearby pixels with similar colors to have the same label
- The second kernel is a **smoothness kernel**, encouraging spatial smoothness
- $\sigma_\alpha, \sigma_\beta, \sigma_\gamma, w_1, w_2$ are hyperparameters

The dense CRF is solved approximately using **mean-field inference** (typically 10 iterations), which operates on the full image and runs in $O(N)$ time per iteration using high-dimensional Gaussian filtering.

---

## Key Results

### PASCAL VOC 2012 Test (mean IoU)

| Method | mIoU |
|---|---|
| FCN-8s | 62.2 |
| DeepLabv1 (VGG-16) | 71.6 |
| **DeepLabv2 (ResNet-101)** | **79.7** |
| DeepLabv2 + COCO pretraining | **79.7** |

### Ablation: Component Contributions (VOC 2012 val)

| Configuration | mIoU |
|---|---|
| ResNet-101, no ASPP, no CRF | 68.7 |
| + ASPP (multi-scale) | 73.5 |
| + Dense CRF | **75.7** |

### Ablation: ASPP Design (VOC 2012 val)

| Method | mIoU |
|---|---|
| Large Field-of-View (single rate $r=12$) | 72.4 |
| **ASPP (rates 6, 12, 18, 24)** | **73.5** |

### Comparison of CRF Effect

| Model | w/o CRF | w/ CRF | Improvement |
|---|---|---|---|
| DeepLabv1 (VGG-16) | 68.7 | 71.6 | +2.9 |
| DeepLabv2 (ResNet-101) | 73.6 | 77.7 | +4.1 |

### Inference Speed

- DeepLabv2 (ResNet-101): ~8 FPS (excluding CRF)
- Dense CRF: ~0.5 seconds per image

---

## Impact & Legacy

- **Atrous (dilated) convolution** became a fundamental tool in dense prediction, adopted in numerous architectures beyond segmentation (WaveNet, PointNet++, etc.).
- **ASPP** inspired multi-scale context aggregation modules in subsequent work (PSPNet's Pyramid Pooling, DeepLabv3's improved ASPP, etc.).
- The DeepLab series continued to evolve: DeepLabv3 (improved ASPP, removed CRF), DeepLabv3+ (encoder-decoder with atrous convolution), becoming one of the most influential families in segmentation.
- Showed that **CRF post-processing** could substantially improve CNN-based segmentation, though later versions (DeepLabv3+) achieved strong results without it.
- Defined a clean framework for thinking about the three challenges (resolution, scale, localization) that remains useful for understanding segmentation methods.

---

## Key Takeaways

1. **Atrous convolution** enlarges the receptive field without losing resolution or adding parameters -- a strictly better alternative to large kernels or aggressive pooling for dense prediction.
2. **Multi-scale context** (via ASPP) is essential for segmenting objects of varying sizes. Parallel branches with different dilation rates capture complementary context scales.
3. **Dense CRF post-processing** provides significant boundary refinement by leveraging low-level image structure (color, position), compensating for the CNN's spatial imprecision.
4. **Controlling output stride** (8 vs. 16 vs. 32) is a critical design decision: lower stride preserves more spatial detail but increases computation.
5. The combination of a strong backbone (ResNet-101), multi-scale features (ASPP), and structured prediction (CRF) addresses complementary weaknesses, yielding a system that is more than the sum of its parts.
