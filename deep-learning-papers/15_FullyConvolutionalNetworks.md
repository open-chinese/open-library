# Fully Convolutional Networks for Semantic Segmentation

**Authors:** Jonathan Long, Evan Shelhamer, Trevor Darrell
**Year:** 2015 (CVPR 2015), major impact and widespread adoption in 2016
**Venue:** CVPR 2015 (Best Paper Honorable Mention)
**Paper:** [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)

---

## Key Contributions

- Introduced **Fully Convolutional Networks (FCN)** for dense pixel-wise prediction, adapting classification CNNs (AlexNet, VGG, GoogLeNet) into segmentation networks.
- Showed that **replacing fully connected layers with $1 \times 1$ convolutions** allows networks to accept arbitrary input sizes and produce spatial output maps.
- Proposed **skip connections** that combine deep, coarse semantic information with shallow, fine appearance information for improved segmentation boundaries.
- Defined the **FCN-8s, FCN-16s, FCN-32s** variants with progressively finer predictions.
- Established end-to-end training for semantic segmentation, replacing patch-based or CRF-dependent approaches.

---

## Background & Motivation

Before FCN, semantic segmentation approaches were fragmented:
- **Patch-based classification**: Classify each pixel by extracting a patch around it and running a CNN. Extremely slow due to redundant computation.
- **Multi-scale / superpixel methods**: Use hand-crafted features or shallow classifiers on superpixels.
- **CRF-based methods**: Use deep features as unary potentials in a Conditional Random Field. Effective but complex and not end-to-end.

The key insight of FCN is that classification CNNs already learn hierarchical spatial features -- they just need to be adapted to produce dense predictions instead of a single class label.

---

## Method / Architecture

### From Classification to Dense Prediction

Any classification CNN can be "convolutionalized" by replacing fully connected layers with convolutional layers:

- A fully connected layer with $n$ inputs and $m$ outputs is equivalent to a $1 \times 1$ convolution with $m$ filters when the spatial dimensions are $1 \times 1$.
- For larger spatial inputs, the $1 \times 1$ convolution produces a spatial map of class scores (a **heatmap**).

For an input of size $H \times W$, the convolutionalized network produces an output of size $\frac{H}{s} \times \frac{W}{s} \times C$ where $s$ is the total stride (e.g., 32 for VGG-16) and $C$ is the number of classes.

### Upsampling via Deconvolution

The coarse output (stride 32) is upsampled back to the original resolution using **transposed convolutions** (also called deconvolutions or fractionally-strided convolutions):

$$y = W^T x$$

where $W^T$ is the transposed convolution kernel. These are initialized as bilinear interpolation filters and can be learned end-to-end.

### Skip Architecture

The key architectural innovation is combining predictions from multiple layers to improve spatial precision:

#### FCN-32s (Baseline)
- Direct $32\times$ upsampling from the final prediction layer (pool5/conv7).
- Produces coarse segmentations.

#### FCN-16s
- Fuses predictions from pool5 ($32\times$ upsampled by $2\times$) with predictions from pool4 (stride 16).
- Combined prediction is upsampled $16\times$ to full resolution.

$$\text{score}_{16s} = \text{Upsample}_{2\times}(\text{score}_{\text{pool5}}) + \text{score}_{\text{pool4}}$$

#### FCN-8s
- Further fuses with pool3 (stride 8).

$$\text{score}_{8s} = \text{Upsample}_{2\times}\left(\text{Upsample}_{2\times}(\text{score}_{\text{pool5}}) + \text{score}_{\text{pool4}}\right) + \text{score}_{\text{pool3}}$$

Each "score" layer is produced by a $1 \times 1$ convolution over the feature map, mapping it to $C$ class channels.

### Training Details

- **Loss**: Per-pixel multinomial cross-entropy, summed over all pixels:

$$L = -\frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{C} y_{ij}^c \log \hat{y}_{ij}^c$$

where $y_{ij}^c$ is the ground truth one-hot label and $\hat{y}_{ij}^c$ is the predicted probability for class $c$ at pixel $(i,j)$.

- **Initialization**: Weights transferred from VGG-16 pretrained on ImageNet. New layers initialized randomly.
- **Optimizer**: SGD with momentum 0.9, learning rate $10^{-3}$ (divided by 10 for the bias), weight decay $5 \times 10^{-4}$.
- **No CRF post-processing** needed (though it can still help).

---

## Key Results

### PASCAL VOC 2011 Segmentation (mean IoU)

| Method | Mean IoU |
|---|---|
| SDS (Hariharan et al., 2014) | 51.6 |
| FCN-32s (VGG-16) | 59.4 |
| FCN-16s (VGG-16) | 62.4 |
| **FCN-8s (VGG-16)** | **62.7** |

### PASCAL VOC 2012 Test (mean IoU)

| Method | Mean IoU |
|---|---|
| Previous SOTA (non-FCN) | ~50 |
| **FCN-8s** | **62.2** |

### Ablation: Effect of Skip Connections

| Model | Mean IoU | Improvement |
|---|---|---|
| FCN-32s | 59.4 | baseline |
| FCN-16s | 62.4 | +3.0 |
| FCN-8s | 62.7 | +0.3 |

The largest gain comes from the first skip (pool4). Adding pool3 gives diminishing returns, suggesting that very fine edges are hard to recover from deep features alone.

### Inference Speed

| Method | Time per Image |
|---|---|
| Patch-based CNN | minutes |
| **FCN-8s (VGG-16)** | **~175 ms** (forward pass) |

### Cross-Dataset Results

| Dataset | Metric | FCN-8s |
|---|---|---|
| NYUDv2 (RGBD) | Mean IoU | 34.0 |
| SIFT Flow | Per-pixel accuracy | 85.2 |
| PASCAL Context | Mean IoU | 35.1 |

---

## Impact & Legacy

- **Founded the field of deep learning for semantic segmentation.** Virtually every subsequent segmentation method (U-Net, DeepLab, PSPNet, SegNet, etc.) uses the FCN paradigm.
- **Introduced the encoder-decoder pattern** for dense prediction, where a downsampling encoder is followed by an upsampling decoder. This became the dominant architecture family.
- **Skip connections** for combining multi-scale features led directly to FPN, U-Net, and related architectures.
- Demonstrated that **transfer learning from classification** to dense prediction is highly effective, establishing the pretrained backbone + task-specific head pattern.
- **Transposed convolutions** for learned upsampling became a standard component in generative models (GANs) and other dense prediction tasks.
- Over 25,000 citations, one of the most influential papers in computer vision.

---

## Key Takeaways

1. **Fully convolutional design** -- replacing FC layers with $1 \times 1$ convolutions -- enables dense prediction for inputs of any size.
2. **Skip connections** that merge deep semantic features with shallow spatial features are crucial for recovering fine-grained boundaries.
3. **Transfer learning** from classification to segmentation works remarkably well; the same features that classify objects can localize them at the pixel level.
4. **End-to-end training** with per-pixel loss is simpler and more effective than pipeline approaches with separate proposal, feature extraction, and classification stages.
5. The **diminishing returns of skip connections** (FCN-16s >> FCN-32s, but FCN-8s is only slightly better than FCN-16s) suggests a practical trade-off between complexity and accuracy.
