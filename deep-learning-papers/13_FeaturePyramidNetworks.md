# Feature Pyramid Networks for Object Detection

**Authors:** Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
**Year:** 2016
**Venue:** CVPR 2017 (arXiv December 2016)
**Paper:** [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

---

## Key Contributions

- Proposed **Feature Pyramid Networks (FPN)**, a generic architecture for building multi-scale feature representations with strong semantics at all levels.
- Introduced a **top-down pathway with lateral connections** that combines low-resolution, semantically strong features with high-resolution, semantically weak features.
- Achieved state-of-the-art results on COCO detection using the basic Faster R-CNN framework, without bells and whistles.
- FPN is architecture-agnostic and can be applied to any backbone CNN, becoming the **de facto standard** feature extractor for detection.

---

## Background & Motivation

Object detection must handle objects at vastly different scales. Previous approaches dealt with this in several ways:

1. **Image pyramids**: Run a detector on multiple resized versions of the image. Accurate but extremely slow and memory-intensive.
2. **Single feature map**: Use only the last layer (e.g., Faster R-CNN on conv5). Fast but poor for small objects because fine spatial detail is lost.
3. **Multi-scale features (SSD-style)**: Predict from multiple layers independently. Fast but lower layers lack strong semantic information.

FPN combines the best of all approaches: it builds a pyramid of features that are both **semantically strong** (via top-down enrichment) and **high-resolution** (via lateral connections from the bottom-up pathway).

---

## Method / Architecture

### Architecture Overview

FPN consists of three components:

1. **Bottom-Up Pathway**: Standard feedforward computation of the backbone CNN (e.g., ResNet). Produces feature maps at multiple scales (one per stage: $\{C_2, C_3, C_4, C_5\}$ for ResNet, with strides $\{4, 8, 16, 32\}$).

2. **Top-Down Pathway**: Starts from the coarsest feature map and progressively upsamples (by $2\times$ nearest neighbor) to produce higher-resolution maps.

3. **Lateral Connections**: $1 \times 1$ convolutions on bottom-up feature maps to reduce channel dimensionality to $d = 256$, then element-wise addition with the corresponding upsampled top-down map.

A final $3 \times 3$ convolution is applied to each merged map to reduce aliasing from upsampling, producing the final feature pyramid $\{P_2, P_3, P_4, P_5\}$.

### Formal Description

For each pyramid level $l$:

$$P_l = \text{Conv}_{3 \times 3}\left(\text{Upsample}_{2\times}(P_{l+1}) + \text{Conv}_{1 \times 1}(C_l)\right)$$

with $P_5 = \text{Conv}_{1 \times 1}(C_5)$ as the starting point.

All pyramid levels have the same channel dimension $d = 256$.

### FPN for Region Proposal Network (RPN)

In standard Faster R-CNN, the RPN operates on a single-scale feature map. With FPN, the RPN is applied independently to each pyramid level. An anchor of area $A$ pixels is assigned to pyramid level:

$$l = \lfloor l_0 + \log_2(\sqrt{A} / 224) \rfloor$$

where $l_0 = 4$ (i.e., an anchor of $224^2$ pixels maps to $P_4$). Each level uses a single anchor scale with aspect ratios $\{1:2, 1:1, 2:1\}$, giving 3 anchors per level and 15 anchors total across the pyramid.

### FPN for Fast R-CNN (Detection Head)

For the second stage (box classification/regression), each RoI is assigned to a pyramid level based on its area:

$$k = \lfloor k_0 + \log_2(\sqrt{wh} / 224) \rfloor$$

where $k_0 = 4$, and $w, h$ are the RoI width and height. Smaller RoIs are mapped to finer-resolution levels, ensuring that small objects are processed on high-resolution feature maps.

RoI pooling is applied to the appropriate pyramid level, followed by the detection head (two fully connected layers).

---

## Key Results

### COCO Detection: FPN + Faster R-CNN (ResNet-101)

| Method | AP | AP$_{50}$ | AP$_{75}$ | AP$_S$ | AP$_M$ | AP$_L$ |
|---|---|---|---|---|---|---|
| Faster R-CNN (conv4, single scale) | 30.4 | 52.4 | 31.2 | 12.4 | 33.9 | 45.4 |
| Faster R-CNN (conv5, single scale) | 31.8 | 53.3 | 33.9 | 13.1 | 35.2 | 47.1 |
| **FPN + Faster R-CNN** | **36.2** | **59.1** | **39.0** | **18.2** | **39.0** | **48.2** |

### Ablation: Component Analysis (ResNet-50, COCO minival)

| Top-Down? | Lateral? | AP |
|---|---|---|
| No | No (single C5) | 31.0 |
| Yes | No | 33.4 |
| Yes | Yes (FPN) | **33.9** |

### RPN Proposals (COCO, AR@1000 with ResNet-101)

| Method | AR$_S$ | AR$_M$ | AR$_L$ | AR |
|---|---|---|---|---|
| RPN on conv5 | 30.1 | 52.2 | 62.1 | 44.6 |
| **RPN on FPN** | **44.2** | **60.0** | **64.2** | **56.3** |

The improvement for small objects (AR$_S$: 30.1 to 44.2) is particularly dramatic.

### Comparison with State-of-the-Art (COCO test-dev)

| Method | AP | AP$_{50}$ | AP$_{75}$ |
|---|---|---|---|
| Faster R-CNN+++ (ResNet-101, multi-scale test) | 34.9 | 55.7 | 37.4 |
| **FPN + Faster R-CNN (ResNet-101)** | **36.2** | **59.1** | **39.0** |

FPN achieves better results with a single-scale test, no bells and whistles.

---

## Impact & Legacy

- **Became the standard feature extractor** for object detection, instance segmentation, and many other dense prediction tasks.
- Adopted in virtually all major detection frameworks: Mask R-CNN, RetinaNet, FCOS, DETR variants, and more.
- The top-down + lateral connection design influenced architectures well beyond detection: panoptic segmentation, keypoint detection, depth estimation.
- Spawned numerous follow-ups: PANet (bottom-up augmentation), NAS-FPN (neural architecture search for FPN), BiFPN (EfficientDet), etc.
- One of the most influential architectural components in modern computer vision, with thousands of citations.

---

## Key Takeaways

1. **Semantic gap across scales** is the core problem: low-level features have spatial detail but weak semantics; high-level features have strong semantics but low resolution. FPN bridges this gap.
2. **Top-down enrichment with lateral connections** is a simple yet powerful paradigm. The $1 \times 1$ lateral convolution + element-wise addition is computationally cheap but highly effective.
3. **Assigning anchors/RoIs to pyramid levels by size** ensures that each object is processed on an appropriately-resolved feature map.
4. **Architecture-agnostic design** is key to FPN's wide adoption -- it works with any backbone and any detection framework.
5. The single biggest gain is for **small objects**, which benefit most from high-resolution, semantically enriched features.
