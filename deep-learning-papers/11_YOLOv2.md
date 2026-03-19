# YOLO9000: Better, Faster, Stronger (YOLOv2)

**Authors:** Joseph Redmon, Ali Farhadi
**Year:** 2016
**Venue:** CVPR 2017 (arXiv December 2016)
**Paper:** [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)

---

## Key Contributions

- Introduced **YOLOv2**, a significantly improved version of YOLO with better accuracy and speed for object detection.
- Proposed a series of practical training tricks (batch normalization, high-resolution classifier, anchor boxes, etc.) that collectively boosted performance.
- Introduced **YOLO9000**, a model capable of detecting over 9,000 object categories by jointly training on detection and classification data.
- Designed a **WordTree** hierarchy to merge datasets with different label granularities (ImageNet + COCO).
- Introduced the **Darknet-19** backbone architecture optimized for detection speed.

---

## Background & Motivation

The original YOLO (You Only Look Once) was fast but lagged behind region-based detectors (Faster R-CNN) in accuracy, especially for small objects. Meanwhile, models like SSD showed that single-shot detectors could be both fast and accurate. YOLOv2 aimed to close this accuracy gap while maintaining or improving speed, and to scale detection to thousands of categories using a novel joint training strategy.

---

## Method / Architecture

### Improvement Strategy: "Better"

YOLOv2 applies a series of incremental improvements to the original YOLO:

| Improvement | Effect |
|---|---|
| **Batch Normalization** | Added to all convolutional layers; improved convergence, +2% mAP |
| **High-Resolution Classifier** | Fine-tune classifier at $448 \times 448$ before detection training |
| **Convolutional Anchor Boxes** | Replace fully connected layers with anchor-based predictions |
| **Dimension Clusters** | Use k-means on training boxes to find good anchor priors |
| **Direct Location Prediction** | Constrain predictions to the grid cell, improving stability |
| **Fine-Grained Features** | Passthrough layer brings features from earlier layers ($26 \times 26$) |
| **Multi-Scale Training** | Randomly change input size every 10 batches (320 to 608) |

### Anchor Box Prediction

For each anchor box, the network predicts 5 values: $t_x, t_y, t_w, t_h$, and objectness $t_o$. The bounding box coordinates are computed as:

$$b_x = \sigma(t_x) + c_x$$

$$b_y = \sigma(t_y) + c_y$$

$$b_w = p_w \, e^{t_w}$$

$$b_h = p_h \, e^{t_h}$$

$$\Pr(\text{object}) \cdot \text{IoU}(b, \text{object}) = \sigma(t_o)$$

where $(c_x, c_y)$ is the top-left corner of the grid cell and $(p_w, p_h)$ are the anchor box dimensions. The sigmoid $\sigma$ constrains the center to remain within the cell.

### Dimension Clusters (k-means on Boxes)

Instead of hand-picking anchor box sizes, YOLOv2 runs k-means clustering on training bounding boxes using a custom distance:

$$d(\text{box}, \text{centroid}) = 1 - \text{IoU}(\text{box}, \text{centroid})$$

With $k = 5$ clusters, this achieves a good balance between model complexity and IoU coverage.

### Darknet-19 Backbone ("Faster")

A new 19-layer convolutional backbone:

| Layer Type | Filters | Output Size |
|---|---|---|
| Conv $3 \times 3$ | 32 | $224 \times 224$ |
| Maxpool $2 \times 2$ | -- | $112 \times 112$ |
| Conv $3 \times 3$ | 64 | $112 \times 112$ |
| Maxpool $2 \times 2$ | -- | $56 \times 56$ |
| Conv $3 \times 3$ / $1 \times 1$ / $3 \times 3$ | 128/64/128 | $56 \times 56$ |
| Maxpool $2 \times 2$ | -- | $28 \times 28$ |
| Conv $3 \times 3$ / $1 \times 1$ / $3 \times 3$ | 256/128/256 | $28 \times 28$ |
| Maxpool $2 \times 2$ | -- | $14 \times 14$ |
| Conv $3 \times 3$ / $1 \times 1$ (x3) | 512/256/512 | $14 \times 14$ |
| Maxpool $2 \times 2$ | -- | $7 \times 7$ |
| Conv $3 \times 3$ / $1 \times 1$ (x2), $3 \times 3$ | 1024/512/1024 | $7 \times 7$ |

Requires only ~5.58 billion FLOPs (vs. VGG-16's ~30.69 billion).

### YOLO9000: Joint Training ("Stronger")

To detect 9,000+ categories, YOLOv2 jointly trains on ImageNet classification and COCO detection data using a **WordTree** -- a hierarchical tree merging ImageNet and COCO labels based on WordNet.

During training:
- Detection images: backpropagate full detection loss.
- Classification images: backpropagate only classification loss using hierarchical softmax over the WordTree.

The conditional probability at each node:

$$\Pr(\text{class}) = \prod_{i=0}^{n} \Pr(C_i \mid \text{parent}(C_i))$$

where each $\Pr(C_i \mid \text{parent}(C_i))$ is a softmax over siblings at that level.

---

## Key Results

### PASCAL VOC 2007 (mAP)

| Method | mAP | FPS |
|---|---|---|
| Fast R-CNN | 70.0 | 0.5 |
| Faster R-CNN (VGG) | 73.2 | 7 |
| SSD300 | 74.3 | 46 |
| YOLOv1 | 63.4 | 45 |
| **YOLOv2 (544)** | **78.6** | **40** |

### COCO test-dev

| Method | mAP@0.5 | mAP@[0.5:0.95] |
|---|---|---|
| Faster R-CNN | 41.5 | 21.2 |
| SSD500 | 46.5 | 26.8 |
| **YOLOv2** | **44.0** | **21.6** |

- YOLOv2 runs at **40-90+ FPS** depending on input resolution, significantly faster than two-stage detectors.
- YOLO9000 detects 9,418 categories and achieves 19.7 mAP on ImageNet detection task (for categories not in COCO).

---

## Impact & Legacy

- Demonstrated that systematic, incremental engineering improvements can yield large cumulative gains -- a practical blueprint for applied deep learning.
- The anchor box clustering technique became standard practice in later detectors.
- Multi-scale training was widely adopted for robustness across object sizes.
- YOLO9000's joint training idea influenced subsequent work on open-vocabulary and large-vocabulary detection.
- Led directly to YOLOv3, YOLOv4, and the broader YOLO family that remains dominant in real-time detection.

---

## Key Takeaways

1. **Batch normalization, anchor priors, and multi-scale training** are individually small but collectively transformative improvements.
2. **Constraining location predictions** with sigmoid activation stabilizes training and improves convergence.
3. **Data-driven anchor selection** (k-means on IoU) outperforms hand-designed anchors.
4. **Hierarchical classification** via WordTree enables scaling detection to thousands of categories by leveraging classification-only data.
5. Real-time detection (~40+ FPS) and strong accuracy are not mutually exclusive when the architecture is carefully designed.
