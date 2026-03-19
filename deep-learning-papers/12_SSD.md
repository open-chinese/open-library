# SSD: Single Shot MultiBox Detector

**Authors:** Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
**Year:** 2016
**Venue:** ECCV 2016
**Paper:** [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)

---

## Key Contributions

- Proposed **SSD**, a single-shot object detector that eliminates the need for region proposals while matching or exceeding the accuracy of two-stage detectors.
- Introduced **multi-scale feature maps** for detection -- predicting objects at multiple resolutions from a single forward pass.
- Used **default (anchor) boxes** of varying aspect ratios at each feature map location, combined with learned offsets and class scores.
- Achieved real-time detection speed (59 FPS for SSD300) with state-of-the-art accuracy on PASCAL VOC and COCO.

---

## Background & Motivation

Before SSD, the dominant paradigm for accurate object detection was the two-stage approach (e.g., Faster R-CNN): first generate region proposals, then classify each proposal. This was accurate but slow. YOLO showed that single-shot detection was fast but sacrificed accuracy, especially for small objects. SSD aimed to combine the speed of single-shot detection with the accuracy of multi-scale feature extraction from two-stage methods.

---

## Method / Architecture

### Overall Architecture

SSD is built on top of a base classification network (VGG-16) truncated before classification layers, with additional convolutional feature layers of progressively decreasing spatial resolution appended.

Detection heads are attached to **multiple feature maps** at different scales:

| Feature Map Source | Size (SSD300) | Size (SSD512) |
|---|---|---|
| Conv4_3 | $38 \times 38$ | $64 \times 64$ |
| Conv7 (FC7) | $19 \times 19$ | $32 \times 32$ |
| Conv8_2 | $10 \times 10$ | $16 \times 16$ |
| Conv9_2 | $5 \times 5$ | $8 \times 8$ |
| Conv10_2 | $3 \times 3$ | $4 \times 4$ |
| Conv11_2 | $1 \times 1$ | $2 \times 2$ |

### Default Boxes and Predictions

At each location in a feature map of size $m \times n$ with $k$ default boxes, the network predicts:
- **4 offsets** per default box: $(\Delta cx, \Delta cy, \Delta w, \Delta h)$
- **$c$ class scores** per default box (including background)

Total predictions per feature map: $m \times n \times k \times (4 + c)$

The scale of default boxes for the $k$-th feature map is:

$$s_k = s_{\min} + \frac{s_{\max} - s_{\min}}{m - 1}(k - 1)$$

where $s_{\min} = 0.2$ and $s_{\max} = 0.9$, and $m$ is the number of feature maps used for detection.

Aspect ratios are chosen from: $a_r \in \{1, 2, 3, \frac{1}{2}, \frac{1}{3}\}$

Default box width and height:

$$w_k^a = s_k \sqrt{a_r}, \quad h_k^a = \frac{s_k}{\sqrt{a_r}}$$

For aspect ratio $a_r = 1$, an additional default box with scale $s_k' = \sqrt{s_k \cdot s_{k+1}}$ is added.

### Matching Strategy

Each ground-truth box is matched to the default box with highest Jaccard overlap (IoU). Additionally, any default box with IoU > 0.5 with a ground-truth is also matched (allowing multiple default boxes per object).

### Training Objective

The overall loss is a weighted sum of localization (loc) and confidence (conf) losses:

$$L(x, c, l, g) = \frac{1}{N}\left(L_{\text{conf}}(x, c) + \alpha \, L_{\text{loc}}(x, l, g)\right)$$

where $N$ is the number of matched default boxes, $\alpha = 1$ (set by cross-validation).

**Localization loss** (Smooth L1 over encoded offsets):

$$L_{\text{loc}}(x, l, g) = \sum_{i \in \text{Pos}} \sum_{m \in \{cx, cy, w, h\}} x_{ij}^k \, \text{smooth}_{L1}(l_i^m - \hat{g}_j^m)$$

The encoded ground truth offsets:

$$\hat{g}_j^{cx} = \frac{g_j^{cx} - d_i^{cx}}{d_i^w}, \quad \hat{g}_j^{cy} = \frac{g_j^{cy} - d_i^{cy}}{d_i^h}$$

$$\hat{g}_j^{w} = \log\frac{g_j^w}{d_i^w}, \quad \hat{g}_j^{h} = \log\frac{g_j^h}{d_i^h}$$

**Confidence loss** (softmax cross-entropy):

$$L_{\text{conf}}(x, c) = -\sum_{i \in \text{Pos}} x_{ij}^p \log(\hat{c}_i^p) - \sum_{i \in \text{Neg}} \log(\hat{c}_i^0)$$

where $\hat{c}_i^p = \frac{\exp(c_i^p)}{\sum_p \exp(c_i^p)}$.

### Hard Negative Mining

Most default boxes are negatives after matching. Instead of using all negatives, SSD sorts them by confidence loss and picks the top ones so that the negative:positive ratio is at most **3:1**. This leads to faster and more stable training.

### Data Augmentation

Each training image is randomly processed by one of:
1. Use the entire original image
2. Sample a patch with IoU of 0.1, 0.3, 0.5, 0.7, or 0.9 with objects
3. Randomly sample a patch

This aggressive augmentation is critical for SSD's performance, especially on small objects.

---

## Key Results

### PASCAL VOC 2007 Test (mAP)

| Method | Input Size | mAP | FPS |
|---|---|---|---|
| Faster R-CNN (VGG) | ~1000 | 73.2 | 7 |
| YOLO | 448 | 63.4 | 45 |
| **SSD300** | **300** | **74.3** | **59** |
| **SSD512** | **512** | **76.8** | **22** |

### PASCAL VOC 2012 Test (mAP)

| Method | mAP |
|---|---|
| Faster R-CNN | 70.4 |
| **SSD300** | **72.4** |
| **SSD512** | **74.9** |

### COCO test-dev

| Method | mAP@0.5 | mAP@[0.5:0.95] |
|---|---|---|
| Faster R-CNN | 41.5 | 21.2 |
| **SSD300** | **41.2** | **23.2** |
| **SSD512** | **46.5** | **26.8** |

### Ablation: Contribution of Each Component

| Configuration | VOC 2007 mAP |
|---|---|
| Baseline (single scale, no augmentation) | 65.5 |
| + Multi-scale feature maps | 71.6 |
| + Default boxes with multiple aspect ratios | 73.0 |
| + Atrous convolution | 73.7 |
| + Data augmentation | 74.3 |

---

## Impact & Legacy

- **Established the multi-scale feature map paradigm** for single-shot detection, which became the foundation for nearly all subsequent detectors (RetinaNet, DSSD, YOLOv3, EfficientDet).
- Showed that carefully designed default boxes and multi-scale predictions can match proposal-based methods.
- The hard negative mining strategy became a standard technique until focal loss (RetinaNet) provided a more elegant solution.
- Directly influenced Feature Pyramid Networks (FPN) and the broader idea of leveraging features at multiple resolutions.
- Remains one of the most cited object detection papers, widely deployed in industry for real-time applications.

---

## Key Takeaways

1. **Multi-scale detection** from multiple feature maps is the key insight -- large feature maps handle small objects, small feature maps handle large objects.
2. **Default (anchor) boxes** with multiple scales and aspect ratios at each location provide dense coverage of the object space.
3. **Hard negative mining** (3:1 ratio) is necessary because of the extreme class imbalance in single-shot detectors.
4. **Aggressive data augmentation** (especially random cropping) is essential for robust small object detection.
5. Single-shot detection can match two-stage detector accuracy while running at real-time speeds when the architecture properly handles multiple scales.
