# SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and <0.5MB Model Size

**Authors:** Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
**Year:** 2016
**Venue:** arXiv (February 2016); ICLR 2017 (workshop)
**ArXiv:** 1602.07360

---

## Key Contributions

- Proposed **SqueezeNet**, a compact CNN architecture that achieves AlexNet-level accuracy on ImageNet with **50x fewer parameters** (1.2M vs. 60M).
- Introduced the **Fire module**, a building block consisting of a "squeeze" layer ($1 \times 1$ convolutions) followed by an "expand" layer (mix of $1 \times 1$ and $3 \times 3$ convolutions).
- When combined with **Deep Compression** (Han et al., 2016), SqueezeNet achieves a model size of **0.47 MB**, which is **510x smaller** than AlexNet (240 MB).
- Articulated a set of **architectural design strategies** for achieving high accuracy with minimal parameter counts.
- Demonstrated that small models are not just useful for deployment but also easier to distribute, update over networks, and deploy on FPGAs and embedded devices.

---

## Background & Motivation

By 2016, state-of-the-art CNNs (VGGNet with 138M parameters, GoogLeNet with ~6.8M, ResNet with 25-60M) were powerful but large. Smaller models are desirable for several reasons:

1. **Distributed training:** Smaller models require less communication bandwidth between servers.
2. **Model export to clients:** Smaller models are easier to push via over-the-air updates (e.g., autonomous vehicles).
3. **FPGA and embedded deployment:** FPGAs typically have limited on-chip memory ($\leq$ 10 MB), so small models can fit entirely on-chip, avoiding costly off-chip DRAM access.

The key question: How small can a CNN be while maintaining competitive accuracy?

---

## Method

### Architectural Design Strategies

The authors propose three strategies:

**Strategy 1: Replace $3 \times 3$ filters with $1 \times 1$ filters.** A $1 \times 1$ filter has $9 \times$ fewer parameters than a $3 \times 3$ filter.

**Strategy 2: Decrease the number of input channels to $3 \times 3$ filters.** Even when $3 \times 3$ filters are needed, reducing their input channels via a "squeeze" layer reduces total parameters. For a $3 \times 3$ convolution layer with $c_{\text{in}}$ input channels and $c_{\text{out}}$ output channels:

$$\text{Parameters} = 3 \times 3 \times c_{\text{in}} \times c_{\text{out}} = 9 \cdot c_{\text{in}} \cdot c_{\text{out}}$$

Reducing $c_{\text{in}}$ directly reduces parameters.

**Strategy 3: Downsample late in the network** so that convolutional layers have large activation maps. Large activation maps (from delayed pooling) tend to yield higher classification accuracy.

### The Fire Module

The Fire module is the core building block of SqueezeNet:

```
Input (H x W x C_in)
  |
  v
Squeeze Layer: s_{1x1} filters of size 1x1  -->  (H x W x s_{1x1})
  |
  v
Expand Layer:  e_{1x1} filters of 1x1  +  e_{3x3} filters of 3x3  -->  concat  -->  (H x W x (e_{1x1} + e_{3x3}))
```

The key constraint is:

$$s_{1 \times 1} < (e_{1 \times 1} + e_{3 \times 3})$$

This ensures the squeeze layer acts as a bottleneck, reducing the dimensionality before the expand layer.

Total parameters for one Fire module:

$$\text{Params} = s_{1 \times 1} \cdot C_{\text{in}} + e_{1 \times 1} \cdot s_{1 \times 1} + 9 \cdot e_{3 \times 3} \cdot s_{1 \times 1}$$

where the first term is the squeeze layer, the second is the $1 \times 1$ expand, and the third is the $3 \times 3$ expand.

### SqueezeNet Architecture

| Layer / Module | Output Size | Details |
|---------------|------------|---------|
| Input         | 224 x 224 x 3 | -- |
| Conv1         | 111 x 111 x 96 | 96 filters, 7x7, stride 2 |
| MaxPool1      | 55 x 55 x 96 | 3x3, stride 2 |
| Fire2         | 55 x 55 x 128 | $s_{1\times1}=16$, $e_{1\times1}=64$, $e_{3\times3}=64$ |
| Fire3         | 55 x 55 x 128 | $s_{1\times1}=16$, $e_{1\times1}=64$, $e_{3\times3}=64$ |
| Fire4         | 55 x 55 x 256 | $s_{1\times1}=32$, $e_{1\times1}=128$, $e_{3\times3}=128$ |
| MaxPool4      | 27 x 27 x 256 | 3x3, stride 2 |
| Fire5         | 27 x 27 x 256 | $s_{1\times1}=32$, $e_{1\times1}=128$, $e_{3\times3}=128$ |
| Fire6         | 27 x 27 x 384 | $s_{1\times1}=48$, $e_{1\times1}=192$, $e_{3\times3}=192$ |
| Fire7         | 27 x 27 x 384 | $s_{1\times1}=48$, $e_{1\times1}=192$, $e_{3\times3}=192$ |
| Fire8         | 27 x 27 x 512 | $s_{1\times1}=64$, $e_{1\times1}=256$, $e_{3\times3}=256$ |
| MaxPool8      | 13 x 13 x 512 | 3x3, stride 2 |
| Fire9         | 13 x 13 x 512 | $s_{1\times1}=64$, $e_{1\times1}=256$, $e_{3\times3}=256$ |
| Conv10        | 13 x 13 x 1000 | 1000 filters, 1x1 |
| AvgPool10     | 1 x 1 x 1000 | Global average pooling |
| Softmax       | 1000 | -- |

**Total parameters:** ~1.24 million.

### Variants with Bypass Connections

The authors also explore adding skip connections (inspired by ResNets):

| Variant | Description | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|---------|-------------|--------------------|--------------------|
| SqueezeNet | No bypass | 57.5 | 80.3 |
| SqueezeNet + simple bypass | Skip connections where dimensions match | 60.4 | 82.5 |
| SqueezeNet + complex bypass | 1x1 conv on skip when dimensions differ | 58.8 | 82.0 |

Simple bypass connections (identity shortcuts between Fire modules with matching dimensions) provide a notable improvement.

---

## Key Results

### Model Size Comparison

| Model                  | Top-1 Acc. (%) | Top-5 Acc. (%) | Model Size | Compression vs. AlexNet |
|------------------------|---------------|---------------|------------|------------------------|
| AlexNet                | 57.2          | 80.3          | 240 MB     | 1x                     |
| AlexNet + Deep Comp.   | 57.2          | 80.3          | 6.9 MB     | 35x                    |
| **SqueezeNet**         | **57.5**      | **80.3**      | **4.8 MB** | **50x**                |
| **SqueezeNet + Deep Comp.** | **57.5** | **80.3**    | **0.47 MB**| **510x**               |

### Comparison with Compact Models

| Model              | Parameters | Top-5 Accuracy (%) |
|--------------------|-----------|---------------------|
| AlexNet            | 60.0M     | 80.3               |
| **SqueezeNet**     | **1.24M** | **80.3**           |
| GoogLeNet (v1)     | 6.8M      | ~88.9              |
| Network in Network | 7.6M      | ~83               |

### Design Space Exploration

The authors systematically varied the squeeze ratio ($SR$), the proportion of $3 \times 3$ filters in the expand layer ($pct_{3 \times 3}$), and the base number of expand filters:

| $SR$ | $pct_{3 \times 3}$ | Model Size (MB) | Top-5 Accuracy (%) |
|------|---------------------|-----------------|---------------------|
| 0.125| 50%                 | 0.5             | 73.1               |
| 0.50 | 50%                 | 4.8             | **80.3**           |
| 0.75 | 50%                 | 7.7             | 80.3               |
| 0.50 | 1%                  | 2.6             | 77.0               |
| 0.50 | 99%                 | 7.0             | 78.8               |

A squeeze ratio of 0.50 and a 50/50 split between $1 \times 1$ and $3 \times 3$ in the expand layer provides the best trade-off.

---

## Impact & Legacy

- SqueezeNet was one of the first papers to explicitly target model compression through architecture design (rather than post-hoc compression of large models), pioneering the "efficient architecture" research direction.
- Directly influenced the development of **MobileNets** (Howard et al., 2017), **ShuffleNet** (Zhang et al., 2018), and the entire family of efficient mobile architectures.
- The Fire module's squeeze-then-expand pattern is conceptually related to the bottleneck design in ResNets and the inverted residual in MobileNet-v2.
- Demonstrated that the combination of architectural efficiency and post-training compression (quantization, pruning, Huffman coding) can yield extreme compression ratios.
- Made on-device inference practical for mobile and embedded applications, where model size directly impacts latency and power consumption.

---

## Key Takeaways

1. **Architecture design is a form of compression.** By designing efficient architectures from the start, you can achieve dramatic parameter reduction without accuracy loss.
2. The **squeeze-and-expand** pattern is a powerful design principle: reduce dimensionality cheaply with $1 \times 1$ convolutions, then process with a mix of $1 \times 1$ and $3 \times 3$ convolutions.
3. Late downsampling (Strategy 3) preserves spatial information and improves accuracy, but increases computation. The trade-off depends on the deployment target.
4. Combining architectural efficiency with post-training compression techniques (deep compression) yields multiplicative benefits: $50 \times$ from architecture, $\sim 10 \times$ from compression, totaling $510 \times$.
5. The optimal design involves a roughly equal mix of $1 \times 1$ and $3 \times 3$ filters in the expand layer, and a squeeze ratio of around 0.50.
