# Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding

**Authors:** Song Han, Huizi Mao, William J. Dally
**Year:** 2016
**Venue:** ICLR 2016 (Best Paper Award)
**ArXiv:** 1510.00149

---

## Key Contributions

- Proposed a three-stage **compression pipeline** (pruning, quantization, Huffman coding) that reduces storage requirements of neural networks by **35x-49x** without loss of accuracy.
- Compressed AlexNet from 240 MB to **6.9 MB** (35x) and VGG-16 from 552 MB to **11.3 MB** (49x) with no loss in top-1 or top-5 accuracy.
- Demonstrated that the three compression techniques are complementary and can be applied sequentially as a pipeline, with each stage building on the previous one.
- Won the **ICLR 2016 Best Paper Award** and became the foundation for the efficient neural network deployment field.

---

## Background & Motivation

State-of-the-art deep neural networks in 2016 had tens to hundreds of millions of parameters:

| Model    | Parameters | Storage  |
|----------|-----------|----------|
| AlexNet  | 60M       | 240 MB   |
| VGG-16   | 138M      | 552 MB   |

Deploying these models on mobile devices, embedded systems, and FPGAs was impractical due to:
1. **Storage constraints:** Mobile devices have limited memory.
2. **Energy cost:** Memory access dominates energy consumption; DRAM access costs $100\times$ more energy than a floating-point multiply.
3. **Bandwidth limitations:** Transmitting large models over networks is costly.

The central insight is that neural networks are significantly **over-parameterized**: most of the parameters (weights) can be removed, quantized, or encoded more efficiently without impacting accuracy.

---

## Method

The Deep Compression pipeline consists of three sequential stages:

### Stage 1: Pruning

**Goal:** Remove unimportant connections (weights close to zero).

1. **Train** the network normally to learn which connections are important.
2. **Prune** all connections with weights below a threshold $t$:

$$w_{ij} = 0 \quad \text{if} \quad |w_{ij}| < t$$

3. **Retrain** (fine-tune) the remaining connections to recover any accuracy lost from pruning.

This process can be iterated. After pruning, the network is stored as a **sparse matrix** using Compressed Sparse Row (CSR) or Compressed Sparse Column (CSC) format, which only stores non-zero weights and their indices.

**Typical pruning rates:**

| Layer Type        | Typical Pruning Rate |
|-------------------|---------------------|
| Fully Connected   | 90-96% pruned       |
| Convolutional     | 50-80% pruned       |

For AlexNet, pruning reduces the number of weights from 60M to 6.7M ($9\times$ reduction) without accuracy loss.

**Index encoding:** Instead of storing absolute indices for non-zero weights, the authors encode the **difference** between consecutive indices. These differences are small and stored with fewer bits (e.g., 4 bits for convolutional layers, 5 bits for fully connected layers). If the difference exceeds the representable range, a zero is inserted as padding.

### Stage 2: Trained Quantization (Weight Sharing)

**Goal:** Reduce the number of bits needed to represent each weight.

Instead of storing each weight as a 32-bit floating-point number, weights are **clustered** into shared groups, and only the cluster index is stored for each weight.

1. Apply **k-means clustering** to the weights of each layer, grouping them into $k$ clusters:

$$\min_{C_1, \ldots, C_k} \sum_{j=1}^{k} \sum_{w \in C_j} |w - c_j|^2$$

where $c_j$ is the centroid of cluster $C_j$.

2. Replace each weight with its **cluster index** (an integer from 0 to $k-1$).
3. Store a **codebook** of $k$ centroid values.
4. During fine-tuning, gradients for all weights in a cluster are **summed** and applied to the shared centroid:

$$c_j \leftarrow c_j - \eta \sum_{w \in C_j} \frac{\partial \mathcal{L}}{\partial w}$$

**Bits per weight:**

| Number of Clusters $k$ | Bits per Weight |
|------------------------|-----------------|
| 4                      | 2               |
| 16                     | 4               |
| 32                     | 5               |
| 256                    | 8               |

The authors found that **4 bits** (16 clusters) for convolutional layers and **2 bits** (4 clusters) for fully connected layers provide good accuracy retention.

**Codebook storage overhead** is negligible: for $k$ clusters and $n$ weights, the codebook uses $32k$ bits while the indices use $n \log_2 k$ bits. Since $n \gg k$, the codebook is a tiny fraction of total storage.

**Initialization of centroids matters.** The authors compare three initialization strategies:

| Initialization | Description | Quality |
|---------------|-------------|---------|
| Random         | Random selection from weight values | Worst |
| Density-based  | Sample proportional to density of weights | Medium |
| **Linear**     | Linearly space between [min, max] | **Best** |

Linear initialization works best because it distributes centroids evenly across the weight range, ensuring that large (important) weights far from zero are well-represented.

### Stage 3: Huffman Coding

**Goal:** Exploit the non-uniform distribution of cluster indices and sparse matrix indices through entropy coding.

The distribution of quantized weights and of index differences is highly non-uniform. Huffman coding assigns shorter bit sequences to more frequent values:

$$\text{Expected bits/symbol} = \sum_{i} p_i \cdot l_i$$

where $p_i$ is the probability of symbol $i$ and $l_i$ is its code length. Huffman coding approaches the entropy lower bound:

$$H = -\sum_i p_i \log_2 p_i$$

This final stage provides an additional **20-30% reduction** in model size.

### Full Pipeline Summary

| Stage           | Technique                        | AlexNet Compression | VGG-16 Compression |
|-----------------|----------------------------------|--------------------|--------------------|
| Original        | 32-bit float                     | 1x (240 MB)       | 1x (552 MB)       |
| After Pruning   | Sparse representation            | 9x (27 MB)        | 13x (42 MB)       |
| After Quantization | Weight sharing + codebook     | 27x (8.9 MB)      | 36x (15 MB)       |
| After Huffman   | Entropy coding                   | **35x (6.9 MB)**  | **49x (11.3 MB)** |

---

## Key Results

### AlexNet Compression

| Model                     | Top-1 Acc. (%) | Top-5 Acc. (%) | Size    |
|---------------------------|---------------|---------------|---------|
| AlexNet (original)        | 57.2          | 80.3          | 240 MB  |
| Pruned                    | 57.2          | 80.3          | 27 MB   |
| Pruned + Quantized        | 57.1          | 80.3          | 8.9 MB  |
| **Pruned + Quantized + Huffman** | **57.2** | **80.3**   | **6.9 MB** |

### VGG-16 Compression

| Model                     | Top-1 Acc. (%) | Top-5 Acc. (%) | Size    |
|---------------------------|---------------|---------------|---------|
| VGG-16 (original)         | 68.5          | 88.7          | 552 MB  |
| **Pruned + Quantized + Huffman** | **68.5** | **88.7**   | **11.3 MB** |

### Comparison with Other Compression Methods

| Method                        | Network | Compression Rate | Accuracy Loss |
|-------------------------------|---------|-----------------|---------------|
| Low-rank decomposition        | VGG-16  | 3x              | ~1%           |
| HashedNets                    | --      | 8x              | variable      |
| Pruning only (Han et al. 2015)| AlexNet | 9x              | none          |
| **Deep Compression (full)**   | AlexNet | **35x**         | **none**      |
| **Deep Compression (full)**   | VGG-16  | **49x**         | **none**      |

### Layer-by-Layer Analysis (AlexNet)

| Layer  | Original Params | After Pruning | Bits per Weight | Compression Ratio |
|--------|----------------|---------------|-----------------|-------------------|
| conv1  | 35K            | 16K (54%)     | 8               | 5x                |
| conv2  | 307K           | 78K (74%)     | 8               | 14x               |
| conv3  | 885K           | 265K (70%)    | 8               | 14x               |
| conv4  | 663K           | 220K (67%)    | 8               | 13x               |
| conv5  | 442K           | 159K (64%)    | 8               | 12x               |
| fc6    | 37.8M          | 3.5M (91%)    | 4               | 63x               |
| fc7    | 16.8M          | 1.7M (90%)    | 4               | 63x               |
| fc8    | 4.1M           | 0.4M (90%)    | 4               | 56x               |

Fully connected layers benefit most from compression due to their high redundancy.

---

## Impact & Legacy

- **Won the ICLR 2016 Best Paper Award**, establishing model compression as a first-class research area.
- Directly motivated the development of **hardware accelerators** (EIE, ESE) designed to exploit sparse and quantized models.
- Spawned an entire subfield of neural network compression research: structured pruning, lottery ticket hypothesis, knowledge distillation, mixed-precision quantization, and neural architecture search for efficiency.
- The pruning-retraining cycle influenced the **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019), which asks whether the pruned subnetwork could have been trained from scratch.
- Industry adoption: model quantization (to 8-bit and 4-bit) became standard practice for mobile deployment (TensorFlow Lite, ONNX Runtime, PyTorch Mobile).
- Complementary to efficient architecture design (SqueezeNet, MobileNet): Deep Compression can be applied on top of already-efficient architectures for further reduction.

---

## Key Takeaways

1. The three stages (pruning, quantization, Huffman coding) are complementary, and their compression ratios roughly multiply: $9 \times 3 \times 1.3 \approx 35$ for AlexNet.
2. **Neural networks are massively over-parameterized.** 90%+ of weights in fully connected layers can be removed without accuracy loss.
3. **Weight sharing via k-means clustering** is a simple yet effective quantization method. Fine-tuning the shared centroids recovers accuracy lost from quantization.
4. **Linear initialization** of k-means centroids outperforms random and density-based initialization because it preserves the representation of large-magnitude weights.
5. The compression pipeline is **lossless** in practice: careful retraining after each stage recovers any accuracy drop, making the full pipeline achieve 35-49x compression with zero accuracy degradation.
6. Fully connected layers are the primary storage bottleneck and benefit most from compression. Modern architectures that reduce or eliminate FC layers (global average pooling) achieve some of these savings through architecture design alone.
