# Pixel Recurrent Neural Networks

**Authors:** Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu
**Year:** 2016
**Venue:** ICML 2016
**ArXiv:** 1601.06759

---

## Key Contributions

- Proposed **PixelRNN** and **PixelCNN**, autoregressive models that generate images one pixel at a time by modeling the joint distribution of pixels as a product of conditionals.
- Introduced **Row LSTM** and **Diagonal BiLSTM** architectures with masked convolutions to capture spatial dependencies while respecting the autoregressive ordering.
- Introduced **masked convolutions** for the PixelCNN variant, enabling parallelized training without violating the causal constraint.
- Achieved state-of-the-art log-likelihood on MNIST and CIFAR-10, demonstrating that autoregressive models can compete with and surpass VAEs and other generative models on density estimation.
- Provided a tractable exact likelihood computation, unlike GANs (no likelihood) and VAEs (approximate likelihood).

---

## Background & Motivation

Generative models of images aim to learn $p(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^{n \times n \times 3}$ is an image. Existing approaches had significant limitations:

- **GANs** produce sharp images but provide no density estimate and suffer from mode collapse.
- **VAEs** provide a tractable lower bound (ELBO) on the likelihood but tend to produce blurry outputs.
- **Autoregressive models** offered exact likelihood computation but prior approaches (NADE, MADE) were limited to low-resolution or binary images.

The key insight is to decompose the image distribution using the chain rule and model each conditional with a powerful neural network that respects the autoregressive ordering.

---

## Method

### Autoregressive Decomposition

An image $\mathbf{x}$ with $n^2$ pixels is generated sequentially. Using a raster-scan ordering (left-to-right, top-to-bottom):

$$p(\mathbf{x}) = \prod_{i=1}^{n^2} p(x_i \mid x_1, x_2, \ldots, x_{i-1})$$

For color images, each pixel has three channels (R, G, B), so the decomposition is further refined:

$$p(x_i) = p(x_{i,R} \mid \mathbf{x}_{<i}) \cdot p(x_{i,G} \mid \mathbf{x}_{<i}, x_{i,R}) \cdot p(x_{i,B} \mid \mathbf{x}_{<i}, x_{i,R}, x_{i,G})$$

Each sub-pixel value is discrete (0-255), modeled as a 256-way softmax.

### Architecture Variants

#### 1. Row LSTM

Processes the image row by row. For each row, an LSTM computes hidden states that depend on the entire context above (via convolutions) and to the left (via the recurrent connection):

$$[\mathbf{o}_i, \mathbf{f}_i, \mathbf{i}_i, \mathbf{g}_i] = \sigma(K^{ss} * \mathbf{h}_{i-1} + K^{is} * \mathbf{x}_i)$$

$$\mathbf{c}_i = \mathbf{f}_i \odot \mathbf{c}_{i-1} + \mathbf{i}_i \odot \mathbf{g}_i$$

$$\mathbf{h}_i = \mathbf{o}_i \odot \tanh(\mathbf{c}_i)$$

where $K^{ss}$ and $K^{is}$ are convolution kernels (state-to-state and input-to-state), and $*$ denotes convolution. The input-to-state convolution uses a kernel of size $k \times 1$ applied to the rows above, capturing a triangular context region.

#### 2. Diagonal BiLSTM

This variant achieves a fully connected receptive field over all previously generated pixels. The image rows are skewed so that computation along each diagonal can be parallelized. Two LSTMs (one left-to-right, one right-to-left on the skewed map) together cover the entire context for each pixel.

The Diagonal BiLSTM has the largest receptive field and the best performance but is also the slowest.

#### 3. PixelCNN (Masked Convolutions)

Instead of recurrence, PixelCNN uses standard convolutional layers with **masks** that zero out weights corresponding to future pixels:

$$p(x_i \mid x_{<i}) = f(\text{MaskedConv}(\mathbf{x}))_i$$

Two types of masks are used:

| Mask Type | Center Pixel Included? | Usage                  |
|-----------|----------------------|------------------------|
| **Mask A** | No                   | First layer only       |
| **Mask B** | Yes                  | All subsequent layers  |

Mask A ensures the prediction for pixel $i$ does not depend on pixel $i$ itself. Mask B allows subsequent layers to use the features computed for the center pixel (since those features already respect causality from Mask A in the first layer).

PixelCNN is significantly faster to train than PixelRNN because all positions can be computed in parallel (teacher forcing). However, it has a bounded receptive field determined by the number of layers and kernel size.

### Output Distribution

Each conditional is parameterized as a 256-way softmax:

$$p(x_i = k \mid x_{<i}) = \frac{\exp(h_{i,k})}{\sum_{j=0}^{255} \exp(h_{i,j})}$$

where $h_{i,k}$ is the network's logit for value $k$ at position $i$.

### Residual Connections

Both PixelRNN and PixelCNN use residual connections between layers:

$$\mathbf{h}^{l+1} = \mathbf{h}^l + \text{Layer}(\mathbf{h}^l)$$

This enables training of deeper models (up to 12 layers of LSTM or many convolutional layers).

---

## Key Results

### Negative Log-Likelihood (NLL) in bits/dim on Test Set

| Model                | MNIST  | CIFAR-10 |
|----------------------|--------|----------|
| **Diagonal BiLSTM**  | **79.20** | **3.00**  |
| Row LSTM             | 79.25  | 3.02     |
| PixelCNN             | 81.30  | 3.14     |
| DRAW (Gregor et al.) | 80.97  | 4.13     |
| NICE (Dinh et al.)   | 4.48   | 4.48     |
| Real NVP             | --     | 3.49     |
| Deep VAE             | 79.66  | --       |

The Diagonal BiLSTM achieved the best results across both benchmarks, with the Row LSTM close behind. Even the simpler PixelCNN outperformed competing generative model families.

---

## Impact & Legacy

- Established autoregressive models as a dominant paradigm for density estimation, eventually leading to PixelCNN++ (Salimans et al., 2017), Gated PixelCNN (van den Oord et al., 2016), and VQ-VAE (van den Oord et al., 2017).
- The **masked convolution** idea became foundational and was later adopted in language modeling (GPT series uses causal masking, though in the attention domain).
- Demonstrated that exact likelihood models can generate high-quality images, providing a principled alternative to GANs.
- Influenced the design of WaveNet (same first author), which adapted the autoregressive convolution idea to audio.
- The slow sequential generation problem motivated research into parallel/non-autoregressive generation methods.

---

## Key Takeaways

1. The chain rule provides a mathematically exact way to decompose any joint distribution into a product of conditionals, and deep networks can model each conditional.
2. Masked convolutions are a simple and effective way to enforce causal ordering in convolutional networks, enabling parallel training.
3. There is a clear trade-off between receptive field completeness (Diagonal BiLSTM is best) and computational efficiency (PixelCNN is fastest).
4. Modeling pixel intensities as discrete categorical values (256-way softmax) works well and avoids assumptions about the form of the continuous distribution.
5. Autoregressive image models provide exact log-likelihoods, making model comparison straightforward, but generation is inherently sequential and slow.
