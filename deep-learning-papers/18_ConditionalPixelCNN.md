# Conditional Image Generation with PixelCNN Decoders

**Authors:** Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, Koray Kavukcuoglu
**Year:** 2016
**Venue:** NeurIPS 2016
**Paper:** [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)

---

## Key Contributions

- Proposed the **Gated PixelCNN**, an improved autoregressive model that replaces ReLU activations with gated convolutional units, significantly improving sample quality.
- Solved the **blind spot problem** of the original PixelCNN by introducing separate vertical and horizontal convolutional stacks.
- Introduced **conditional generation** with PixelCNN by conditioning on class labels (class-conditional) or on latent representations from other models (e.g., a VAE encoder).
- Proposed using PixelCNN as a powerful **decoder for VAEs**, replacing the typical factored Gaussian decoder and enabling much sharper generated images.
- Achieved state-of-the-art log-likelihood results on CIFAR-10 and ImageNet.

---

## Background & Motivation

Autoregressive generative models decompose the joint distribution of an image into a product of conditionals:

$$p(\mathbf{x}) = \prod_{i=1}^{N} p(x_i \mid x_1, \ldots, x_{i-1})$$

where pixels are ordered (e.g., raster scan: left-to-right, top-to-bottom).

**PixelRNN** (van den Oord et al., 2016) modeled these conditionals with LSTMs -- accurate but slow due to sequential processing. **PixelCNN** replaced LSTMs with masked convolutions for parallelizable training but suffered from two problems:

1. **Blind spot**: The masked convolution pattern could not access all previously generated pixels, creating a "blind spot" in the receptive field to the right of the current pixel.
2. **Limited expressiveness**: Standard ReLU convolutions were insufficient for modeling the complex conditional distributions.

Gated PixelCNN addresses both issues and extends the model to conditional generation.

---

## Method / Architecture

### Masked Convolutions Review

To maintain the autoregressive property, convolutions are masked so that the prediction for pixel $i$ depends only on pixels $1, \ldots, i-1$. Two mask types:

- **Mask A** (first layer only): Excludes the center pixel.
- **Mask B** (subsequent layers): Includes the center pixel (since it was already masked in the first layer).

### Blind Spot Problem and Split Stacks

The original PixelCNN's horizontal masking created a blind spot: pixels to the upper-right of the current position could not influence the prediction.

**Solution**: Split the network into two separate convolutional stacks:

1. **Vertical Stack**: Applies $k \times k$ convolutions using only rows above the current row (no masking needed horizontally). This captures the full context from all rows above.

2. **Horizontal Stack**: Applies $1 \times k$ convolutions using only pixels to the left of (and including, with Mask B) the current position in the current row.

The vertical stack feeds into the horizontal stack via $1 \times 1$ convolutions at each layer, so the horizontal stack has access to all previously generated pixels -- eliminating the blind spot.

### Gated Activation Units

Replacing ReLU activations with gated units (inspired by LSTMs and gated recurrent units):

$$\mathbf{y} = \tanh(W_{k,f} * \mathbf{x}) \odot \sigma(W_{k,g} * \mathbf{x})$$

where:
- $W_{k,f}$ is the filter convolution (produces the "content")
- $W_{k,g}$ is the gate convolution (controls information flow)
- $\odot$ is element-wise multiplication
- $\tanh$ provides the candidate activation
- $\sigma$ (sigmoid) provides the gate

This gating mechanism models multiplicative interactions between features, which is more expressive than additive ReLU for complex pixel distributions.

### Conditional Generation

The model can be conditioned on an external variable $\mathbf{h}$ (e.g., class label, latent vector):

$$p(\mathbf{x} \mid \mathbf{h}) = \prod_{i=1}^{N} p(x_i \mid x_1, \ldots, x_{i-1}, \mathbf{h})$$

The conditioning is incorporated into the gated activation:

$$\mathbf{y} = \tanh(W_{k,f} * \mathbf{x} + V_{k,f}^T \mathbf{h}) \odot \sigma(W_{k,g} * \mathbf{x} + V_{k,g}^T \mathbf{h})$$

For a **global** conditioning variable (e.g., class label one-hot vector), $V^T \mathbf{h}$ is a simple bias added to every spatial position.

For **spatial** conditioning (e.g., from another network), $\mathbf{h}$ is a spatial feature map and $V * \mathbf{h}$ is a $1 \times 1$ convolution:

$$\mathbf{y} = \tanh(W_{k,f} * \mathbf{x} + V_{k,f} * \mathbf{h}) \odot \sigma(W_{k,g} * \mathbf{x} + V_{k,g} * \mathbf{h})$$

### PixelCNN as VAE Decoder

Standard VAEs use a factored Gaussian decoder $p_\theta(\mathbf{x} \mid \mathbf{z}) = \prod_i \mathcal{N}(x_i; \mu_i(\mathbf{z}), \sigma_i(\mathbf{z}))$, which produces blurry samples because it assumes pixel independence given $\mathbf{z}$.

Replacing this with a conditional PixelCNN decoder:

$$p_\theta(\mathbf{x} \mid \mathbf{z}) = \prod_{i=1}^{N} p_\theta(x_i \mid x_1, \ldots, x_{i-1}, \mathbf{z})$$

This autoregressive decoder can model sharp, detailed images because it captures pixel-level dependencies. The latent $\mathbf{z}$ captures global structure while the PixelCNN captures local detail.

The VAE objective remains:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$$

A challenge is **posterior collapse**: the PixelCNN decoder may learn to ignore $\mathbf{z}$ entirely if it can model $p(\mathbf{x})$ well on its own. The paper addresses this by ensuring $\mathbf{z}$ carries useful high-level information.

### Output Distribution

Each color channel is modeled as a 256-way softmax (discrete values 0-255):

$$p(x_i \mid \cdot) = \text{Softmax}(f_\theta(x_1, \ldots, x_{i-1}))$$

with separate distributions for R, G, B channels conditioned on previously generated channels.

---

## Key Results

### CIFAR-10 (Negative Log-Likelihood in bits/dim)

| Model | NLL (bits/dim) |
|---|---|
| NICE (Dinh et al., 2014) | 4.48 |
| DRAW (Gregor et al., 2015) | 4.13 |
| Pixel RNN (Row LSTM) | 3.00 |
| Pixel RNN (Diagonal BiLSTM) | 2.93 |
| PixelCNN (original) | 3.03 |
| **Gated PixelCNN** | **2.92** |

### ImageNet (Negative Log-Likelihood in bits/dim)

| Model | NLL (bits/dim) |
|---|---|
| Pixel RNN (Row LSTM) | 3.63 |
| Pixel RNN (Diagonal BiLSTM) | 3.57 |
| **Gated PixelCNN** | **3.57** |

Gated PixelCNN matches PixelRNN accuracy while being significantly faster to train (fully parallelizable during training).

### PixelCNN + VAE

| Model | NLL (bits/dim) on CIFAR-10 |
|---|---|
| VAE (Gaussian decoder) | 4.51 |
| VAE + IAF | 3.11 |
| **PixelCNN decoder (no latents)** | **2.92** |
| **PixelCNN + VAE** | Competitive with pure PixelCNN |

The PixelCNN + VAE combination produces sharper samples than standard VAE while learning meaningful latent representations (unlike pure PixelCNN, which has no global latent space).

### Training Speed

- Gated PixelCNN trains ~2x faster than PixelRNN (Diagonal BiLSTM) due to full parallelization of convolutions during training.
- Sampling remains sequential ($O(N)$ for $N$ pixels).

---

## Impact & Legacy

- **Gated PixelCNN became the standard autoregressive image model**, replacing PixelRNN in most subsequent work due to its training efficiency.
- The **vertical/horizontal stack** architecture for solving the blind spot problem became the standard approach for masked convolutions.
- **Conditional generation** via PixelCNN enabled class-conditional image synthesis and influenced subsequent conditional generative models.
- The **PixelCNN + VAE** combination demonstrated that autoregressive decoders can be integrated with latent variable models, leading to VQ-VAE and VQ-VAE-2 (which use PixelCNN priors over discrete latent codes).
- **WaveNet** (van den Oord et al., 2016) applied similar gated, dilated autoregressive convolutions to audio, revolutionizing speech synthesis.
- The autoregressive paradigm for images, while largely surpassed by diffusion models for generation quality, remains important for density estimation and lossless compression.
- PixelCNN's use as a prior/decoder influenced the design of VQ-VAE (2017), a key precursor to DALL-E (2021).

---

## Key Takeaways

1. **Gated activations** ($\tanh \odot \sigma$) are significantly more expressive than ReLU for modeling complex conditional distributions in autoregressive models.
2. **Splitting into vertical and horizontal stacks** is a clean solution to the blind spot problem, ensuring every previously generated pixel contributes to predictions.
3. **Conditioning** can be flexibly incorporated (global labels, spatial features, latent codes) by adding bias terms to the gated activations.
4. **Autoregressive decoders** for VAEs produce much sharper images than factored Gaussian decoders, because they model pixel-level dependencies rather than assuming conditional independence.
5. There is a **fundamental tension** in PixelCNN + VAE: the powerful decoder can ignore the latent code (posterior collapse). Managing this trade-off between local autoregressive modeling and global latent representation remains an active challenge.
