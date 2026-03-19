# WaveNet: A Generative Model for Raw Audio

**Authors:** Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
**Year:** 2016
**Venue:** arXiv preprint (presented at SSW9 2016)
**ArXiv:** 1609.03499

---

## Key Contributions

- Introduced a deep generative model that operates directly on raw audio waveforms at the sample level (e.g., 16,000 samples per second).
- Used **dilated causal convolutions** to achieve very large receptive fields while maintaining computational tractability.
- Applied **mu-law companding** and a softmax distribution over 256 quantized values to model the conditional distribution of each audio sample.
- Achieved state-of-the-art text-to-speech (TTS) quality, significantly closing the gap to natural human speech.
- Demonstrated the model's versatility on music generation and speech recognition tasks.

---

## Background & Motivation

Prior to WaveNet, text-to-speech systems relied on one of two approaches:

1. **Concatenative synthesis:** Stitching together fragments of pre-recorded speech. High quality but inflexible and requiring large databases.
2. **Parametric synthesis:** Generating audio from a statistical model (e.g., HMM-based). More flexible but producing muffled, robotic output.

Neural network approaches had been applied to parametric systems but still operated on derived features (spectrograms, vocoder parameters) rather than the raw waveform. WaveNet proposed to model the raw audio waveform directly with an autoregressive neural network, predicting one sample at a time.

---

## Method

### Autoregressive Formulation

The joint probability of a waveform $\mathbf{x} = \{x_1, x_2, \ldots, x_T\}$ is factored as:

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})$$

Each conditional distribution $p(x_t \mid x_1, \ldots, x_{t-1})$ is modeled by the neural network.

### Mu-Law Companding

Raw audio samples are 16-bit integers (65,536 possible values). To make softmax modeling feasible, the authors apply a mu-law companding transformation and quantize to 256 levels:

$$f(x_t) = \text{sign}(x_t) \frac{\ln(1 + \mu |x_t|)}{\ln(1 + \mu)}$$

where $\mu = 255$. This nonlinear quantization better preserves perceptual quality at low bit depths compared to linear quantization.

### Causal Convolutions

Standard convolutions violate the autoregressive property because they can access future timesteps. **Causal convolutions** mask the filter so that the output at time $t$ depends only on inputs at times $\leq t$:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

### Dilated Causal Convolutions

A stack of causal convolutions with kernel size $K$ and $L$ layers has a receptive field of $O(K \cdot L)$, which grows linearly and is too slow for audio. **Dilated convolutions** expand the receptive field exponentially by inserting gaps (dilation) between filter taps:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - k \cdot d}$$

where $d$ is the dilation factor. By stacking layers with dilation factors $d = 1, 2, 4, 8, \ldots, 512$ and repeating this pattern multiple times, WaveNet achieves receptive fields spanning thousands of timesteps.

| Layer | Dilation | Receptive Field Growth |
|-------|----------|----------------------|
| 1     | 1        | 2                    |
| 2     | 2        | 4                    |
| 3     | 4        | 8                    |
| ...   | ...      | ...                  |
| 10    | 512      | 1024                 |

With a stack of 10 layers repeated multiple times, the total receptive field covers several hundred milliseconds of audio.

### Gated Activation Units

Each layer uses a gated activation function inspired by the LSTM gating mechanism:

$$\mathbf{z} = \tanh(W_{f,k} * \mathbf{x}) \odot \sigma(W_{g,k} * \mathbf{x})$$

where $*$ denotes the (dilated) convolution, $\odot$ is element-wise multiplication, $\sigma$ is the sigmoid function, $W_{f,k}$ is the filter convolution, and $W_{g,k}$ is the gate convolution at layer $k$.

### Residual and Skip Connections

Each layer produces:
- A **residual** output added back to the input for the next layer.
- A **skip connection** output that is summed across all layers, passed through ReLU activations and $1 \times 1$ convolutions, and finally through a softmax to produce the output distribution.

### Conditioning

For TTS, WaveNet is conditioned on linguistic features (or speaker identity):

- **Global conditioning** (e.g., speaker ID): A single embedding $h$ is added to all layers: $\tanh(W_f * \mathbf{x} + V_f h) \odot \sigma(W_g * \mathbf{x} + V_g h)$
- **Local conditioning** (e.g., linguistic features at each timestep): A time-varying signal $\mathbf{h}_t$ is upsampled and added at each layer.

---

## Key Results

### Text-to-Speech MOS (Mean Opinion Score, 1-5 scale)

| System                      | English MOS | Mandarin MOS |
|-----------------------------|-------------|--------------|
| Natural speech              | 4.55        | 4.21         |
| **WaveNet**                 | **4.21**    | **4.08**     |
| Parametric (best baseline)  | 3.86        | 3.47         |
| Concatenative (best baseline)| 4.09       | 3.81         |

WaveNet significantly outperformed both parametric and concatenative baselines and narrowed the gap to natural speech.

### Speech Recognition (TIMIT)

| Model              | Test PER (%) |
|--------------------|--------------|
| **WaveNet**        | **18.8**     |
| Best RNN baseline  | 17.7         |

While not state-of-the-art on TIMIT, the result demonstrated that WaveNet's learned representations are useful beyond generation.

---

## Impact & Legacy

- WaveNet fundamentally changed the TTS landscape. Google deployed WaveNet in Google Assistant, making neural TTS the industry standard.
- The **dilated causal convolution** architecture became a foundational building block adopted in many subsequent models (ByteNet, TCN, Conv-TasNet).
- Inspired follow-up work to address the slow autoregressive generation: **Parallel WaveNet** (2017), **WaveRNN** (2018), **WaveGlow** (2018).
- Demonstrated that autoregressive models can handle extremely long sequences (tens of thousands of timesteps) when equipped with the right architectural inductive biases.
- Opened the door to end-to-end speech synthesis pipelines (Tacotron + WaveNet).

---

## Key Takeaways

1. Dilated causal convolutions enable exponentially growing receptive fields with linear parameter growth, making it feasible to model long-range dependencies in sequential data.
2. Quantizing audio to 256 levels via mu-law companding and modeling with a categorical distribution (softmax) works surprisingly well, avoiding the need for mixture density or continuous distribution outputs.
3. Gated activations consistently outperform ReLU for modeling audio, likely because audio signals have both positive and negative components with complex multiplicative interactions.
4. Conditioning mechanisms (global and local) provide a clean framework for controllable generation.
5. The main drawback of WaveNet is slow autoregressive inference: generating one second of 16 kHz audio requires 16,000 sequential forward passes.
