# Sequence to Sequence Learning with Neural Networks

**Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le
**Year:** 2014 (major application impact through 2016)
**Venue:** NeurIPS 2014
**Institution:** Google

---

## Key Contributions

- Introduced a general end-to-end sequence-to-sequence (Seq2Seq) framework using two LSTMs: one as an encoder and one as a decoder.
- Demonstrated that a simple, large deep LSTM can achieve competitive machine translation results without task-specific engineering.
- Discovered that reversing the order of source sentence words significantly improves performance.
- Established the encoder-decoder paradigm that became the foundation for virtually all neural sequence transduction models through 2016 and beyond.

---

## Background & Motivation

Traditional sequence transduction tasks (e.g., machine translation, speech recognition) relied on complex pipelines with hand-crafted features and domain-specific components. Deep neural networks had shown success on fixed-dimensional inputs and outputs, but mapping variable-length sequences to variable-length sequences remained a challenge. The authors sought a minimal, general-purpose architecture that could learn this mapping end-to-end from data.

---

## Method / Architecture

### Encoder-Decoder Framework

The model uses two separate multilayer LSTM networks:

1. **Encoder LSTM:** Reads the input sequence $x_1, x_2, \ldots, x_T$ one token at a time and produces a fixed-dimensional context vector $\mathbf{c}$, which is the final hidden state of the encoder.

2. **Decoder LSTM:** Generates the output sequence $y_1, y_2, \ldots, y_{T'}$ conditioned on $\mathbf{c}$.

### Objective

The model maximizes the conditional log-probability of the target sequence given the source:

$$\max_\theta \frac{1}{|S|} \sum_{(x,y) \in S} \log p(y \mid x; \theta)$$

where $S$ is the training set of (source, target) pairs.

### Conditional Probability Decomposition

The probability of the output sequence is factorized autoregressively:

$$p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t \mid \mathbf{c}, y_1, \ldots, y_{t-1})$$

Each conditional $p(y_t \mid \mathbf{c}, y_1, \ldots, y_{t-1})$ is represented by a softmax over the vocabulary at each decoder time step.

### LSTM Equations

At each time step $t$, the LSTM computes:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise multiplication.

### Key Design Choices

- **Deep LSTMs:** 4 layers deep for both encoder and decoder (deep models significantly outperformed shallow ones).
- **Input reversal:** The source sentence is fed in reverse order ($x_T, x_{T-1}, \ldots, x_1$), which introduced many short-term dependencies and improved optimization.
- **No explicit alignment mechanism:** The entire source meaning is compressed into a single fixed-length vector $\mathbf{c}$.

---

## Key Results

### WMT'14 English-to-French Translation

| Model | BLEU Score |
|:------|:----------:|
| Baseline phrase-based SMT | 33.3 |
| Single Seq2Seq LSTM | 34.8 |
| Ensemble of 5 Seq2Seq LSTMs | 34.8 |
| Seq2Seq + SMT 1000-best rescoring | **36.5** |
| Best WMT'14 system (phrase-based + large LM) | 37.0 |

### Effect of Input Reversal

| Configuration | BLEU |
|:------|:----:|
| Normal source order | 30.6 |
| Reversed source order | **33.3** |

Reversing the input improved BLEU by 2.7 points, a substantial gain attributed to introducing short-term dependencies between aligned word pairs.

### Sentence Length Robustness

The model handled sentences of length up to 30-35 tokens well, which was surprising given the fixed-dimensional bottleneck. Performance degraded for very long sentences, motivating subsequent attention mechanisms.

---

## Impact & Legacy

- **Foundation of modern NLP:** The Seq2Seq paradigm became the standard framework for machine translation, summarization, dialogue systems, and many other tasks.
- **Motivated attention mechanisms:** The fixed-length bottleneck limitation directly inspired Bahdanau et al. (2015) to develop attention, and subsequently the Transformer architecture (2017).
- **Enabled end-to-end learning:** Shifted the field away from pipeline-based systems toward end-to-end trainable models.
- **Widely adopted by industry:** Google, Facebook, and others integrated Seq2Seq models into production translation systems by 2016.
- **Spawned countless extensions:** Attention, copy mechanisms, pointer networks, and more all build on the encoder-decoder foundation.

---

## Key Takeaways

1. A simple encoder-decoder LSTM architecture can learn to map variable-length sequences to variable-length sequences without any task-specific engineering.
2. Depth matters: 4-layer LSTMs significantly outperform single-layer LSTMs for sequence transduction.
3. Reversing the source input is a surprisingly effective trick that improves learning by creating shorter-range dependencies between aligned words.
4. The fixed-length context vector is both the strength (simplicity) and the weakness (information bottleneck) of the approach, directly motivating the development of attention mechanisms.
5. Large-scale data and compute can compensate for architectural simplicity -- the model used 8 GPUs training for 10 days.
