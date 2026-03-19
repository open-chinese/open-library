# Effective Approaches to Attention-based Neural Machine Translation (Luong Attention)

**Authors:** Minh-Thang Luong, Hieu Pham, Christopher D. Manning
**Year:** 2015 (widely adopted 2015-2016)
**Venue:** EMNLP 2015
**Institution:** Stanford University

---

## Key Contributions

- Proposed and systematically compared **global** and **local** attention mechanisms for neural machine translation.
- Introduced **multiplicative (dot-product) attention**, a simpler and computationally cheaper alternative to Bahdanau's additive attention.
- Demonstrated that local attention, which focuses on a small window of source positions, can be more efficient and sometimes more effective than attending to all positions.
- Established a cleaner architectural framework where attention is applied at the top LSTM layer rather than being interleaved with the RNN computation.

---

## Background & Motivation

Bahdanau et al. (2015) introduced attention for NMT but used a relatively complex additive scoring function and integrated attention into the decoder in a way that mixed GRU dynamics with attention computation. Luong et al. sought to simplify the attention mechanism, explore alternative scoring functions, and investigate whether attending to only a subset of source positions (local attention) could match or outperform global attention while reducing computational cost.

---

## Method / Architecture

### Overall Architecture

Both encoder and decoder use stacked LSTMs. The key difference from Bahdanau attention is when and how the context vector is computed:

1. The decoder LSTM first produces hidden state $h_t$ at step $t$.
2. The context vector $c_t$ is computed from $h_t$ and the source hidden states.
3. The attentional hidden state is:

$$\tilde{h}_t = \tanh(W_c [c_t; h_t])$$

4. The output distribution is:

$$p(y_t \mid y_{<t}, x) = \text{softmax}(W_s \tilde{h}_t)$$

### Score Functions

The paper proposes three scoring functions for computing alignment between decoder state $h_t$ and source state $\bar{h}_s$:

| Name | Score Function $\text{score}(h_t, \bar{h}_s)$ |
|:-----|:----------------------------------------------|
| **Dot** | $h_t^\top \bar{h}_s$ |
| **General** | $h_t^\top W_a \bar{h}_s$ |
| **Concat** (Bahdanau-style) | $v_a^\top \tanh(W_a [h_t; \bar{h}_s])$ |

The **dot product** variant is the simplest and requires no additional parameters. The **general** variant adds a learned weight matrix. Both are multiplicative alternatives to Bahdanau's additive (concat) approach.

### Global Attention

Global attention considers **all** source hidden states when computing the context vector:

$$\alpha_{ts} = \frac{\exp(\text{score}(h_t, \bar{h}_s))}{\sum_{s'=1}^{S} \exp(\text{score}(h_t, \bar{h}_{s'}))}$$

$$c_t = \sum_{s} \alpha_{ts} \bar{h}_s$$

This is analogous to Bahdanau attention but with the cleaner top-layer formulation and alternative score functions.

### Local Attention

Local attention attends to a **small window** of source positions centered around an aligned position $p_t$:

$$c_t = \sum_{s=p_t - D}^{p_t + D} \alpha_{ts} \bar{h}_s$$

where $D$ is the window half-width (a hyperparameter).

Two variants for determining $p_t$:

**Monotonic alignment (local-m):**

$$p_t = t$$

Simply aligns position $t$ in the target to position $t$ in the source.

**Predictive alignment (local-p):**

$$p_t = S \cdot \sigma(v_p^\top \tanh(W_p h_t))$$

where $S$ is the source sentence length. A Gaussian bias favors positions near $p_t$:

$$\alpha_{ts} = \text{align}(h_t, \bar{h}_s) \cdot \exp\left(-\frac{(s - p_t)^2}{2\sigma^2}\right)$$

with $\sigma = D/2$ set empirically.

### Comparison: Bahdanau vs. Luong Attention

| Aspect | Bahdanau (Additive) | Luong (Multiplicative) |
|:-------|:-------------------:|:----------------------:|
| Score function | $v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$ | $h_t^\top W_a \bar{h}_s$ or $h_t^\top \bar{h}_s$ |
| Decoder state used | Previous state $s_{i-1}$ | Current state $h_t$ |
| Attention location | Between GRU layers | On top of stacked LSTM |
| Scope | Global only | Global and local |
| Computational cost | Higher (MLP) | Lower (matrix multiply) |

### Input Feeding

The paper also introduces **input feeding**, where the attentional vector $\tilde{h}_t$ is concatenated with the next input to the decoder:

$$\text{input}_t = [y_{t-1}; \tilde{h}_{t-1}]$$

This allows the model to be aware of previous alignment decisions when making the next one.

---

## Key Results

### WMT'14 English-to-German Translation

| System | BLEU |
|:-------|:----:|
| Baseline (no attention) | 17.4 |
| Global attention (dot) | 20.9 |
| Global attention (general) | 20.6 |
| Local-p attention (general) | 20.9 |
| Local-p + input feeding | **21.0** |
| Ensemble of 8 models | **23.0** |
| WMT'15 best single NMT | 20.9 |

### WMT'15 English-to-German Translation

| System | BLEU |
|:-------|:----:|
| Luong attention ensemble | **25.9** |
| Best NMT at WMT'15 | 24.9 |

### Attention Type Comparison

| Attention Type | Perplexity | BLEU |
|:---------------|:----------:|:----:|
| None (no attention) | 10.6 | 17.4 |
| Global (dot) | 7.3 | 20.9 |
| Global (general) | 7.6 | 20.6 |
| Global (concat) | 8.0 | 19.8 |
| Local-m (general) | 8.5 | 19.4 |
| Local-p (general) | **7.0** | **20.9** |

The simple dot-product score performed competitively with more complex alternatives, while local-p attention with predictive alignment achieved the best perplexity.

---

## Impact & Legacy

- **Dot-product attention became the default:** The Transformer's scaled dot-product attention directly descends from Luong's dot-product scoring.
- **Local attention inspired windowed/sparse attention:** Later works on efficient attention (Longformer, BigBird) echo the local attention concept.
- **Input feeding widely adopted:** Feeding attention context back into the decoder became standard practice in RNN-based Seq2Seq models.
- **Cleaner formulation:** The top-layer attention approach simplified implementation and became the standard in frameworks like OpenNMT.
- **Comparative study value:** The systematic comparison across scoring functions and attention types guided practitioners in choosing appropriate mechanisms.

---

## Key Takeaways

1. Multiplicative (dot-product) attention is simpler, faster, and comparably effective to additive attention, making it the preferred choice in most subsequent work.
2. Local attention offers a useful trade-off: reduced computational cost with competitive or better accuracy, especially with predictive alignment.
3. Using the current decoder hidden state (rather than the previous one) for attention scoring is a cleaner and more effective design.
4. Input feeding -- passing the previous attention context back to the decoder -- helps the model make alignment-aware predictions.
5. The systematic comparison of attention variants provided the field with foundational design guidance that influenced the Transformer and many other architectures.
