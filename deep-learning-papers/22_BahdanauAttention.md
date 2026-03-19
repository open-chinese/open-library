# Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)

**Authors:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
**Year:** 2015 (widely used through 2016 and beyond)
**Venue:** ICLR 2015
**Institution:** Universite de Montreal

---

## Key Contributions

- Introduced the **attention mechanism** for neural machine translation, allowing the decoder to selectively focus on different parts of the source sentence at each generation step.
- Eliminated the information bottleneck of compressing the entire source sentence into a single fixed-length vector.
- Demonstrated significant improvements over the basic encoder-decoder model, especially on long sentences.
- Established the additive attention (Bahdanau attention) formulation that became one of the two canonical attention variants.

---

## Background & Motivation

The Seq2Seq model (Sutskever et al., 2014) encodes an entire source sentence into a single fixed-dimensional vector, which the decoder must use to generate the full translation. This creates an information bottleneck that particularly harms performance on long sentences. The authors hypothesized that allowing the model to automatically search for parts of the source sentence relevant to each target word would alleviate this problem, analogous to how human translators look back and forth across the source text.

---

## Method / Architecture

### Encoder: Bidirectional RNN

The encoder is a **bidirectional RNN** (BiRNN) that reads the source sentence in both forward and backward directions:

**Forward hidden states:**

$$\overrightarrow{h}_j = f(\overrightarrow{h}_{j-1}, x_j)$$

**Backward hidden states:**

$$\overleftarrow{h}_j = f(\overleftarrow{h}_{j+1}, x_j)$$

The annotation for each source position $j$ is the concatenation:

$$h_j = [\overrightarrow{h}_j; \overleftarrow{h}_j]$$

This captures both preceding and following context for word $j$.

### Attention Mechanism (Additive / Bahdanau Attention)

At each decoder time step $i$, the model computes an **alignment score** between the current decoder state $s_{i-1}$ and each encoder annotation $h_j$:

$$e_{ij} = a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

where $W_a$, $U_a$, and $v_a$ are learned parameters.

The alignment scores are normalized via softmax to produce **attention weights**:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

The **context vector** for time step $i$ is the weighted sum of encoder annotations:

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

### Decoder

The decoder is a GRU-based RNN. At each step $i$, the decoder state is updated as:

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

The conditional probability of the next target word is:

$$p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i)$$

where $g$ is a feedforward neural network with a softmax output layer.

### Key Difference from Seq2Seq

| Aspect | Seq2Seq (Sutskever) | Bahdanau Attention |
|:-------|:-------------------:|:------------------:|
| Context vector | Single fixed $\mathbf{c}$ for all steps | Dynamic $c_i$ per step |
| Encoder | Unidirectional LSTM | Bidirectional GRU |
| Source access | Only via bottleneck | Direct weighted access |
| Long sentences | Performance degrades | Robust |

---

## Key Results

### English-to-French Translation (WMT' 14 benchmarks)

| Model | BLEU |
|:------|:----:|
| RNNenc-50 (basic encoder-decoder) | 26.75 |
| RNNsearch-30 (attention, max length 30) | 31.92 |
| RNNsearch-50 (attention, max length 50) | **34.16** |
| Moses (phrase-based SMT baseline) | 33.30 |

### Performance by Sentence Length

The attention model maintained strong BLEU scores even for sentences longer than 30 words, whereas the basic encoder-decoder model's performance dropped sharply beyond length 20. This confirmed the hypothesis that the fixed-length bottleneck was the primary cause of degradation on long sequences.

### Attention Alignment Visualization

The learned attention weights closely corresponded to a monotonic (but not strictly diagonal) alignment between French and English words, with appropriate deviations for word reordering -- demonstrating that the model learned linguistically meaningful alignments without explicit supervision.

---

## Impact & Legacy

- **Universally adopted:** Attention became a standard component in virtually every neural sequence model from 2015 onward, far beyond machine translation.
- **Enabled the Transformer:** The self-attention mechanism in "Attention Is All You Need" (2017) directly evolved from this work.
- **Extended to vision:** Attention was adapted for image captioning (Xu et al., 2015), visual question answering, and other multimodal tasks.
- **Two attention paradigms:** Bahdanau (additive) attention and Luong (multiplicative) attention became the two standard formulations.
- **Interpretability:** Attention weights provided a form of model interpretability, showing which source tokens the model focused on for each prediction.
- **One of the most cited ML papers:** Accumulated tens of thousands of citations and fundamentally reshaped the field.

---

## Key Takeaways

1. The attention mechanism solves the fixed-length bottleneck problem by allowing the decoder to dynamically attend to relevant parts of the source at each generation step.
2. Additive attention computes alignment scores using a learned feedforward network: $e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$.
3. Bidirectional encoding is critical for providing rich source representations that capture both left and right context.
4. Attention weights serve as soft alignments and provide interpretable visualizations of what the model has learned.
5. This mechanism is not specific to translation -- it is a general-purpose technique for relating elements across sequences, which became the basis for modern deep learning.
