# Neural Programmer-Interpreters

**Authors:** Scott Reed, Nando de Freitas
**Year:** 2016
**Venue:** ICLR 2016
**Institution:** Google DeepMind

---

## Key Contributions

- Proposed the **Neural Programmer-Interpreter (NPI)**, a recurrent neural network that learns to represent and execute programs by observing input-output execution traces.
- Demonstrated that a single NPI model can learn multiple algorithms (addition, sorting, canonicalizing 3D car models) using **compositional, recursive program representations**.
- Introduced a key-value program memory that allows the NPI to call learned subroutines, enabling program compositionality and transfer.
- Showed strong generalization to inputs substantially longer than those seen during training (e.g., trained on sorting 3 numbers, generalizes to 20+).
- Provided a framework bridging neural networks and classical programming, combining the learning capabilities of the former with the compositional structure of the latter.

---

## Background & Motivation

Classical programs have desirable properties: compositionality (programs call subroutines), interpretability (each step has clear semantics), and generalization (a sorting algorithm works for any input size). Neural networks, by contrast, learn from data but often lack these properties.

The NPI aims to combine both worlds: learn programs from data (execution traces) while maintaining compositional, recursive structure that enables generalization. The key question is whether a neural network can learn to **interpret and execute** programs composed of reusable subroutines, where each subroutine is itself a learned neural program.

---

## Method / Architecture

### Overall Architecture

The NPI consists of four main components:

1. **Task-specific encoder** $f_{\text{enc}}$: Encodes the current environment state into a fixed-length vector.
2. **Core LSTM** $f_{\text{lstm}}$: Maintains the execution state across time steps.
3. **Program memory** $M$: A key-value store mapping program IDs to trainable embedding vectors.
4. **Output decoders**: Predict the next action (program to call, arguments, and whether to stop).

### Computation at Each Time Step

At time step $t$:

**Step 1: Encode environment.**

$$s_t = f_{\text{enc}}(\text{env}_t)$$

**Step 2: Retrieve program embedding.**

Given the current program ID $i$, look up its embedding:

$$p_i = M[i]$$

**Step 3: Update core LSTM.**

$$h_t = f_{\text{lstm}}(s_t, p_i, a_{t-1}, h_{t-1})$$

where $a_{t-1}$ is the previous argument encoding and $h_{t-1}$ is the previous hidden state.

**Step 4: Decode outputs.**

The LSTM hidden state $h_t$ is decoded into three outputs:

- **End-of-program probability:**

$$r_t = \sigma(W_r h_t + b_r)$$

- **Next program ID:**

$$\text{prog}_t = \arg\max_j \text{softmax}(W_{\text{prog}} h_t \cdot M[j]^\top)$$

- **Arguments:**

$$a_t = W_a h_t + b_a$$

### Program Execution (Recursive)

NPI execution is inherently recursive. When the NPI decides to call a subprogram, it:

1. Pushes the current program state onto a stack.
2. Begins executing the called subprogram (which updates the environment and LSTM state).
3. When the subprogram signals $r_t > 0.5$ (end), pops the stack and returns to the calling program.

This mimics function calls in classical programming.

### Architecture Diagram (Conceptual)

| Component | Input | Output |
|:----------|:------|:-------|
| Encoder $f_{\text{enc}}$ | Raw environment state | State vector $s_t$ |
| Program Memory $M$ | Program ID $i$ | Embedding $p_i$ |
| Core LSTM $f_{\text{lstm}}$ | $s_t, p_i, a_{t-1}, h_{t-1}$ | Hidden state $h_t$ |
| End Decoder | $h_t$ | Stop probability $r_t$ |
| Program Decoder | $h_t, M$ | Next program ID |
| Argument Decoder | $h_t$ | Arguments $a_t$ |

### Training

The NPI is trained via **supervised learning on execution traces**. Each trace is a sequence of (environment state, program ID, arguments, return signal) tuples that demonstrate the correct execution of a program.

The training loss is:

$$\mathcal{L} = \sum_t \left[ \mathcal{L}_{\text{prog}}(t) + \mathcal{L}_{\text{arg}}(t) + \mathcal{L}_{\text{end}}(t) \right]$$

where:

- $\mathcal{L}_{\text{prog}}(t)$: Cross-entropy loss for predicting the correct subprogram.
- $\mathcal{L}_{\text{arg}}(t)$: Loss for predicting correct arguments.
- $\mathcal{L}_{\text{end}}(t)$: Binary cross-entropy for the end-of-program signal.

### Curriculum Learning

Training uses a curriculum that starts with the simplest subroutines and gradually introduces more complex programs that compose them:

1. Train low-level operations (e.g., MOVE_PTR, WRITE).
2. Train mid-level programs that call low-level operations (e.g., ADD1 calls MOVE_PTR and WRITE).
3. Train high-level programs that compose mid-level ones (e.g., ADDITION calls ADD1 and CARRY).

---

## Key Results

### Addition

Train on adding numbers with up to a certain number of digits, test on longer numbers.

| Train Digits | Test Digits | Accuracy |
|:------------|:-----------|:--------:|
| Up to 3 | 3 | 100% |
| Up to 3 | 10 | 100% |
| Up to 3 | 20 | **100%** |

Perfect generalization to problem sizes 6-7x larger than training.

### Sorting (Bubble Sort)

| Train Size | Test Size | Accuracy |
|:----------|:---------|:--------:|
| Up to 3 | 3 | 100% |
| Up to 3 | 10 | 100% |
| Up to 3 | 20 | **100%** |

The NPI learned a correct bubble sort implementation that generalized to much longer arrays.

### 3D Car Canonicalization

Given a 3D model of a car in an arbitrary orientation, rotate it to a canonical pose.

| Task | Success Rate |
|:-----|:-----------:|
| Single rotation axis | 100% |
| Multiple rotation axes | **100%** |

### Compositionality

A key result was that subroutines learned for one task (e.g., MOVE_PTR) could be **reused** by other tasks without retraining, demonstrating genuine compositional program structure.

### Comparison with Non-Compositional Baselines

| Model | Generalization Beyond Training Length |
|:------|:-------------------------------------:|
| Standard LSTM (flat) | Poor |
| NPI without recursion | Moderate |
| **NPI with recursion** | **Strong** |

---

## Impact & Legacy

- **Program synthesis and induction:** NPI influenced a broad line of work on neural program synthesis, including Neural Turing Machines extensions, differentiable interpreters, and AlphaCode.
- **Compositional generalization:** Demonstrated that giving neural networks explicit compositional structure (subroutines, recursion) dramatically improves generalization, a theme that remains central in AI research.
- **Learning from demonstrations:** The execution-trace training paradigm is a precursor to learning from demonstrations / imitation learning approaches in robotics and agent AI.
- **Neural-symbolic integration:** NPI sits at the intersection of neural and symbolic AI, offering a framework where neural networks execute symbolic-style programs -- an idea that has gained renewed interest.
- **Curriculum learning:** The hierarchical curriculum (low-level to high-level programs) influenced training strategies for complex tasks.

---

## Key Takeaways

1. The NPI combines neural learning with classical program structure: a single LSTM core learns to execute multiple programs composed of reusable subroutines.
2. Compositional, recursive program execution enables remarkable generalization -- models trained on tiny inputs (3-digit addition) generalize perfectly to much larger inputs (20+ digits).
3. The program memory (key-value store of program embeddings) allows the NPI to flexibly select which subroutine to call, acting as a learned program counter.
4. Curriculum learning over program complexity is essential: simple subroutines must be mastered before complex programs that compose them.
5. The work highlights that combining the structure of programming with the flexibility of neural networks yields systems that are both learnable and generalizable -- a design principle that remains highly relevant.
