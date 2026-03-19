# Learning to Learn by Gradient Descent by Gradient Descent

**Authors:** Marcin Andrychowicz, Misha Denil, Sergio Gomez Colmenarejo, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, Nando de Freitas
**Year:** 2016
**Venue:** NeurIPS 2016
**Institution:** Google DeepMind

---

## Key Contributions

- Proposed replacing hand-designed optimization algorithms (SGD, Adam, RMSProp) with a **learned optimizer** parameterized as an LSTM that is itself trained by gradient descent.
- Demonstrated that the learned optimizer can outperform standard optimizers on the tasks it was trained on, and can generalize to related but unseen optimization problems.
- Introduced a clean and elegant meta-learning formulation where the "outer loop" trains the optimizer and the "inner loop" uses it to optimize a task-specific model.
- Showed that the learned optimizer captures adaptive learning rate schedules, momentum-like behavior, and task-specific optimization strategies automatically.
- Provided a principled framework for **meta-learning** that influenced a broad range of subsequent work.

---

## Background & Motivation

Optimization algorithms like SGD, Adam, and RMSProp are designed by human experts and apply the same update rules regardless of the specific loss landscape. Key observations motivating this work:

1. Different problems have different loss landscape properties; a single hand-designed optimizer is unlikely to be optimal for all of them.
2. An optimizer that **learns** from experience across many optimization problems could potentially discover update rules tailored to specific problem families.
3. The update rule of an optimizer (mapping gradients to parameter updates) can itself be parameterized by a neural network and trained via backpropagation.

The title captures the recursive idea perfectly: "Learning to learn [the optimizer] by gradient descent [the meta-training method] by gradient descent [the optimizer being learned]."

---

## Method / Architecture

### Standard Optimization (Recap)

A traditional optimizer updates parameters $\theta$ at step $t$:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta f(\theta_t)$$

where $f$ is the loss function and $\alpha$ is the learning rate.

More sophisticated optimizers like Adam maintain additional state (momentum, second moments) and apply complex update rules.

### Learned Optimizer

Replace the hand-designed update rule with a learned function $g$ parameterized by $\phi$:

$$\theta_{t+1} = \theta_t + g_t(\nabla_t, \phi)$$

where $g_t$ is an LSTM that takes the gradient $\nabla_t = \nabla_\theta f(\theta_t)$ as input and outputs the parameter update.

### Meta-Learning Objective

The meta-learner (optimizer) is trained to minimize the expected loss over a distribution of optimization problems:

$$\mathcal{L}(\phi) = \mathbb{E}_f \left[ \sum_{t=1}^{T} w_t f(\theta_t) \right]$$

where:
- $f$ is sampled from a distribution of tasks.
- $\theta_t$ is the optimizee's parameters at step $t$, produced by the learned optimizer.
- $w_t \geq 0$ are weights (typically $w_t = 1$ for all $t$, or only the final step).

### Computing Meta-Gradients

The key technical challenge is differentiating through the entire optimization trajectory. The gradient of the meta-loss with respect to the optimizer parameters $\phi$ is:

$$\frac{\partial \mathcal{L}(\phi)}{\partial \phi} = \sum_{t=1}^{T} w_t \frac{\partial f(\theta_t)}{\partial \theta_t} \cdot \frac{\partial \theta_t}{\partial \phi}$$

Using the chain rule through the unrolled optimization:

$$\frac{\partial \theta_{t+1}}{\partial \phi} = \frac{\partial \theta_t}{\partial \phi} + \frac{\partial g_t}{\partial \phi} + \frac{\partial g_t}{\partial \nabla_t} \cdot \frac{\partial^2 f(\theta_t)}{\partial \theta_t \partial \phi}$$

In practice, higher-order gradient terms are often truncated to keep computation manageable.

### Coordinatewise LSTM Optimizer

To handle high-dimensional parameter spaces, the authors use a **coordinatewise** architecture:

- A small LSTM operates independently on each coordinate (parameter) of the optimizee.
- The same LSTM (shared weights) processes the gradient of each parameter.
- Each LSTM maintains its own hidden state, capturing per-parameter optimization history.

For parameter $\theta_i$ at step $t$:

$$\Delta \theta_{i,t}, h_{i,t} = \text{LSTM}(\nabla_{\theta_i} f(\theta_t), h_{i,t-1}; \phi)$$

$$\theta_{i,t+1} = \theta_{i,t} + \Delta \theta_{i,t}$$

This makes the approach scalable: the number of LSTM parameters is independent of the optimizee's dimensionality.

### Preprocessing Gradients

Raw gradients span many orders of magnitude. The paper preprocesses them using a log-scale transformation:

$$\text{preprocess}(\nabla) = \begin{cases} \left(\frac{\log(|\nabla|)}{p}, \text{sign}(\nabla)\right) & \text{if } |\nabla| \geq e^{-p} \\ (-1, e^p \nabla) & \text{otherwise} \end{cases}$$

where $p > 0$ is a hyperparameter (typically $p = 10$). This provides numerical stability and puts all gradients on a comparable scale.

### Training Procedure

| Step | Description |
|:-----|:-----------|
| 1 | Sample an optimization problem $f$ from the task distribution |
| 2 | Initialize optimizee parameters $\theta_0$ randomly |
| 3 | Run the learned optimizer for $T$ steps to produce $\theta_1, \ldots, \theta_T$ |
| 4 | Compute meta-loss $\mathcal{L}(\phi) = \sum_t w_t f(\theta_t)$ |
| 5 | Backpropagate through the unrolled computation to get $\partial \mathcal{L}/\partial \phi$ |
| 6 | Update optimizer parameters $\phi$ using Adam |

---

## Key Results

### Quadratic Functions

Optimizing $f(\theta) = \| W\theta - y \|^2$ for randomly sampled $W, y$.

| Optimizer | Final Loss (after 100 steps) |
|:----------|:----------------------------:|
| SGD (tuned) | $10^{-1}$ |
| RMSProp (tuned) | $10^{-2}$ |
| Adam (tuned) | $10^{-3}$ |
| **Learned LSTM optimizer** | $\mathbf{10^{-5}}$ |

The learned optimizer converged orders of magnitude faster than all hand-designed baselines.

### Training Small Neural Networks on MNIST

| Optimizer | Final Loss | Steps to Converge |
|:----------|:----------:|:-----------------:|
| SGD | 0.28 | >10000 |
| RMSProp | 0.18 | ~5000 |
| Adam | 0.18 | ~5000 |
| **Learned optimizer** | **0.14** | **~3000** |

### Generalization: Training on MNIST, Testing on CIFAR-10

| Optimizer | CIFAR-10 Final Loss |
|:----------|:-------------------:|
| Adam | 2.0 |
| **Learned optimizer (trained on MNIST)** | **1.7** |

The learned optimizer generalized to a different dataset and image distribution.

### Generalization Across Architectures

An optimizer trained on small MLPs was tested on:

| Test Architecture | Performance vs. Adam |
|:-----------------|:-------------------:|
| Larger MLPs | Comparable or better |
| Different activation functions | Comparable |
| CNNs (small) | Slightly better |

### Neural Art (Style Transfer)

The learned optimizer was applied to neural style transfer optimization, producing visually appealing results faster than standard SGD with momentum.

---

## Impact & Legacy

- **Meta-learning renaissance:** This paper was a catalyst for the explosion of meta-learning research from 2016 onward, alongside MAML (Finn et al., 2017) and Matching Networks.
- **Learned optimizers research line:** Directly spawned subsequent work on learned optimizers, including VeLO (Google, 2022) and other large-scale optimizer learning efforts.
- **Differentiating through optimization:** Established the technique of backpropagating through unrolled optimization as a practical meta-learning tool, influencing hyperparameter optimization and neural architecture search.
- **Optimizer design:** The insight that optimization algorithms can be learned rather than designed by hand influenced how the community thinks about optimization -- even hand-designed optimizers (like AdaFactor, LAMB) drew inspiration from properties of learned optimizers.
- **Foundation for RLHF and alignment:** The meta-learning framework of "training a model to guide training of another model" is conceptually related to reward modeling and RLHF approaches used in modern LLM alignment.

---

## Key Takeaways

1. Optimization algorithms can be **learned rather than hand-designed**: an LSTM-based optimizer trained by gradient descent can outperform SGD, Adam, and RMSProp on tasks within its training distribution.
2. The coordinatewise LSTM design makes the approach scalable: the same small LSTM processes each parameter independently, with the number of optimizer parameters independent of the optimizee size.
3. Gradient preprocessing (log-scale transformation) is crucial for numerical stability when feeding gradients into a neural network.
4. The learned optimizer shows meaningful **transfer**: an optimizer trained on one task family (e.g., MNIST) can generalize to related tasks (e.g., CIFAR-10) and different architectures.
5. The recursive elegance of the approach -- learning to optimize by optimization -- exemplifies meta-learning and has inspired a broad research direction on learning components of the machine learning pipeline itself.
