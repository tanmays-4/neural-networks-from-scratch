# 🧠 Neural Networks From Scratch

> Pure NumPy implementations of neural networks, optimizers, and regularization techniques — no TensorFlow, no PyTorch, no shortcuts.

Built as part of the **AIML FFCS Club Tasks** at VIT Chennai.

---

## 📁 Repository Structure

```
neural-networks-from-scratch/
├── README.md
├── task1/
│   └── tiny_neural_network.py        # Feedforward NN on XOR/OR from scratch
├── task2/
│   ├── task2_part1_optimizers.py     # Optimizer comparison (GD, Momentum, RMSProp, Adam)
│   ├── task2_part2_regularization.py # Overfitting & Regularization (L2, Dropout, Early Stop)
│   └── plots/
│       ├── optimizer_comparison.png
│       ├── overlay.png
│       ├── regularization.png
│       └── gap_comparison.png
```

---

## 📌 Task 1 — Build & Train a Tiny Neural Network

**Architecture:** `2 inputs → 2 hidden neurons → 1 output`  
**Dataset:** XOR / OR  
**Training:** 200+ epochs, Sigmoid activation, Binary Cross-Entropy loss

### What's implemented
- Forward pass (matrix multiplication + sigmoid)
- Backpropagation (chain rule, manually derived gradients)
- Vanilla gradient descent weight updates
- Loss vs Epoch plot

---

## 📌 Task 2 — Optimizers & Regularization

### Part 1 — Implement & Compare Training Optimizers

**Architecture:** `2 → 4 → 1`  &nbsp;|&nbsp; **Dataset:** XOR  &nbsp;|&nbsp; **Epochs:** 500

All four optimizers implemented **from scratch in NumPy**:

| Optimizer | Core Idea | Notes |
|---|---|---|
| **Vanilla Gradient Descent** | `θ ← θ - α·∇θ` | Simple, slow, sensitive to lr |
| **Momentum** | EMA of gradients as velocity | Reduces oscillations |
| **RMSProp** | Divides lr by running mean of squared gradients | Adaptive per-parameter lr |
| **Adam** | Momentum + RMSProp + bias correction | Best overall convergence |

#### Comparison Summary

- **Vanilla GD** — Stalls on XOR (non-convex loss surface). High final loss.
- **Momentum** — Slightly faster convergence but still struggles without adaptive lr.
- **RMSProp** — Converges fastest here due to adaptive per-parameter scaling.
- **Adam** — Close second; more robust across different problems in general.

---

### Part 2 — Overfitting & Regularization: Build, Break, Fix

**Architecture:** `2 → 32 → 1` (intentionally large)  
**Dataset:** Noisy XOR | **Train:** 20 samples | **Val:** 500 samples | **Epochs:** 2000

The large network + tiny dataset combination forces the model to **memorise** training data, clearly demonstrating overfitting.

All regularization techniques implemented **from scratch**:

| Technique | How It Works | Effect |
|---|---|---|
| **No Regularization** | Baseline — plain training | Large train/val gap |
| **L2 (Weight Decay)** | Adds `λ·‖W‖²` penalty; shrinks weights toward zero | Gap significantly reduced |
| **Dropout** | Random neuron masking per batch (inverted dropout) | Forces distributed representations |
| **Early Stopping** | Monitors val loss; restores best checkpoint | Stops before overfitting peaks |

---

## 🚀 How to Run

**Requirements:** Python 3.x, NumPy, Matplotlib

```bash
pip install numpy matplotlib
```

**Task 1:**
```bash
python task1/tiny_neural_network.py
```

**Task 2 — Optimizers:**
```bash
python task2/task2_part1_optimizers.py
```

**Task 2 — Regularization:**
```bash
python task2/task2_part2_regularization.py
```

Each script trains the network, prints a results table to the terminal, and saves plots to the current directory.

---

## 🔑 Key Concepts

- **Backpropagation** — Chain rule applied layer by layer to compute gradients
- **Sigmoid Activation** — `σ(z) = 1 / (1 + e^{-z})`, used for binary classification
- **Binary Cross-Entropy** — `L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`
- **Overfitting** — When train loss → 0 but val loss stays high (model memorised data)
- **Generalization Gap** — `|Val Loss - Train Loss|`, the core metric for measuring overfit

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)

**No ML libraries used.** Everything is built on raw NumPy and math.

---

## 👤 Author

**Tanmay Sharma**  
MIC CLUB _ AIML dept — VIT Chennai

---

*Made with math and no shortcuts.*
