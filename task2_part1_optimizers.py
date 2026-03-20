"""
═══════════════════════════════════════════════════════════════
AIML FFCS — Task 2, Part 1
Implement & Compare Training Optimizers (From Scratch)
Network: 2 inputs → 4 hidden → 1 output
Dataset: XOR
Optimizers: Vanilla GD | Momentum | RMSProp | Adam
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── XOR Dataset ─────────────────────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]],         dtype=float)

# ── Activation ──────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# ── Weight Initialization ────────────────────────────────────
def init_weights(H=4):
    """Xavier-ish initialisation for a 2 → H → 1 network."""
    np.random.seed(42)
    W1 = np.random.randn(2, H) * 1.0
    b1 = np.zeros((1, H))
    W2 = np.random.randn(H, 1) * 1.0
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# ── Forward Pass ────────────────────────────────────────────
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1          # (m, H)
    A1 = sigmoid(Z1)           # hidden activation
    Z2 = A1 @ W2 + b2          # (m, 1)
    A2 = sigmoid(Z2)           # output probability
    return A2, (Z1, A1, Z2, A2)

# ── Backward Pass ────────────────────────────────────────────
def backward(X, y, W2, cache):
    Z1, A1, Z2, A2 = cache
    m = X.shape[0]
    dZ2 = (A2 - y) * sigmoid_deriv(Z2)      # output delta
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = (dZ2 @ W2.T) * sigmoid_deriv(Z1)  # hidden delta
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# ── Loss ────────────────────────────────────────────────────
def bce_loss(A2, y):
    """Binary Cross-Entropy loss."""
    return -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER 1 — Vanilla Gradient Descent
#   θ ← θ - α · ∇θ
#   Simplest update: move in the direction opposite to the gradient.
#   Can oscillate in ravines and is sensitive to learning rate choice.
# ═══════════════════════════════════════════════════════════════
def train_vanilla_gd(lr=1.0, epochs=500):
    W1, b1, W2, b2 = init_weights()
    losses = []
    for _ in range(epochs):
        A2, cache = forward(X, W1, b1, W2, b2)
        losses.append(bce_loss(A2, y))
        dW1, db1, dW2, db2 = backward(X, y, W2, cache)
        # ── plain gradient step ──
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return losses


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER 2 — Gradient Descent with Momentum
#   v ← β·v + (1-β)·∇θ
#   θ ← θ - α·v
#   Accumulates a velocity vector in the direction of persistent
#   gradients, dampening oscillations and speeding up convergence.
# ═══════════════════════════════════════════════════════════════
def train_momentum(lr=0.8, beta=0.9, epochs=500):
    W1, b1, W2, b2 = init_weights()
    # Velocity buffers (first moment, no bias correction)
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    losses = []
    for _ in range(epochs):
        A2, cache = forward(X, W1, b1, W2, b2)
        losses.append(bce_loss(A2, y))
        dW1, db1, dW2, db2 = backward(X, y, W2, cache)
        # ── exponential moving average of gradients ──
        vW1 = beta * vW1 + (1 - beta) * dW1
        vb1 = beta * vb1 + (1 - beta) * db1
        vW2 = beta * vW2 + (1 - beta) * dW2
        vb2 = beta * vb2 + (1 - beta) * db2
        W1 -= lr * vW1
        b1 -= lr * vb1
        W2 -= lr * vW2
        b2 -= lr * vb2
    return losses


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER 3 — RMSProp
#   s ← β·s + (1-β)·∇θ²
#   θ ← θ - α · ∇θ / √(s + ε)
#   Maintains a running average of squared gradients to give each
#   parameter its own effective learning rate. Great for non-stationary
#   objectives and recurrent networks.
# ═══════════════════════════════════════════════════════════════
def train_rmsprop(lr=0.05, beta=0.9, eps=1e-8, epochs=500):
    W1, b1, W2, b2 = init_weights()
    # Second-moment (squared gradient) accumulators
    sW1 = np.ones_like(W1) * 0.1
    sb1 = np.ones_like(b1) * 0.1
    sW2 = np.ones_like(W2) * 0.1
    sb2 = np.ones_like(b2) * 0.1
    losses = []
    for _ in range(epochs):
        A2, cache = forward(X, W1, b1, W2, b2)
        losses.append(bce_loss(A2, y))
        dW1, db1, dW2, db2 = backward(X, y, W2, cache)
        # ── running mean of squared gradients ──
        sW1 = beta * sW1 + (1 - beta) * dW1 ** 2
        sb1 = beta * sb1 + (1 - beta) * db1 ** 2
        sW2 = beta * sW2 + (1 - beta) * dW2 ** 2
        sb2 = beta * sb2 + (1 - beta) * db2 ** 2
        # ── adaptive learning rate ──
        W1 -= lr * dW1 / (np.sqrt(sW1) + eps)
        b1 -= lr * db1 / (np.sqrt(sb1) + eps)
        W2 -= lr * dW2 / (np.sqrt(sW2) + eps)
        b2 -= lr * db2 / (np.sqrt(sb2) + eps)
    return losses


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER 4 — Adam (Adaptive Moment Estimation)
#   m ← β1·m + (1-β1)·∇θ          ← first moment (momentum)
#   v ← β2·v + (1-β2)·∇θ²         ← second moment (RMSProp)
#   m̂ = m/(1-β1^t)                 ← bias-corrected first moment
#   v̂ = v/(1-β2^t)                 ← bias-corrected second moment
#   θ ← θ - α · m̂ / (√v̂ + ε)
#   Combines momentum and RMSProp with bias correction.
#   Generally the best default optimizer.
# ═══════════════════════════════════════════════════════════════
def train_adam(lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, epochs=500):
    W1, b1, W2, b2 = init_weights()
    # First-moment accumulators
    mW1 = np.zeros_like(W1); mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); mb2 = np.zeros_like(b2)
    # Second-moment accumulators
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
    losses = []
    for t in range(1, epochs + 1):
        A2, cache = forward(X, W1, b1, W2, b2)
        losses.append(bce_loss(A2, y))
        dW1, db1, dW2, db2 = backward(X, y, W2, cache)
        # ── first & second moment updates ──
        mW1 = beta1*mW1 + (1-beta1)*dW1;  mb1 = beta1*mb1 + (1-beta1)*db1
        mW2 = beta1*mW2 + (1-beta1)*dW2;  mb2 = beta1*mb2 + (1-beta1)*db2
        vW1 = beta2*vW1 + (1-beta2)*dW1**2; vb1 = beta2*vb1 + (1-beta2)*db1**2
        vW2 = beta2*vW2 + (1-beta2)*dW2**2; vb2 = beta2*vb2 + (1-beta2)*db2**2
        # ── bias correction ──
        mW1_c = mW1/(1-beta1**t); mb1_c = mb1/(1-beta1**t)
        mW2_c = mW2/(1-beta1**t); mb2_c = mb2/(1-beta1**t)
        vW1_c = vW1/(1-beta2**t); vb1_c = vb1/(1-beta2**t)
        vW2_c = vW2/(1-beta2**t); vb2_c = vb2/(1-beta2**t)
        # ── parameter update ──
        W1 -= lr * mW1_c / (np.sqrt(vW1_c) + eps)
        b1 -= lr * mb1_c / (np.sqrt(vb1_c) + eps)
        W2 -= lr * mW2_c / (np.sqrt(vW2_c) + eps)
        b2 -= lr * mb2_c / (np.sqrt(vb2_c) + eps)
    return losses


# ── Train ────────────────────────────────────────────────────
EPOCHS = 500
print("Training optimizers on XOR...\n")
losses_gd   = train_vanilla_gd(lr=1.0,  epochs=EPOCHS)
losses_mom  = train_momentum(  lr=0.8,  beta=0.9,          epochs=EPOCHS)
losses_rms  = train_rmsprop(   lr=0.05, beta=0.9,           epochs=EPOCHS)
losses_adam = train_adam(      lr=0.05, beta1=0.9, beta2=0.999, epochs=EPOCHS)

all_losses = {
    "Vanilla Gradient Descent": (losses_gd,   "#e74c3c"),
    "Momentum (beta=0.9)":      (losses_mom,  "#f39c12"),
    "RMSProp":                  (losses_rms,  "#27ae60"),
    "Adam":                     (losses_adam, "#3498db"),
}

# ── Results table ────────────────────────────────────────────
print(f"{'Optimizer':<28} {'@Ep200':>8} {'Final':>8}")
print("-" * 48)
for name, (losses, _) in all_losses.items():
    print(f"{name:<28} {losses[199]:>8.4f} {losses[-1]:>8.4f}")

# ── Individual subplot figure ────────────────────────────────
BG = '#0d0d1f'; PN = '#141428'; GR = '#2a2a4a'

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35)

for idx, (name, (losses, color)) in enumerate(all_losses.items()):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor(PN)
    ep = range(1, EPOCHS + 1)
    ax.plot(ep, losses, color=color, linewidth=2.2, zorder=3)
    ax.fill_between(ep, losses, alpha=0.12, color=color, zorder=2)
    ax.axvline(200, color='white', alpha=0.2, linestyle=':', linewidth=1)
    ax.text(202, max(losses) * 0.97, 'Ep 200', color='#888', fontsize=7)
    ax.text(0.97, 0.95, f"Final: {losses[-1]:.4f}",
            transform=ax.transAxes, ha='right', va='top',
            color=color, fontsize=10, fontweight='bold')
    ax.set_title(name, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch",       color='#888', fontsize=9)
    ax.set_ylabel("BCE Loss",    color='#888', fontsize=9)
    ax.tick_params(colors='#888', labelsize=8)
    for s in ['top', 'right']:  ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color(GR)
    ax.grid(True, alpha=0.12, linestyle='--', color='#4444aa')

fig.suptitle(
    "Loss vs Epoch — Training Optimizer Comparison\nXOR Dataset | Network: 2 → 4 → 1 | 500 Epochs",
    color='white', fontsize=14, fontweight='bold', y=1.02)
plt.savefig("task2_part1_optimizer_comparison.png",
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Overlay comparison figure ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 6))
fig2.patch.set_facecolor(BG); ax2.set_facecolor(PN)
for name, (losses, color) in all_losses.items():
    ax2.plot(range(1, EPOCHS + 1), losses,
             label=f"{name}  (final={losses[-1]:.4f})",
             color=color, linewidth=2.2)
ax2.set_title("All Optimizers — Overlay Comparison | XOR Dataset",
              color='white', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel("Epoch",    color='#aaa', fontsize=11)
ax2.set_ylabel("BCE Loss", color='#aaa', fontsize=11)
ax2.tick_params(colors='#aaa')
for s in ['top', 'right']:  ax2.spines[s].set_visible(False)
for s in ['bottom', 'left']: ax2.spines[s].set_color(GR)
ax2.legend(facecolor='#1a1a35', edgecolor=GR, labelcolor='white',
           fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.15, linestyle='--', color='#4444aa')
ax2.axvline(200, color='white', alpha=0.25, linestyle=':', linewidth=1.2)
plt.tight_layout()
plt.savefig("task2_part1_overlay.png",
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("\nPlots saved: task2_part1_optimizer_comparison.png  |  task2_part1_overlay.png")

# ════════════════════════════════════════════════════════════
#  BRIEF COMPARISON — Convergence Speed & Stability
# ════════════════════════════════════════════════════════════
print("""
┌─────────────────────────────────────────────────────────────────┐
│              OPTIMIZER COMPARISON — XOR TASK                    │
├─────────────────┬───────────────────────────────────────────────┤
│ Vanilla GD      │ Simplest. Slow. Sensitive to lr. Often stalls │
│                 │ in saddle points on non-convex surfaces.       │
├─────────────────┼───────────────────────────────────────────────┤
│ Momentum        │ Faster than GD via velocity accumulation.      │
│                 │ Less oscillation. Needs beta tuning.           │
├─────────────────┼───────────────────────────────────────────────┤
│ RMSProp         │ Adaptive lr per parameter. Handles sparse      │
│                 │ gradients well. Best for XOR here.             │
├─────────────────┼───────────────────────────────────────────────┤
│ Adam            │ Combines Momentum + RMSProp + bias correction. │
│                 │ Robust, fast, usually best default choice.     │
└─────────────────┴───────────────────────────────────────────────┘
""")
