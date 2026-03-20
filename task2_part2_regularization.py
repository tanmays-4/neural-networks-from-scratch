"""
═══════════════════════════════════════════════════════════════
AIML FFCS — Task 2, Part 2
Overfitting & Regularization: Build, Break, Fix
Network: 2 inputs → 32 hidden → 1 output  (large, prone to overfit)
Dataset: XOR with Gaussian noise
Techniques: L2 Regularization | Dropout | Early Stopping
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── Dataset: Noisy XOR ───────────────────────────────────────
def make_noisy_xor(n: int, noise: float = 0.15, seed: int = 42):
    """
    XOR dataset with Gaussian noise.
    Small train set + large val set → forces overfitting to show.
    """
    np.random.seed(seed)
    corners  = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    labels   = np.array([0., 1., 1., 0.])
    idx      = np.random.randint(0, 4, n)
    X = corners[idx] + np.random.randn(n, 2) * noise
    y = labels[idx].reshape(-1, 1)
    return X, y

# 20 training samples → easy to memorize, hard to generalise
X_train, y_train = make_noisy_xor(20,  noise=0.05, seed=1)
# 500 val samples → reveals whether the model truly learned XOR
X_val,   y_val   = make_noisy_xor(500, noise=0.15, seed=2)

# ── Helpers ──────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def bce(pred, true):
    """Binary Cross-Entropy."""
    return -np.mean(true * np.log(pred + 1e-8) + (1 - true) * np.log(1 - pred + 1e-8))

def accuracy(pred, true):
    return float(np.mean((pred > 0.5) == true))

# ── Weight init ───────────────────────────────────────────────
H = 32  # large hidden layer — intentionally oversized to cause overfit

def init_weights(seed: int = 1):
    np.random.seed(seed)
    return (np.random.randn(2, H) * 0.5,  # W1
            np.zeros((1, H)),               # b1
            np.random.randn(H, 1) * 0.5,  # W2
            np.zeros((1, 1)))               # b2

# ── Forward pass (supports dropout mask) ────────────────────
def forward(X, W1, b1, W2, b2, mask=None):
    """
    mask: binary array shape (1, H) — applied to hidden layer.
          Used during training with Dropout. None = no dropout.
    """
    A1 = sigmoid(X @ W1 + b1)
    if mask is not None:
        A1 = A1 * mask          # zero out dropped neurons
    A2 = sigmoid(A1 @ W2 + b2)
    return A2, A1

# ── Backward pass (supports dropout mask) ───────────────────
def backward(X, y, A1, A2, W2, mask=None):
    m = len(X)
    dZ2 = (A2 - y) * A2 * (1 - A2)     # sigmoid derivative folded in
    dW2 = A1.T @ dZ2 / m
    db2 = dZ2.mean(axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    if mask is not None:
        dA1 = dA1 * mask                 # gradients also masked
    h   = sigmoid(X @ W2.T)             # NOTE: recompute hidden pre-act if needed
    # Use chain rule with known A1
    dZ1 = dA1 * A1 * (1 - A1)           # A1*(1-A1) = sigmoid'(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = dZ1.mean(axis=0, keepdims=True)
    return dW1, db1, dW2, db2


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Overfit (no regularization, large network)
#   The model memorises training data: train loss → 0,
#   but val loss plateaus or rises = classic overfitting.
# ════════════════════════════════════════════════════════════════
def train_overfit(lr=3.0, epochs=2000):
    W1, b1, W2, b2 = init_weights()
    tr_losses, val_losses = [], []
    for _ in range(epochs):
        A2, A1 = forward(X_train, W1, b1, W2, b2)
        tr_losses.append(bce(A2, y_train))
        dW1, db1, dW2, db2 = backward(X_train, y_train, A1, A2, W2)
        W1 -= lr*dW1;  b1 -= lr*db1
        W2 -= lr*dW2;  b2 -= lr*db2
        Av, _ = forward(X_val, W1, b1, W2, b2)
        val_losses.append(bce(Av, y_val))
    Af, _ = forward(X_val, W1, b1, W2, b2)
    return tr_losses, val_losses, accuracy(Af, y_val)


# ════════════════════════════════════════════════════════════════
# TECHNIQUE 1 — L2 Regularization (Weight Decay)
#   Adds a penalty term λ·Σw² to the loss.
#   Gradient of penalty is λ·w, so the update becomes:
#     W ← W - lr · (∇L + λ·W)
#   This shrinks weights toward zero → prevents large, sharp weights
#   that lead to memorisation.
# ════════════════════════════════════════════════════════════════
def train_l2(lr=3.0, lam=0.003, epochs=2000):
    W1, b1, W2, b2 = init_weights()
    tr_losses, val_losses = [], []
    for _ in range(epochs):
        A2, A1 = forward(X_train, W1, b1, W2, b2)
        # L2 penalty added to reported loss (for fair comparison)
        l2_pen = (lam / (2 * len(X_train))) * (np.sum(W1**2) + np.sum(W2**2))
        tr_losses.append(bce(A2, y_train) + l2_pen)
        dW1, db1, dW2, db2 = backward(X_train, y_train, A1, A2, W2)
        # Weight decay: gradient of ½λ‖w‖² is λ·w
        W1 -= lr * (dW1 + lam * W1)
        b1 -= lr * db1                   # biases not regularised
        W2 -= lr * (dW2 + lam * W2)
        b2 -= lr * db2
        Av, _ = forward(X_val, W1, b1, W2, b2)
        val_losses.append(bce(Av, y_val))
    Af, _ = forward(X_val, W1, b1, W2, b2)
    return tr_losses, val_losses, accuracy(Af, y_val)


# ════════════════════════════════════════════════════════════════
# TECHNIQUE 2 — Dropout
#   During each forward pass, randomly zero out neurons with
#   probability (1-keep_prob). Scale survivors by 1/keep_prob
#   (inverted dropout) so expected activation magnitude stays the same.
#   At inference: NO mask (full network), no scaling needed.
#   Forces the network to learn redundant, distributed representations.
# ════════════════════════════════════════════════════════════════
def train_dropout(lr=3.0, keep_prob=0.65, epochs=2000):
    W1, b1, W2, b2 = init_weights()
    tr_losses, val_losses = [], []
    for _ in range(epochs):
        # Generate binary mask (inverted dropout)
        mask = (np.random.rand(1, H) < keep_prob).astype(float) / keep_prob
        A2, A1 = forward(X_train, W1, b1, W2, b2, mask=mask)
        tr_losses.append(bce(A2, y_train))
        dW1, db1, dW2, db2 = backward(X_train, y_train, A1, A2, W2, mask=mask)
        W1 -= lr*dW1;  b1 -= lr*db1
        W2 -= lr*dW2;  b2 -= lr*db2
        # Validation: no dropout (full network inference)
        Av, _ = forward(X_val, W1, b1, W2, b2, mask=None)
        val_losses.append(bce(Av, y_val))
    Af, _ = forward(X_val, W1, b1, W2, b2)
    return tr_losses, val_losses, accuracy(Af, y_val)


# ════════════════════════════════════════════════════════════════
# TECHNIQUE 3 — Early Stopping
#   Monitor validation loss every epoch.
#   If val loss has not improved by > δ for `patience` epochs,
#   stop training and restore the best-seen weights.
#   No math changes to the update rule — just stops before overfit.
# ════════════════════════════════════════════════════════════════
def train_early_stopping(lr=3.0, patience=100, epochs=2000):
    W1, b1, W2, b2 = init_weights()
    tr_losses, val_losses = [], []
    best_val  = float('inf')
    best_weights = None
    no_improve   = 0
    stopped_at   = epochs
    for ep in range(epochs):
        A2, A1 = forward(X_train, W1, b1, W2, b2)
        tr_losses.append(bce(A2, y_train))
        dW1, db1, dW2, db2 = backward(X_train, y_train, A1, A2, W2)
        W1 -= lr*dW1;  b1 -= lr*db1
        W2 -= lr*dW2;  b2 -= lr*db2
        Av, _ = forward(X_val, W1, b1, W2, b2)
        v = bce(Av, y_val)
        val_losses.append(v)
        # ── early-stopping logic ──────────────────────────
        if v < best_val - 1e-4:
            best_val    = v
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stopped_at = ep + 1
                break
    # Restore best weights
    W1, b1, W2, b2 = best_weights
    Af, _ = forward(X_val, W1, b1, W2, b2)
    return tr_losses, val_losses, accuracy(Af, y_val), stopped_at


# ── Run all experiments ───────────────────────────────────────
print("Running experiments...\n")
tr_ov,  val_ov,  acc_ov         = train_overfit()
tr_l2,  val_l2,  acc_l2         = train_l2()
tr_dr,  val_dr,  acc_dr         = train_dropout()
tr_es,  val_es,  acc_es, ep_stop = train_early_stopping()

print(f"{'Technique':<22} {'Train':>8} {'Val':>8} {'Gap':>8} {'Acc':>6}")
print("-" * 58)
for nm, tl, vl, a in [("No Reg (Overfit)", tr_ov, val_ov, acc_ov),
                        ("L2 (lam=0.003)",  tr_l2, val_l2, acc_l2),
                        ("Dropout (p=0.65)", tr_dr, val_dr, acc_dr),
                        ("Early Stop",       tr_es, val_es, acc_es)]:
    gap = abs(vl[-1] - tl[-1])
    print(f"{nm:<22} {tl[-1]:>8.4f} {vl[-1]:>8.4f} {gap:>8.4f} {a:>6.3f}")

# ── Plotting ──────────────────────────────────────────────────
BG = '#0d0d1f'; PN = '#141428'; GR = '#2a2a4a'

cfgs = [
    ("No Regularization\n(Overfitting)",   tr_ov, val_ov, acc_ov, "#e74c3c", None),
    ("L2 Regularization\n(lambda=0.003)",  tr_l2, val_l2, acc_l2, "#f39c12", None),
    ("Dropout\n(keep_prob=0.65)",           tr_dr, val_dr, acc_dr, "#27ae60", None),
    (f"Early Stopping\n(patience=100, stopped@{ep_stop})",
                                            tr_es, val_es, acc_es, "#3498db", ep_stop),
]

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.33)

for i, (title, tl, vl, a, c, stop) in enumerate(cfgs):
    ax = fig.add_subplot(gs[i//2, i%2]); ax.set_facecolor(PN)
    ep = range(1, len(tl)+1)
    ax.plot(ep, tl, color=c, lw=2.2, label='Train Loss')
    ax.plot(ep, vl, color=c, lw=2.2, ls='--', alpha=0.7, label='Val Loss')
    ax.fill_between(ep, tl, vl, alpha=0.1, color=c)
    if stop:
        ax.axvline(stop, color='white', ls=':', lw=1.5, alpha=0.7)
        ax.text(stop+15, max(vl)*0.92, f'stop\n@{stop}', color='#ccc', fontsize=7)
    g = abs(vl[-1] - tl[-1])
    ax.text(0.98, 0.97,
            f"Train: {tl[-1]:.4f}\nVal:   {vl[-1]:.4f}\nGap:   {g:.4f}\nAcc:   {a:.3f}",
            transform=ax.transAxes, ha='right', va='top',
            color='white', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='#1e1e3a', ec=c, alpha=0.92))
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch",    color='#888', fontsize=9)
    ax.set_ylabel("BCE Loss", color='#888', fontsize=9)
    ax.tick_params(colors='#888', labelsize=8)
    for s in ['top', 'right']:   ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color(GR)
    ax.grid(True, alpha=0.12, ls='--', color='#4444aa')
    ax.legend(facecolor='#1a1a35', edgecolor=GR, labelcolor='white', fontsize=9)

fig.suptitle(
    "Overfitting & Regularization — Build, Break, Fix\n"
    "Noisy XOR | Net: 2->32->1 | Train=20 | Val=500 | 2000 Epochs",
    color='white', fontsize=14, fontweight='bold', y=1.02)
plt.savefig("task2_part2_regularization.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Generalization gap bar chart ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor(BG); ax2.set_facecolor(PN)
names = ["No Reg", "L2 (0.003)", "Dropout (0.65)", "Early Stop"]
clrs  = ["#e74c3c","#f39c12","#27ae60","#3498db"]
gaps  = [abs(val_ov[-1]-tr_ov[-1]), abs(val_l2[-1]-tr_l2[-1]),
         abs(val_dr[-1]-tr_dr[-1]), abs(val_es[-1]-tr_es[-1])]
bars  = ax2.bar(names, gaps, color=clrs, width=0.5, zorder=3)
for b, g in zip(bars, gaps):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
             f"{g:.3f}", ha='center', va='bottom', color='white',
             fontsize=12, fontweight='bold')
ax2.set_title("|Val Loss - Train Loss|  (Generalization Gap)\nLower = Less Overfitting",
              color='white', fontsize=12, fontweight='bold')
ax2.set_ylabel("Generalization Gap", color='#aaa', fontsize=10)
ax2.tick_params(colors='#aaa')
for s in ['top', 'right']:   ax2.spines[s].set_visible(False)
for s in ['bottom', 'left']: ax2.spines[s].set_color(GR)
ax2.grid(True, axis='y', alpha=0.15, ls='--', color='#4444aa')
plt.tight_layout()
plt.savefig("task2_part2_gap_comparison.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("\nPlots saved: task2_part2_regularization.png  |  task2_part2_gap_comparison.png")

# ── Comparison summary ───────────────────────────────────────
print("""
┌──────────────────────────────────────────────────────────────────┐
│           REGULARIZATION COMPARISON SUMMARY                      │
├──────────────────┬───────────────────────────────────────────────┤
│ No Regularisation│ Train loss collapses, val loss stays high.     │
│                  │ Large gap = model memorised training data.     │
├──────────────────┼───────────────────────────────────────────────┤
│ L2 (lambda=0.003)│ Penalises large weights. Smoother decision     │
│                  │ boundaries. Gap significantly reduced.         │
├──────────────────┼───────────────────────────────────────────────┤
│ Dropout (p=0.65) │ Random neuron deactivation builds redundancy.  │
│                  │ Effective regulariser, especially on large nets.│
├──────────────────┼───────────────────────────────────────────────┤
│ Early Stopping   │ Stops before val loss rises. Simple and very   │
│                  │ effective; saves the best checkpoint.          │
└──────────────────┴───────────────────────────────────────────────┘
""")
