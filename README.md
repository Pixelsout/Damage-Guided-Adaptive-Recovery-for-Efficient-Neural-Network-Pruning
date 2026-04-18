# Damage-Guided Adaptive Recovery for Efficient Neural Network Pruning

> **Independent Research Project** — An original post-training pruning system that detects which layers were hurt most by pruning and intelligently recovers only the critical connections, without retraining the full network.


> 🔗 **Related Work:** [Self-Pruning Neural Network](https://github.com/Pixelsout/tredence-case-study) — a complementary approach where the network learns to prune itself during training using learnable sigmoid gates.

---

## Abstract

Standard magnitude pruning removes the smallest weights globally, treating all layers equally. This ignores a fundamental asymmetry — some layers are far more sensitive to pruning than others. This project introduces a **damage-guided recovery pipeline** that:

1. Measures per-layer output deviation after pruning (the "damage score")
2. Ranks layers by how much they were hurt
3. Selectively restores only the most important pruned weights in critical layers
4. Fine-tunes exclusively the damaged layers — not the full network

The result is a principled, compute-efficient recovery system that outperforms uniform pruning + full retraining on both accuracy and training cost.

---

## Motivation

When a network is pruned aggressively, accuracy drops — but not because every layer suffered equally. Some layers act as bottlenecks; removing their weights causes cascading errors throughout the forward pass. Others are highly redundant and can be pruned heavily with almost no accuracy loss.

Standard recovery approaches ignore this structure and either:
- Fully retrain all layers (expensive, wastes compute on already-stable layers)
- Apply no recovery at all (accuracy stays degraded)

**Damage-Guided Recovery** identifies exactly which layers need attention and focuses recovery effort there.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE                            │
│                                                             │
│  1. Train VGGSmall baseline (CIFAR-10, 50 epochs)          │
│         ↓                                                   │
│  2. Apply global magnitude pruning (30% – 90%)             │
│         ↓                                                   │
│  3. DamageDetector                                          │
│     Forward pass: original model vs pruned model           │
│     D(ℓ) = MSE(activations_orig, activations_pruned)       │
│         ↓                                                   │
│  4. Rank layers by D(ℓ) → identify critical top-25%        │
│         ↓                                                   │
│  5. WeightRestorer                                          │
│     Restore top-k pruned weights in critical layers        │
│     Selection: highest magnitude in original model         │
│         ↓                                                   │
│  6. FocusedTrainer                                          │
│     Freeze stable layers, fine-tune critical layers only   │
│         ↓                                                   │
│  7. Evaluate: accuracy recovery vs baseline                │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. DamageDetector

Measures how much each layer's output changed after pruning using forward-pass hooks on both the original and pruned model simultaneously.

```python
class DamageDetector:
    def compute_damage(self, loader):
        # Registers hooks on both models
        # Runs a forward pass on the same batch
        # D(ℓ) = MSE(output_orig[ℓ], output_pruned[ℓ])
        damage[name] = (orig_out - pruned_out).pow(2).mean().item()
```

The damage score `D(ℓ)` is a direct measure of how much a layer's representation was disrupted — not a proxy like weight magnitude.

### 2. WeightRestorer

Given the damage-ranked critical layers, selectively restores the most important pruned weights. Restoration amount is proportional to the layer's damage score.

```python
class WeightRestorer:
    def restore(self, alpha=0.1):
        # Only touches critical layers
        # Only restores currently-zero weights
        # Selects by original magnitude (most important first)
        k = int(alpha * damage_score * layer_size)
        restore top-k pruned weights from original checkpoint
```

Key design decision: the classifier (output layer) is never restored — it must learn the new sparse representation directly.

### 3. FocusedTrainer

Freezes all non-critical layers and fine-tunes only the damaged ones. This saves compute compared to full retraining while concentrating gradient updates where they are needed most.

```python
class FocusedTrainer:
    def freeze_non_critical(self):
        for name, param in model.named_parameters():
            param.requires_grad = (layer_name in critical_layers)
```

### 4. ImportanceScorer (3-Factor)

Three progressive importance metrics for pruning decision quality:

| Mode | Formula | Description |
|------|---------|-------------|
| `magnitude` | `\|w\|` | Weight magnitude only |
| `mag_grad` | `\|w\| × \|∇w\|` | Magnitude weighted by gradient sensitivity |
| `mag_grad_act` | `\|w\| × \|∇w\| × activation_freq` | Full 3-factor importance |

The 3-factor scorer uses gradient hooks and activation hooks simultaneously to capture how important each weight is to both the loss surface and the activation flow.

---

## Experiments

### Experiment 1 — Pruning Severity & Breaking Point

Magnitude pruning applied at 30%, 50%, 70%, 80%, 90% sparsity. Accuracy tracked to find the network's breaking point.


### Experiment 2 — Magnitude vs Random Pruning

Magnitude pruning consistently outperforms random pruning at every sparsity level, confirming that weight magnitude is a meaningful proxy for importance.


### Experiment 3 — Per-Layer Damage Analysis

After 70% pruning, damage scores reveal a highly non-uniform distribution. Early convolutional layers suffer more than later layers — they learn general low-level features that many subsequent layers depend on.


### Experiment 4 — Recovery Pipeline Results

| Stage | Accuracy (50% pruned) | Accuracy (70% pruned) |
|-------|:---------------------:|:---------------------:|
| Baseline | ~85% | ~85% |
| After Pruning | ~80% | ~65% |
| After Restoration | ~82% | ~72% |
| After Focused Fine-Tuning | ~84% | ~80% |


### Experiment 5 — Importance Scoring Comparison

3-factor importance (Magnitude × Gradient × Activation) consistently outperforms magnitude-only pruning at high sparsity levels.


### Experiment 6 — Multi-Dataset Generalization

The full pipeline tested across three datasets to verify generalization:

| Dataset | Classes | Baseline | After 70% Pruning | After Recovery |
|---------|---------|----------|:----------------:|:--------------:|
| CIFAR-10 | 10 | ~85% | ~65% | ~80% |
| CIFAR-100 | 100 | ~55% | ~40% | ~50% |
| FashionMNIST | 10 | ~91% | ~85% | ~89% |


---

## Key Findings

**1. Damage is highly non-uniform.** The top 25% most-damaged layers account for the majority of accuracy loss. Focusing recovery there is sufficient.

**2. Selective restoration works.** Restoring only the top-k weights by original magnitude in critical layers recovers 60–70% of the accuracy drop, with no training required.

**3. Focused fine-tuning is more efficient.** Training only critical layers (typically 25% of all layers) achieves comparable recovery to full retraining in roughly 30% of the training steps.

**4. Gradient × activation importance outperforms magnitude alone** at high sparsity (70%+). At low sparsity (30–50%), the difference is minimal.

**5. Breaking points are dataset-dependent.** CIFAR-100 breaks earlier (~50% sparsity) than FashionMNIST (~80% sparsity) due to higher classification complexity.

---

## Repository Structure

```
Damage-Guided-Adaptive-Recovery/
├── neural_pruning.ipynb      # Main research notebook
│   ├── VGGSmall baseline training
│   ├── MagnitudeScorer + RandomPruner
│   ├── Pruning severity sweep
│   ├── GradientHookScorer
│   ├── ImportanceScorer (3-factor)
│   ├── DamageDetector + visualization
│   ├── WeightRestorer
│   └── FocusedTrainer
│
├── Multidata_Testing.ipynb   # Multi-dataset generalization experiments
│   ├── CIFAR-100 experiments
│   ├── FashionMNIST experiments
│   ├── Adaptive vs Uniform pruning comparison
│   └── Adaptive recovery experiment
│
└── README.md
```

---

## Run

Open either notebook directly in Google Colab using the badge at the top. Enable GPU via `Runtime → Change runtime type → T4 GPU`.

All dependencies are standard:
```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## Design Decisions Worth Noting

**Why MSE of activations as the damage metric?**
Weight-level metrics (magnitude, gradient) measure potential importance but not actual impact. Activation MSE directly measures how much the layer's output changed — it captures the real effect of pruning on the information flow.

**Why not restore all pruned weights in critical layers?**
Restoring all weights would undo the pruning entirely. The damage-proportional restoration (`k = alpha × D(ℓ) × layer_size`) ensures that layers with higher damage get more weights back, while still maintaining the overall compression ratio.

**Why freeze non-critical layers during fine-tuning?**
Gradient updates to stable layers can disturb their already-good representations. Freezing them concentrates the training signal on the layers that actually need it, and reduces training time significantly.

---

*Author: Biswajeet*
