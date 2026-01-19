# Transformer Tree Packing - Architecture Documentation

**Last Updated:** 2025-01-19

---

## 1. Problem Definition

**Goal:** Place N Christmas trees to minimize `score = s²/n` where:
- `s` = max(width, height) of bounding box
- `n` = number of trees placed

**Action Space:** Continuous 3D
- `x` ∈ [0, 10], `y` ∈ [0, 10], `rotation` ∈ [0, 360]

**Episode Termination:** All trees placed OR **collision occurs**

---

## 2. Observation Space

### 2.1 Tree Features (100, 39) - RAW (no normalization)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-1 | center_x, center_y | [0, 10] | Absolute position |
| 2 | rotation | [0, 360] | Degrees |
| 3-32 | 15 corners × 2 | [0, 10] | Vertex coordinates |
| 33-34 | dx, dy to last tree | [-10, 10] | Cartesian delta |
| 35-36 | distance, angle | [0, 14], [-π, π] | Polar to last tree |
| 37-38 | dx, dy to bbox center | [-10, 10] | Cartesian to bbox |

### 2.2 Other Observations

| Name | Shape | Description |
|------|-------|-------------|
| mask | (100,) | Valid tree mask |
| scalars | (5,) | n_placed, bbox_size, fill_ratio, remaining, total_area |
| grid | (512, 512) | Occupancy map |
| bbox | (4,) | [minx, miny, maxx, maxy] RAW |

---

## 3. Reward Function

```python
# COLLISION → Immediate termination
if collision:
    return -20.0

# SUCCESSFUL PLACEMENT
reward = 1.0  # Base bonus
reward += (old_kaggle - new_kaggle)  # Kaggle improvement
# where kaggle = bbox_size² / n_placed
```

---

## 4. Neural Network Architecture

### 4.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TransformerActorCritic                           │
├─────────────────────────────────────────────────────────────────────┤
│  TREE FEATURES (100, 39)                                             │
│  ├── Absolute (33D) ──→ AbsoluteSpatialEncoding ──→ 96D             │
│  └── Relative (6D) ───→ HybridRelativeEncoding ───→ 32D             │
│                              ↓                                       │
│                    CombinedTreeEncoder ──→ (100, 128)               │
│                              ↓                                       │
│                    TransformerEncoder ──→ (100, 128)                │
├─────────────────────────────────────────────────────────────────────┤
│  SCALARS (5) ───→ ScalarEncoder ───→ 128D                           │
│  GRID (512²) ──→ GridEncoder (ResNet) ──→ 128D                      │
│                              ↓                                       │
│              CrossAttentionFusion(trees, scalars, grid) ──→ 128D    │
│                              ↓                                       │
│  BBOX (4) ───→ SpatialContextEncoder ───→ 64D                       │
│                              ↓                                       │
│                         Concat ──→ 192D                              │
│                              ↓                                       │
│              ┌──────────────┴──────────────┐                        │
│              ↓                              ↓                        │
│         Actor Head                     Critic Head                   │
│         ──→ μ, log_σ (3)               ──→ V (1)                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 CombinedTreeEncoder (Hybrid)

```
Input: (batch, 100, 39)

ABSOLUTE BRANCH (33 features):
├── center_x, center_y (2)
├── rotation (1)
└── 15 corners × 2 (30)
    ↓
Fourier: 33 × 8 freqs × 2 (sin/cos) = 528D
    ↓
Linear(528, 96) → 96D

RELATIVE BRANCH (6 features):
├── CARTESIAN: dx, dy, dx_bbox, dy_bbox (4)
│   ↓
│   Fourier: 4 × 8 × 2 = 64D (standard freqs)
│   Linear(64, 16) → 16D
│
└── POLAR: dist, angle (2)
    ↓
    Fourier: 2 × 8 × 2 = 32D (higher freqs: 2,4,...,256)
    Linear(32, 16) → 16D
    ↓
Concat → 32D

FUSION:
Concat(96D, 32D) = 128D → Linear(128, 128) → 128D
```

### 4.3 GridEncoder (ResNet-style)

```
Input: (batch, 1, 512, 512)
    ↓
Conv2d(1→32, 7×7, stride=2) + BN + ReLU    → 256²
ResidualBlock(32→64, stride=2)              → 128²
ResidualBlock(64→128, stride=2)             → 64²
ResidualBlock(128→128, stride=2)            → 32²
ResidualBlock(128→128, stride=2)            → 16²
ResidualBlock(128→128, stride=2)            → 8²
    ↓
AdaptiveAvgPool2d(1) → 128D
Linear(128, 128) → 128D
```

### 4.4 SpatialContextEncoder (BBox)

```
Input: (batch, 4) = [minx, miny, maxx, maxy]
    ↓
Fourier: 4 × 8 × 2 = 64D
Linear(64, 64) + ReLU + Linear(64, 64) → 64D
```

### 4.5 Actor/Critic Heads

```
State: 192D (128 fusion + 64 bbox context)
    ↓
Actor:  Linear(192, 128) → ReLU → Linear(128, 6)  # μ(3) + log_σ(3)
Critic: Linear(192, 128) → ReLU → Linear(128, 1)  # V
```

### 4.6 Parameter Count

| Component | Params |
|-----------|--------|
| CombinedTreeEncoder | ~85K |
| TransformerEncoder (6 layers) | ~1.2M |
| ScalarEncoder | ~17K |
| GridEncoder (ResNet) | ~1.1M |
| SpatialContextEncoder | ~12K |
| CrossAttentionFusion | ~200K |
| Actor/Critic Heads | ~50K |
| **Total** | **~2.86M** |

---

## 5. Training (PPO)

```python
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
n_epochs = 4
batch_size = 64
entropy_coef = 0.01
value_coef = 0.5
```

---

## 6. Usage

```bash
python3 training/transformer_training.py --n_trees=50 --n_episodes=1000
```

---

## 7. Key Design Decisions

1. **Hybrid Relative Encoding** - Polar (dist, angle) and Cartesian (dx, dy) processed separately with different frequency bands
2. **Higher frequencies for Polar** - Fine-grained distance/angle discrimination
3. **No normalization** - Raw coordinates for Fourier encoding
4. **Collision terminates** - Forces collision avoidance learning
5. **Kaggle metric in reward** - Direct optimization of competition score
