# Relative Transformer Architecture v4.0

## Overview
Fully polar coordinate-based tree packing with spatial awareness via CoordConv + SPP.

**Key innovations:**
- Polar action space aligned with polar observation features
- Spatial CNN with Polar CoordConv for position awareness
- Spatial Pyramid Pooling for multi-scale spatial info
- Minimal hand-crafted features (scalars/collision disabled)

---

## Network Summary

| Component | Parameters | Description |
|-----------|------------|-------------|
| TreeEncoder | ~50K | Fourier-encoded 6D polar features |
| Transformer | ~800K | 6-layer, 8-head attention |
| SpatialGridEncoder | ~300K | Polar CoordConv + ResNet + SPP |
| SimpleFusion | ~100K | Tree + Grid concatenation |
| Actor/Critic Heads | ~500K | Policy and value networks |
| **Total** | **~1.78M** | |

---

## Action Space (3D Polar)

```
[r, θ, rotation]
- r ∈ [0, 5]: distance from board center
- θ ∈ [-π, π]: angle from board center  
- rotation ∈ [0, 360]: tree rotation
```

Anchor: **Fixed board center (5, 5)** for translation invariance.

---

## Observation Space

### 1. Tree Features (100 × 6D) — POLAR

Per placed tree, relative to board center:

| Feature | Range | Description |
|---------|-------|-------------|
| r | [0, ~7] | Distance to board center |
| sin_θ | [-1, 1] | Direction sine |
| cos_θ | [-1, 1] | Direction cosine |
| r_bbox | [0, ~7] | Distance to bbox center |
| sin_θ_bbox | [-1, 1] | Direction to bbox sine |
| cos_θ_bbox | [-1, 1] | Direction to bbox cosine |

**Note:** Clearance and density removed — CNN handles spatial context.

### 2. Scalar Features (11D) — DISABLED

Currently zeroed for testing. CNN provides global context.

### 3. Collision Features (6D) — DISABLED

Currently zeroed. CNN handles collision detection via occupancy grid.

### 4. Occupancy Grid (512 × 512)

Grayscale image of placed trees, processed by SpatialGridEncoder.

---

## SpatialGridEncoder Architecture

```
Input: (batch, 1, 512, 512) grayscale occupancy grid

┌─────────────────────────────────────────────────────────┐
│ Polar CoordConv                                         │
│ Add coordinate channels: (r, sin_θ, cos_θ) from center  │
│ → (batch, 4, 512, 512)                                  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Stem Conv (7×7, stride=2)                               │
│ 4 → 16 channels, 512 → 256                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ ResBlock Stack                                          │
│ 16 → 32 (256→128) → 64 (128→64) → 64 (64→32) → 64 (32→16) │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Spatial Pyramid Pooling (SPP)                           │
│ Pool at 4×4 + 2×2 + 1×1 = 21 cells                      │
│ → (batch, 64 × 21 = 1344)                               │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Projection                                              │
│ 1344 → 128 (embed_dim)                                  │
└─────────────────────────────────────────────────────────┘

Output: (batch, 128)
```

### Polar CoordConv Details

```python
# Board normalized to [-1, 1]
r = sqrt(x² + y²) / sqrt(2)   # [0, 1] from center
sin_θ = y / r                  # Direction
cos_θ = x / r                  # Direction
```

**Why Polar CoordConv?**
- CNN sees each pixel's distance and direction from center
- Aligns with polar action space
- Agent can identify "gaps at 45° direction"

### SPP Benefits

| Scale | Cells | Info Captured |
|-------|-------|---------------|
| 4×4 | 16 | Regional density per quadrant |
| 2×2 | 4 | Quadrant-level layout |
| 1×1 | 1 | Global occupancy |

---

## Network Flow

```
Observation
    │
    ├── relative_trees (100, 6)
    │       ↓
    │   RelativeTreeEncoder (Fourier)
    │       ↓ (100, 128)
    │   TransformerEncoder (6 layers)
    │       ↓ (100, 128)
    │   MaskedMeanPool
    │       ↓ (128,)
    │
    └── grid (512, 512)
            ↓
        SpatialGridEncoder
            ↓ (128,)
                  │
                  ↓
        ┌─────────────────┐
        │  SimpleFusion   │
        │  concat + MLP   │
        │  (256 → 128)    │
        └─────────────────┘
                  ↓
           State (128,)
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   ActorHead           CriticHead
   (128 → 3D)          (128 → 1)
   mean + logstd       value
```

---

## Reward Function

### Collision Penalty (Progress-Based)
```python
if collision_area > 0:
    remaining_trees = n_trees - current_index
    return -float(remaining_trees)  # Early=-40, Late=-5
```

### Success Rewards
```python
placement_bonus = +3.0  # Per successful placement

# Exploration bonus
if n_placed >= 2:
    dist = distance(last_tree, prev_tree)
    exploration_bonus = min(dist/2, 1) * 0.5  # Up to +0.5

# Kaggle improvement
kaggle_improvement = (old_score - new_score) * 10
```

---

## CLI Usage

```bash
python3 training/relative_transformers_training.py \
    --n_trees=50 \
    --n_episodes=1000 \
    --anchor_mode=board_center \
    --action_mode=polar \
    --feature_mode=polar
```

---

## Design Philosophy

1. **Polar Everywhere:** Actions, features, and CNN coordinates all polar
2. **Learn, Don't Engineer:** Disabled hand-crafted scalars/collision features
3. **Spatial Awareness:** CoordConv + SPP gives CNN position understanding
4. **Compact:** ~1.78M params total
