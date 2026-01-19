# Transformer Tree Packing - Architecture Documentation

**Son Güncelleme:** 2025-01-19

---

## 1. Problem Tanımı

**Amaç:** N Noel ağacını minimum kare bounding box içine yerleştirmek
**Kaggle Skor:** `score = s²/n` (s = max(width, height), n = yerleştirilen ağaç sayısı)

**Episode Sonlandırma:**
- Tüm ağaçlar yerleştirildi VEYA
- **3 collision** oldu (3-strike rule)

---

## 2. Action Space (Continuous 3D)

| Boyut | Değişken | Aralık | Açıklama |
|-------|----------|--------|----------|
| 0 | x | [0, 10] | Yatay pozisyon |
| 1 | y | [0, 10] | Dikey pozisyon |
| 2 | rotation | [0, 360] | Dönme açısı (derece) |

---

## 3. Observation Space

### 3.1 Tree Features: `(100, 39)` - RAW (no normalization)

| Index | Feature | Aralık | Açıklama |
|-------|---------|--------|----------|
| 0-1 | center_x, center_y | [0, 10] | Mutlak pozisyon |
| 2 | rotation | [0, 360] | Derece |
| 3-32 | 15 corners × 2 | [0, 10] | Köşe koordinatları |
| 33-34 | dx, dy to last tree | [-10, 10] | Cartesian delta |
| 35-36 | distance, angle | [0, 14], [-π, π] | Polar to last tree |
| 37-38 | dx, dy to bbox center | [-10, 10] | Cartesian to bbox |

### 3.2 Collision Features: `(6,)` - **YENİ!**

| Index | Feature | Aralık | Açıklama |
|-------|---------|--------|----------|
| 0 | last_collision_area | [0, 1] | Son çarpışma alanı |
| 1 | collision_risk | [0, 1] | Yakın ağaç oranı |
| 2 | avg_min_dist | [0, 1] | Ortalama min mesafe |
| 3 | min_clearance | [0, 1] | Minimum boşluk |
| 4 | crowding_factor | [0, 1] | Doluluk oranı |
| 5 | collision_history_ratio | [0, 1] | Tarihsel çarpışma oranı |

### 3.3 Diğer Observations

| İsim | Shape | Açıklama |
|------|-------|----------|
| mask | (100,) | Geçerli ağaç maskesi |
| scalars | (5,) | n_placed, bbox_size, fill_ratio, remaining, total_area |
| grid | (512, 512) | Occupancy map |
| bbox | (4,) | [minx, miny, maxx, maxy] |

---

## 4. Neural Network Architecture

### 4.1 Genel Akış

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TransformerActorCritic                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  TREE FEATURES (100, 39)                                                     │
│  ├── Absolute (33D) ──→ AbsoluteSpatialEncoding (Fourier) ──→ 96D           │
│  └── Relative (6D) ───→ HybridRelativeEncoding ────────────→ 32D            │
│                    │        ├── Cartesian (4D) → 16D                        │
│                    │        └── Polar (2D, high freq) → 16D                 │
│                    ↓                                                         │
│         CombinedTreeEncoder → Fusion(96+32=128) → Linear → 128D              │
│                    ↓                                                         │
│         TransformerEncoder (6 layers, 8 heads) → (100, 128)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  SCALARS (5) ───────→ ScalarEncoder ────────→ 128D                          │
│  COLLISION (6) ─────→ CollisionEncoder ─────→ 64D  [YENİ!]                  │
│  GRID (512×512) ────→ GridEncoder (ResNet) ──→ 128D                         │
├─────────────────────────────────────────────────────────────────────────────┤
│         CrossAttentionFusion(trees, scalars, grid) → 128D                   │
│  BBOX (4) ──────────→ SpatialContextEncoder (Fourier) → 64D                 │
│                    ↓                                                         │
│              Concat(fusion + bbox_context + collision) → 256D               │
├─────────────────────────────────────────────────────────────────────────────┤
│                         STATE (256D)                                         │
│           ┌──────────────┼──────────────┬──────────────────┐                │
│           ↓              ↓              ↓                  ↓                │
│      Actor Mean     Actor LogStd    Critic        Collision Predictor      │
│      → μ (3)        → log_σ (3)     → V (1)       → logit (1) [YENİ!]      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 CombinedTreeEncoder (Hybrid Encoding)

```
Input: (batch, 100, 39)

ABSOLUTE BRANCH (33 features):
├── center_x, center_y (2)
├── rotation (1)
└── 15 corners × 2 (30)
    ↓
Fourier: 33 × 8 freqs × 2 (sin/cos) = 528D
Linear(528, 96) → 96D

RELATIVE BRANCH (6 features):
├── CARTESIAN: dx, dy, dx_bbox, dy_bbox (4)
│   → Fourier: 4 × 8 × 2 = 64D (standard freqs)
│   → Linear(64, 16) → 16D
│
└── POLAR: dist, angle (2)
    → Fourier: 2 × 8 × 2 = 32D (higher freqs: 2,4,...,256)
    → Linear(32, 16) → 16D

Fusion: Concat(96D, 32D) = 128D → Linear(128, 128) → 128D
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

### 4.4 CollisionEncoder **[YENİ!]**

```
Input: (batch, 6) - collision_features
    ↓
Linear(6, 32) → ReLU → Linear(32, 64) → 64D
```

### 4.5 Collision Predictor **[YENİ!]**

```
Auxiliary task: Sonraki action'ın collision olup olmayacağını tahmin et
    ↓
Input: state (256D)
Linear(256, 128) → ReLU
Linear(128, 64) → ReLU
Linear(64, 1) → raw logit
    ↓
Loss: BCEWithLogits (sigmoid internal)
```

### 4.6 Actor/Critic Heads

```
State: 256D (128 fusion + 64 bbox + 64 collision)
    ↓
Actor Mean:  Linear(256,256) → ReLU → Linear(256,128) → ReLU → Linear(128,3) → Tanh
Actor LogStd: Linear(256,64) → ReLU → Linear(64,3)
Critic: Linear(256,256) → ReLU → Linear(256,128) → ReLU → Linear(128,1)
```

### 4.7 Parameter Sayısı

| Component | Params |
|-----------|--------|
| CombinedTreeEncoder | ~85K |
| TransformerEncoder (6L, 8H) | ~1.2M |
| ScalarEncoder | ~17K |
| CollisionEncoder | ~3K |
| GridEncoder (ResNet) | ~1.1M |
| SpatialContextEncoder | ~12K |
| CrossAttentionFusion | ~200K |
| Actor/Critic Heads | ~150K |
| Collision Predictor | ~25K |
| **TOPLAM** | **~2.94M** |

---

## 5. Reward Function

### 5.1 Collision Penalty (Gradient-based)
```python
if collision:
    collision_ratio = min(overlap / tree_area, 1.0)
    penalty = -5.0 - 15.0 * collision_ratio  # [-20, -5]
```

### 5.2 Successful Placement
```python
reward = 1.0  # Base bonus
reward += (old_kaggle - new_kaggle) * 10  # Kaggle improvement
reward += near_collision_penalty  # [-2, 0] - yakın yerleşim cezası
reward += density_bonus  # [0, 0.5] - optimal mesafede bonus
```

---

## 6. Training (PPO + Auxiliary)

```python
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
n_epochs = 10
batch_size = 32
entropy_coef = 0.01
max_grad_norm = 0.5

# Loss function
loss = policy_loss 
     + 0.5 × value_loss 
     + 0.01 × entropy_loss 
     + 0.1 × collision_loss  # Auxiliary task
```

---

## 7. Kullanım

```bash
cd ~/Documents/Santa2025/KaggleSanta
python3 training/transformer_training.py --n_trees=50 --n_episodes=1000
```

---

## 8. Key Design Decisions

1. **Hybrid Relative Encoding** - Polar (dist, angle) ve Cartesian (dx, dy) ayrı frekans bantlarıyla
2. **3-Strike Rule** - 3 collision'da episode biter (learning signal için)
3. **Collision Predictor** - Auxiliary task olarak collision prediction
4. **Gradient Penalty** - Binary yerine smooth collision penalty
5. **Raw Coordinates** - Normalizasyon yok, Fourier için raw değerler
6. **512×512 Grid** - Yüksek çözünürlüklü spatial input
