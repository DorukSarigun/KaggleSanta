# Relative Transformer Training - Architecture Documentation

**Dosya:** `relative_transformers_training.py`  
**Son Güncelleme:** 2025-01-19

---

## 1. Felsefe

**Neden Relative?**
- Agent mutlak koordinat yerine **pattern** öğrenir
- "Son ağacın 0.5 solunda yerleştir" gibi davranış öğrenebilir
- CNN (512×512 grid) zaten mutlak spatial bilgiyi sağlıyor

---

## 2. Action Space

```
Action: [dx, dy, rotation]
- dx:       [-5, +5]   Son ağaçtan x offset
- dy:       [-5, +5]   Son ağaçtan y offset  
- rotation: [0, 360]   Dönme açısı (derece)
```

**Anchor:**
- İlk ağaç → Merkez (5, 5)
- Sonraki → Son yerleştirilen ağaç

---

## 3. Observation Space

### Tree Features: `(100, 4)`

| # | Feature | Açıklama |
|---|---------|----------|
| 0 | dx | Anchor'dan x farkı |
| 1 | dy | Anchor'dan y farkı |
| 2 | distance | Anchor'a mesafe |
| 3 | angle | Anchor'a açı (radyan) |

### Diğer

| İsim | Shape | Açıklama |
|------|-------|----------|
| scalars | (5,) | n_placed, bbox, fill, remaining, area |
| collision_features | (6,) | Risk metrikleri |
| grid | (512, 512) | Occupancy map (CNN input) |

---

## 4. Network Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      RelativeActorCritic                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ TREE BRANCH                                                  │ │
│  │                                                              │ │
│  │   Input: (batch, 100, 4)                                     │ │
│  │      ↓                                                       │ │
│  │   RelativeTreeEncoder                                        │ │
│  │   ├── Fourier: 4 × 8 freq × 2 = 64D                         │ │
│  │   └── Linear(64→128) → ReLU → Linear(128→128)               │ │
│  │      ↓                                                       │ │
│  │   Output: (batch, 100, 128)                                  │ │
│  │      ↓                                                       │ │
│  │   TransformerEncoder                                         │ │
│  │   ├── 6 layers                                               │ │
│  │   ├── 8 heads                                                │ │
│  │   ├── d_model = 128                                          │ │
│  │   └── feedforward = 512                                      │ │
│  │      ↓                                                       │ │
│  │   Output: (batch, 100, 128)                                  │ │
│  │      ↓                                                       │ │
│  │   Mean Pool (masked) → 128D                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ SCALAR BRANCH                                                │ │
│  │   Input: (batch, 5)                                          │ │
│  │   Linear(5→32) → ReLU → Linear(32→128) → 128D               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ COLLISION BRANCH                                             │ │
│  │   Input: (batch, 6)                                          │ │
│  │   Linear(6→32) → ReLU → Linear(32→128) → 128D               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ GRID BRANCH (CNN)                                            │ │
│  │   Input: (batch, 1, 512, 512)                                │ │
│  │   Conv2d(1→32, 7×7, s=2) + BN + ReLU   → 256×256            │ │
│  │   Conv2d(32→64, 3×3, s=2) + BN + ReLU  → 128×128            │ │
│  │   Conv2d(64→128, 3×3, s=2) + BN + ReLU → 64×64              │ │
│  │   AdaptiveAvgPool2d(1)                  → 128               │ │
│  │   Linear(128→128)                       → 128D              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                     ATTENTION FUSION                              │
│                                                                   │
│   Query = proj(scalar + collision + grid) → 128D                 │
│         ↓                                                         │
│   CrossAttention(Q=query, K=tree_seq, V=tree_seq)                │
│         ↓                                                         │
│   LayerNorm + FFN + LayerNorm → 128D                             │
│         ↓                                                         │
│                   STATE: 128D                                     │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                         HEADS                                     │
│                                                                   │
│   ┌──────────────┐  ┌──────────────┐  ┌─────────┐  ┌───────────┐│
│   │ Actor Mean   │  │ Actor LogStd │  │ Critic  │  │ Collision ││
│   │              │  │              │  │         │  │ Predictor ││
│   │ 128→128→64→3│  │ 128→32→3    │  │128→128  │  │128→64→1   ││
│   │     ↓       │  │     ↓        │  │   →64→1 │  │    ↓      ││
│   │   Tanh      │  │  clamp[-5,0] │  │         │  │  (logit)  ││
│   │     ↓       │  │     ↓        │  │    ↓    │  │           ││
│   │   μ (3)     │  │  log_σ (3)   │  │  V (1)  │  │ logit (1) ││
│   └──────────────┘  └──────────────┘  └─────────┘  └───────────┘│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Reward Function

```
if COLLISION:
    collision_ratio = min(overlap / tree_area, 1.0)
    return -5.0 - 15.0 × collision_ratio   # [-20, -5]

else (SUCCESSFUL PLACEMENT):
    reward = +1.0                          # Base bonus
    
    # Kaggle improvement
    improvement = old_kaggle - new_kaggle
    reward += improvement × 10              # (usually small)
    
    return reward
```

**Kaggle Score:** `s² / n` (s = bbox size, n = trees)

---

## 6. Training (PPO)

```python
# Hyperparameters
lr = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
n_epochs = 10
batch_size = 32

# Loss
loss = policy_loss 
     + 0.5 × value_loss 
     + 0.01 × entropy_loss 
     + 0.1 × collision_loss  # BCE with logits
```

---

## 7. Termination

```python
terminated = (trees >= n_trees) OR (collision_count >= 3)
```

---

## 8. Kullanım

```bash
python3 training/relative_transformers_training.py --n_trees=50 --n_episodes=1000
```
