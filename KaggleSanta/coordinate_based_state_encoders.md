# 🧠 Koordinat Bazlı State Representation için Neural Network Mimarileri

## Identical Trees - Grid-Free Yaklaşım

---

# Bölüm 1: Problem Tanımı

## 1.1 Input Formatı

```
Yerleştirilmiş ağaçlar: S = {(x₁, y₁, θ₁), (x₂, y₂, θ₂), ..., (xₖ, yₖ, θₖ)}

Her eleman:
- xᵢ: Ağacın merkez x koordinatı
- yᵢ: Ağacın merkez y koordinatı  
- θᵢ: Ağacın rotasyon açısı (radyan)

k: Şu ana kadar yerleştirilmiş ağaç sayısı (0 ≤ k < n)
n: Toplam yerleştirilecek ağaç sayısı
```

## 1.2 Problemin Özellikleri

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUT ÖZELLİKLERİ                               │
│                                                                     │
│  1. DEĞİŞKEN UZUNLUK                                               │
│     - İlk adımda: 0 ağaç (boş set)                                 │
│     - Son adımda: n-1 ağaç                                         │
│     - Her adımda farklı boyut                                      │
│                                                                     │
│  2. PERMUTATION INVARIANCE                                          │
│     - {(1,2,0°), (3,4,90°)} = {(3,4,90°), (1,2,0°)}               │
│     - Sıralama sonucu değiştirmemeli                               │
│                                                                     │
│  3. ELEMANLAR ARASI İLİŞKİ                                         │
│     - Ağaçlar birbirine yakın mı?                                  │
│     - Collision var mı?                                            │
│     - Cluster oluşmuş mu?                                          │
│                                                                     │
│  4. IMPLICIT GEOMETRY                                               │
│     - Network ağaç şeklini öğrenmeli                               │
│     - (x, y, θ) → Hangi alan kaplanıyor?                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 1.3 Çıktı (State Embedding)

```
Network'ün üretmesi gereken:
- Global state embedding: ℝ^d (sabit boyutlu vektör)
- Bu embedding'den:
  - Actor: Sonraki (x, y, θ) kararı
  - Critic: V(s) değeri
```

---

# Bölüm 2: DeepSets

## 2.1 Teori

DeepSets, set-structured data için matematiksel olarak permutation invariant olması kanıtlanmış ilk mimaridir.

### Temel Teorem:
```
Bir f fonksiyonu permutation invariant ⟺ 
f(X) = ρ(∑ᵢ φ(xᵢ)) formunda yazılabilir

Burada:
- φ: Eleman bazlı transform (her xᵢ için ayrı)
- ∑: Permutation invariant aggregation (toplam, ortalama, max)
- ρ: Aggregated feature üzerinde transform
```

### Mimari:
```
┌─────────────────────────────────────────────────────────────────────┐
│                         DEEPSETS                                    │
│                                                                     │
│   Input: {x₁, x₂, ..., xₖ}  (her xᵢ ∈ ℝ³: koordinat+rotation)     │
│                                                                     │
│          x₁ ──→ [φ] ──→ h₁ ─┐                                      │
│          x₂ ──→ [φ] ──→ h₂ ─┼──→ [∑/mean/max] ──→ z ──→ [ρ] ──→ y │
│          ...                 │                                      │
│          xₖ ──→ [φ] ──→ hₖ ─┘                                      │
│                                                                     │
│   φ: ℝ³ → ℝᵈ  (shared MLP)                                        │
│   ρ: ℝᵈ → ℝᵈ  (output MLP)                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 2.2 Implementasyon

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class DeepSets(nn.Module):
    """
    DeepSets: Permutation Invariant Neural Network
    
    Paper: "Deep Sets" (Zaheer et al., 2017)
    
    Avantajlar:
    - Matematiksel permutation invariance garantisi
    - Basit ve hızlı
    - Değişken boyutlu input
    
    Dezavantajlar:
    - Elemanlar arası ilişkiyi doğrudan modellemez
    - Sadece aggregation üzerinden dolaylı ilişki
    """
    
    def __init__(
        self,
        input_dim: int = 3,           # (x, y, θ)
        hidden_dim: int = 128,
        output_dim: int = 256,
        n_phi_layers: int = 3,        # φ network derinliği
        n_rho_layers: int = 2,        # ρ network derinliği
        aggregation: str = 'mean',    # 'mean', 'max', 'sum', 'attention'
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        
        # ════════════════════════════════════════════════════════════
        # φ Network: Eleman bazlı transform
        # Her (x, y, θ) → hidden representation
        # ════════════════════════════════════════════════════════════
        
        phi_layers = []
        in_dim = input_dim
        
        for i in range(n_phi_layers):
            out_dim = hidden_dim
            phi_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        self.phi = nn.Sequential(*phi_layers)
        
        # ════════════════════════════════════════════════════════════
        # Attention pooling (optional)
        # ════════════════════════════════════════════════════════════
        
        if aggregation == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # ════════════════════════════════════════════════════════════
        # ρ Network: Aggregated feature → output
        # ════════════════════════════════════════════════════════════
        
        rho_layers = []
        in_dim = hidden_dim
        
        # Multi-head aggregation kullanıyorsak input boyutu artar
        if aggregation == 'multi':
            in_dim = hidden_dim * 3  # mean + max + sum
        
        for i in range(n_rho_layers):
            out_dim = output_dim if i == n_rho_layers - 1 else hidden_dim
            rho_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU() if i < n_rho_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_rho_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.rho = nn.Sequential(*rho_layers)
        
        # ════════════════════════════════════════════════════════════
        # Empty set handling
        # Boş set için learnable embedding
        # ════════════════════════════════════════════════════════════
        
        self.empty_set_embedding = nn.Parameter(torch.randn(output_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch_size, max_set_size, input_dim) - Padded input
            mask: (batch_size, max_set_size) - True for valid elements
        
        Returns:
            output: (batch_size, output_dim) - Set embedding
        """
        batch_size, max_set_size, _ = x.shape
        
        # Handle empty sets
        if mask is None:
            mask = torch.ones(batch_size, max_set_size, dtype=torch.bool, device=x.device)
        
        # Check for completely empty sets
        set_sizes = mask.sum(dim=1)  # (batch_size,)
        empty_mask = set_sizes == 0  # (batch_size,)
        
        # ════════════════════════════════════════════════════════════
        # φ: Element-wise transform
        # ════════════════════════════════════════════════════════════
        
        # (batch, max_set_size, input_dim) → (batch, max_set_size, hidden_dim)
        h = self.phi(x)
        
        # Mask invalid elements (set to zero before aggregation)
        h = h * mask.unsqueeze(-1).float()
        
        # ════════════════════════════════════════════════════════════
        # Aggregation: Permutation invariant pooling
        # ════════════════════════════════════════════════════════════
        
        if self.aggregation == 'mean':
            # Mean pooling (mask-aware)
            z = h.sum(dim=1) / (set_sizes.unsqueeze(-1).float() + 1e-8)
        
        elif self.aggregation == 'max':
            # Max pooling (mask invalid with -inf)
            h_masked = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            z = h_masked.max(dim=1)[0]
        
        elif self.aggregation == 'sum':
            z = h.sum(dim=1)
        
        elif self.aggregation == 'attention':
            # Attention-weighted pooling
            attn_scores = self.attention_weights(h).squeeze(-1)  # (batch, max_set_size)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, max_set_size)
            z = (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        
        elif self.aggregation == 'multi':
            # Multiple aggregations concatenated
            z_mean = h.sum(dim=1) / (set_sizes.unsqueeze(-1).float() + 1e-8)
            
            h_masked = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            z_max = h_masked.max(dim=1)[0]
            z_max = z_max.masked_fill(z_max == float('-inf'), 0)
            
            z_sum = h.sum(dim=1)
            
            z = torch.cat([z_mean, z_max, z_sum], dim=-1)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # ════════════════════════════════════════════════════════════
        # ρ: Final transform
        # ════════════════════════════════════════════════════════════
        
        output = self.rho(z)
        
        # Handle empty sets: use learned embedding
        if empty_mask.any():
            output[empty_mask] = self.empty_set_embedding
        
        return output


class DeepSetsEncoder(nn.Module):
    """
    Tree Packing için DeepSets-based State Encoder
    
    Ek özellikler:
    - Koordinat normalizasyonu
    - Rotation encoding (sin/cos)
    - Global features ekleme
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 256,
        use_rotation_encoding: bool = True,
        include_global_features: bool = True
    ):
        super().__init__()
        
        self.use_rotation_encoding = use_rotation_encoding
        self.include_global_features = include_global_features
        
        # Input dim: x, y + rotation encoding
        if use_rotation_encoding:
            input_dim = 4  # x, y, sin(θ), cos(θ)
        else:
            input_dim = 3  # x, y, θ
        
        # DeepSets core
        self.deepsets = DeepSets(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            aggregation='multi'
        )
        
        # Global features encoder
        if include_global_features:
            self.global_encoder = nn.Sequential(
                nn.Linear(4, 64),  # n_placed, n_remaining, bbox_area, efficiency
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            
            # Fusion
            self.fusion = nn.Sequential(
                nn.Linear(output_dim + 64, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
    
    def preprocess(
        self, 
        coordinates: torch.Tensor,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> torch.Tensor:
        """
        Koordinatları preprocess et
        
        Args:
            coordinates: (batch, k, 3) - x, y, θ
            bbox: Current bounding box for normalization
        """
        x = coordinates[..., 0]
        y = coordinates[..., 1]
        theta = coordinates[..., 2]
        
        # Normalize coordinates (0-1 range)
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            x = (x - minx) / (maxx - minx + 1e-8)
            y = (y - miny) / (maxy - miny + 1e-8)
        else:
            # Auto-normalize based on data
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        # Rotation encoding
        if self.use_rotation_encoding:
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            return torch.stack([x, y, sin_theta, cos_theta], dim=-1)
        else:
            # Normalize theta to [0, 1]
            theta_norm = theta / (2 * np.pi)
            return torch.stack([x, y, theta_norm], dim=-1)
    
    def forward(
        self,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            coordinates: (batch, max_k, 3) - Placed tree coordinates
            mask: (batch, max_k) - Valid elements mask
            global_features: (batch, 4) - Global state features
        
        Returns:
            state_embedding: (batch, output_dim)
        """
        # Preprocess
        x = self.preprocess(coordinates)
        
        # DeepSets encoding
        set_embedding = self.deepsets(x, mask)
        
        # Add global features
        if self.include_global_features and global_features is not None:
            global_encoded = self.global_encoder(global_features)
            combined = torch.cat([set_embedding, global_encoded], dim=-1)
            state_embedding = self.fusion(combined)
        else:
            state_embedding = set_embedding
        
        return state_embedding
```

## 2.3 DeepSets Özet

```
✅ Avantajlar:
- Matematiksel permutation invariance garantisi
- Basit implementasyon
- Hızlı training ve inference
- Değişken boyutlu input doğal olarak desteklenir

❌ Dezavantajlar:
- Elemanlar arası ilişki sadece aggregation üzerinden
- "Ağaç A, Ağaç B'ye yakın" bilgisi kaybolabilir
- Spatial reasoning için yetersiz kalabilir
```

---

# Bölüm 3: Transformer (Self-Attention)

## 3.1 Teori

Transformer, attention mekanizması ile her elemanın diğer tüm elemanlarla ilişki kurmasını sağlar.

### Self-Attention:
```
Attention(Q, K, V) = softmax(QKᵀ / √d) V

Her eleman için:
- Query (Q): "Ne arıyorum?"
- Key (K): "Ne sunuyorum?"
- Value (V): "Değerim ne?"

Sonuç: Her eleman, diğer tüm elemanlara göre ağırlıklandırılmış bilgi alır
```

### Neden Transformer?
```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  DeepSets:    x₁ ─→ φ ─┐                                           │
│               x₂ ─→ φ ─┼→ aggregate → ρ → output                   │
│               x₃ ─→ φ ─┘                                           │
│                                                                     │
│  Elemanlar birbirini GÖRMİYOR (sadece aggregate'de birleşiyor)     │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Transformer: x₁ ←──attention──→ x₂ ←──attention──→ x₃            │
│                 ↖                 ↑                 ↗               │
│                   ←──────────────────────────────→                  │
│                                                                     │
│  Her eleman diğer TÜM elemanları GÖRÜYOR                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.2 Implementasyon

```python
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention
    
    Her head farklı bir ilişki türünü öğrenebilir:
    - Head 1: Yakınlık ilişkisi
    - Head 2: Rotasyon benzerliği
    - Head 3: Cluster membership
    - ...
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Her head'in boyutu
        
        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - True for valid positions
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head: (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask (for padded positions)
        if mask is not None:
            # mask: (batch, seq_len) → (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_k)
        
        # Reshape back: (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Yapı:
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-Attention
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len)
        """
        # Self-Attention with residual
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-Forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class TransformerSetEncoder(nn.Module):
    """
    Transformer-based Set Encoder for Tree Packing
    
    Özellikler:
    - Positional encoding YOK (permutation invariance için)
    - CLS token ile global representation
    - Değişken uzunluklu input
    """
    
    def __init__(
        self,
        input_dim: int = 3,           # (x, y, θ)
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        pooling: str = 'cls'          # 'cls', 'mean', 'max', 'attention'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.pooling = pooling
        
        # ════════════════════════════════════════════════════════════
        # Input Embedding
        # ════════════════════════════════════════════════════════════
        
        # Coordinate embedding (x, y, θ) → d_model
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # CLS token (learnable)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Empty set embedding
        self.empty_embedding = nn.Parameter(torch.randn(d_model))
        
        # ════════════════════════════════════════════════════════════
        # Transformer Layers
        # ════════════════════════════════════════════════════════════
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # ════════════════════════════════════════════════════════════
        # Output Pooling
        # ════════════════════════════════════════════════════════════
        
        if pooling == 'attention':
            self.pool_attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - Koordinatlar
            mask: (batch, seq_len) - Valid elements
        
        Returns:
            output: (batch, d_model) - Set embedding
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Handle empty sets
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        set_sizes = mask.sum(dim=1)
        empty_mask = set_sizes == 0
        
        # ════════════════════════════════════════════════════════════
        # Input Embedding
        # ════════════════════════════════════════════════════════════
        
        h = self.input_embed(x)  # (batch, seq_len, d_model)
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            h = torch.cat([cls_tokens, h], dim=1)  # (batch, 1 + seq_len, d_model)
            
            # Update mask for CLS token (always valid)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # ════════════════════════════════════════════════════════════
        # Transformer Layers
        # ════════════════════════════════════════════════════════════
        
        for layer in self.layers:
            h = layer(h, mask)
        
        # ════════════════════════════════════════════════════════════
        # Pooling → Global Representation
        # ════════════════════════════════════════════════════════════
        
        if self.pooling == 'cls' and self.use_cls_token:
            # CLS token representation
            output = h[:, 0, :]  # (batch, d_model)
        
        elif self.pooling == 'mean':
            # Masked mean pooling
            if self.use_cls_token:
                h = h[:, 1:, :]  # Remove CLS token
                mask = mask[:, 1:]
            
            h_masked = h * mask.unsqueeze(-1).float()
            output = h_masked.sum(dim=1) / (set_sizes.unsqueeze(-1).float() + 1e-8)
        
        elif self.pooling == 'max':
            if self.use_cls_token:
                h = h[:, 1:, :]
                mask = mask[:, 1:]
            
            h_masked = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            output = h_masked.max(dim=1)[0]
            output = output.masked_fill(output == float('-inf'), 0)
        
        elif self.pooling == 'attention':
            if self.use_cls_token:
                h = h[:, 1:, :]
                mask = mask[:, 1:]
            
            attn_scores = self.pool_attention(h).squeeze(-1)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)
            output = (h * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Final normalization
        output = self.final_norm(output)
        
        # Handle empty sets
        if empty_mask.any():
            output[empty_mask] = self.empty_embedding
        
        return output


class TransformerStateEncoder(nn.Module):
    """
    Complete Transformer-based State Encoder with preprocessing
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        include_global: bool = True
    ):
        super().__init__()
        
        self.include_global = include_global
        
        # Input: (x, y, sin(θ), cos(θ)) = 4 dim
        self.transformer = TransformerSetEncoder(
            input_dim=4,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            pooling='cls'
        )
        
        # Global features
        if include_global:
            self.global_encoder = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(d_model + 64, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            )
    
    def preprocess_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        (x, y, θ) → (x_norm, y_norm, sin(θ), cos(θ))
        """
        x = coords[..., 0]
        y = coords[..., 1]
        theta = coords[..., 2]
        
        # Normalize x, y (simple min-max, can be improved)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        # Rotation encoding
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        
        return torch.stack([x_norm, y_norm, sin_t, cos_t], dim=-1)
    
    def forward(
        self,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            coordinates: (batch, max_k, 3) - (x, y, θ)
            mask: (batch, max_k)
            global_features: (batch, 4)
        
        Returns:
            state_embedding: (batch, d_model)
        """
        # Preprocess
        x = self.preprocess_coordinates(coordinates)
        
        # Transformer encoding
        set_embedding = self.transformer(x, mask)
        
        # Fusion with global features
        if self.include_global and global_features is not None:
            global_encoded = self.global_encoder(global_features)
            combined = torch.cat([set_embedding, global_encoded], dim=-1)
            state_embedding = self.fusion(combined)
        else:
            state_embedding = set_embedding
        
        return state_embedding
```

## 3.3 Transformer Özet

```
✅ Avantajlar:
- Her eleman diğer TÜM elemanları görür
- Spatial relationships doğrudan modellenir
- Çok güçlü representation öğrenme
- State-of-the-art birçok set probleminde

❌ Dezavantajlar:
- O(n²) complexity (n eleman sayısı)
- Daha fazla parametre, yavaş training
- Küçük setlerde overkill olabilir
```

---

# Bölüm 4: Set Transformer

## 4.1 Teori

Set Transformer, DeepSets ve Transformer'ın en iyi yönlerini birleştirir.

### Ana Fikir:
```
Problem: Transformer O(n²) - büyük setler için yavaş
Çözüm: Inducing points ile O(nm) complexity (m << n)

Inducing Points:
- m adet "özet" nokta öğren
- Set elemanları önce inducing points'e attend eder
- Sonra inducing points birbirine attend eder
- Son olarak set elemanları inducing points'ten okur
```

### Mimari:
```
┌─────────────────────────────────────────────────────────────────────┐
│                       SET TRANSFORMER                               │
│                                                                     │
│   Input Set (n eleman)                                             │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────┐              │
│   │  ISAB: Induced Self-Attention Block             │              │
│   │                                                 │              │
│   │  Set ──→ [Attend to Inducing] ──→ [Inducing    │              │
│   │          Points (m)             Attention]      │              │
│   │                ↓                    │           │              │
│   │          [Read from Inducing] ←─────┘           │              │
│   │                ↓                                │              │
│   │           Updated Set                           │              │
│   └─────────────────────────────────────────────────┘              │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────┐              │
│   │  PMA: Pooling by Multi-head Attention           │              │
│   │                                                 │              │
│   │  Seed vectors ──→ [Attend to Set] ──→ Output   │              │
│   └─────────────────────────────────────────────────┘              │
│        │                                                            │
│        ▼                                                            │
│   Output (fixed size)                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2 Implementasyon

```python
class MultiHeadAttention(nn.Module):
    """
    Generalized Multi-Head Attention
    Q, K, V ayrı inputlardan gelebilir (cross-attention için)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Q: (batch, q_len, d_model) - Query
            K: (batch, k_len, d_model) - Key
            V: (batch, v_len, d_model) - Value (usually same as K)
            mask: (batch, k_len) - Mask for keys
        
        Returns:
            output: (batch, q_len, d_model)
        """
        batch_size = Q.shape[0]
        q_len = Q.shape[1]
        k_len = K.shape[1]
        
        # Project
        Q = self.W_q(Q).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, k_len)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        return self.W_o(context)


class MAB(nn.Module):
    """
    Multi-head Attention Block
    
    MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = LayerNorm(X + Attention(X, Y, Y))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            X: (batch, x_len, d_model) - Query source
            Y: (batch, y_len, d_model) - Key/Value source
            mask: (batch, y_len) - Mask for Y
        """
        # Attention
        H = self.norm1(X + self.attention(X, Y, Y, mask))
        
        # Feed-forward
        output = self.norm2(H + self.ff(H))
        
        return output


class SAB(nn.Module):
    """
    Self-Attention Block
    
    SAB(X) = MAB(X, X)
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, d_ff, dropout)
    
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mab(X, X, mask)


class ISAB(nn.Module):
    """
    Induced Self-Attention Block
    
    Complexity: O(nm) instead of O(n²)
    
    ISAB_m(X) = MAB(X, H) where H = MAB(I, X)
    I: m learnable inducing points
    
    Flow:
    1. Inducing points attend to input → H (m × d)
    2. Input attends to H → output (n × d)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_inducing: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_inducing = n_inducing
        
        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, d_model))
        
        # Two MABs
        self.mab1 = MAB(d_model, n_heads, d_ff, dropout)  # I → X
        self.mab2 = MAB(d_model, n_heads, d_ff, dropout)  # X → H
    
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            X: (batch, n, d_model)
            mask: (batch, n)
        """
        batch_size = X.shape[0]
        
        # Expand inducing points for batch
        I = self.inducing_points.expand(batch_size, -1, -1)
        
        # Inducing points attend to X
        H = self.mab1(I, X, mask)  # (batch, m, d_model)
        
        # X attends to H (no mask needed for H)
        output = self.mab2(X, H)  # (batch, n, d_model)
        
        return output


class PMA(nn.Module):
    """
    Pooling by Multi-head Attention
    
    Uses learnable seed vectors to pool set into fixed-size output
    
    PMA_k(Z) = MAB(S, Z)
    S: k learnable seed vectors
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_seeds: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_seeds = n_seeds
        
        # Learnable seed vectors
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model))
        
        # MAB for pooling
        self.mab = MAB(d_model, n_heads, d_ff, dropout)
    
    def forward(self, Z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            Z: (batch, n, d_model) - Encoded set
            mask: (batch, n)
        
        Returns:
            output: (batch, n_seeds, d_model)
        """
        batch_size = Z.shape[0]
        
        # Expand seeds
        S = self.seeds.expand(batch_size, -1, -1)
        
        # Seeds attend to Z
        output = self.mab(S, Z, mask)
        
        return output


class SetTransformer(nn.Module):
    """
    Set Transformer
    
    Paper: "Set Transformer: A Framework for Attention-based 
           Permutation-Invariant Neural Networks" (Lee et al., 2019)
    
    Architecture:
    1. Input embedding
    2. Encoder: Stack of ISAB (or SAB)
    3. Decoder: PMA for pooling
    4. Output
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 2,
        n_inducing: int = 32,
        n_seeds: int = 1,
        dropout: float = 0.1,
        use_isab: bool = True       # True: ISAB (efficient), False: SAB
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            if use_isab:
                self.encoder_layers.append(
                    ISAB(d_model, n_heads, d_ff, n_inducing, dropout)
                )
            else:
                self.encoder_layers.append(
                    SAB(d_model, n_heads, d_ff, dropout)
                )
        
        # Decoder (pooling)
        self.decoder = nn.Sequential(
            PMA(d_model, n_heads, d_ff, n_seeds, dropout),
            SAB(d_model, n_heads, d_ff, dropout)  # Optional: process pooled output
        )
        
        # Final projection
        self.output_proj = nn.Linear(d_model * n_seeds, d_model)
        
        # Empty set embedding
        self.empty_embedding = nn.Parameter(torch.randn(d_model))
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n, input_dim)
            mask: (batch, n)
        
        Returns:
            output: (batch, d_model)
        """
        batch_size = x.shape[0]
        device = x.device
        
        if mask is None:
            mask = torch.ones(batch_size, x.shape[1], dtype=torch.bool, device=device)
        
        set_sizes = mask.sum(dim=1)
        empty_mask = set_sizes == 0
        
        # Input embedding
        h = self.input_embed(x)
        
        # Encoder
        for layer in self.encoder_layers:
            h = layer(h, mask)
        
        # Decoder (pooling)
        pooled = self.decoder[0](h, mask)  # PMA: (batch, n_seeds, d_model)
        pooled = self.decoder[1](pooled)    # SAB: (batch, n_seeds, d_model)
        
        # Flatten and project
        output = pooled.view(batch_size, -1)
        output = self.output_proj(output)
        
        # Handle empty sets
        if empty_mask.any():
            output[empty_mask] = self.empty_embedding
        
        return output


class SetTransformerEncoder(nn.Module):
    """
    Complete Set Transformer Encoder for Tree Packing
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        n_inducing: int = 32,
        include_global: bool = True
    ):
        super().__init__()
        
        self.include_global = include_global
        
        # Set Transformer (input: x, y, sin(θ), cos(θ))
        self.set_transformer = SetTransformer(
            input_dim=4,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_inducing=n_inducing,
            use_isab=True
        )
        
        # Global features
        if include_global:
            self.global_encoder = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(d_model + 64, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            )
    
    def preprocess(self, coords: torch.Tensor) -> torch.Tensor:
        """Koordinat preprocessing"""
        x, y, theta = coords[..., 0], coords[..., 1], coords[..., 2]
        
        # Simple normalization
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        return torch.stack([x_norm, y_norm, torch.sin(theta), torch.cos(theta)], dim=-1)
    
    def forward(
        self,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        x = self.preprocess(coordinates)
        set_embedding = self.set_transformer(x, mask)
        
        if self.include_global and global_features is not None:
            global_enc = self.global_encoder(global_features)
            combined = torch.cat([set_embedding, global_enc], dim=-1)
            return self.fusion(combined)
        
        return set_embedding
```

## 4.3 Set Transformer Özet

```
✅ Avantajlar:
- Transformer gücü + verimlilik (O(nm) vs O(n²))
- Inducing points ile "özet" öğrenir
- PMA ile esnek pooling
- Büyük setler için ölçeklenebilir

❌ Dezavantajlar:
- Daha karmaşık mimari
- Inducing point sayısı hyperparameter
- Küçük setlerde overhead
```

---

# Bölüm 5: Graph Neural Network (GNN)

## 5.1 Teori

GNN, ağaçları graph node'ları olarak modelleyip aralarındaki spatial ilişkileri edge olarak temsil eder.

### Graph Construction:
```
Nodes: Her ağaç bir node
Edges: Spatial proximity veya k-NN

Node features: (x, y, θ)
Edge features: mesafe, açı farkı, vb.
```

### Message Passing:
```
Her iterasyonda:
1. Her node komşularından mesaj alır
2. Mesajlar aggregate edilir
3. Node güncellenır

h_i^(l+1) = UPDATE(h_i^(l), AGGREGATE({h_j^(l) : j ∈ N(i)}))
```

## 5.2 Implementasyon

```python
class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) Layer
    
    Her node, komşularına farklı attention weight'leri verir
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        
        # Per-head transformation
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.randn(n_heads, out_features))
        self.a_dst = nn.Parameter(torch.randn(n_heads, out_features))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, in_features)
            adj: (batch, n_nodes, n_nodes) - Adjacency matrix
        
        Returns:
            output: (batch, n_nodes, out_features * n_heads) if concat
                    (batch, n_nodes, out_features) otherwise
        """
        batch_size, n_nodes, _ = x.shape
        
        # Linear transformation
        h = self.W(x)  # (batch, n, out_features * n_heads)
        h = h.view(batch_size, n_nodes, self.n_heads, self.out_features)
        
        # Attention scores
        # e_ij = LeakyReLU(a_src · h_i + a_dst · h_j)
        
        attn_src = (h * self.a_src).sum(dim=-1)  # (batch, n, n_heads)
        attn_dst = (h * self.a_dst).sum(dim=-1)  # (batch, n, n_heads)
        
        # Broadcasting: attn_src[i] + attn_dst[j] for all pairs
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)  # (batch, n, n, n_heads)
        attn = self.leaky_relu(attn)
        
        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1)
        attn = attn.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn = F.softmax(attn, dim=2)  # Normalize over neighbors
        attn = self.dropout(attn)
        
        # Apply attention
        # (batch, n, n, n_heads) × (batch, n, n_heads, out_features)
        h = h.permute(0, 2, 1, 3)  # (batch, n_heads, n, out_features)
        attn = attn.permute(0, 3, 1, 2)  # (batch, n_heads, n, n)
        
        out = torch.matmul(attn, h)  # (batch, n_heads, n, out_features)
        out = out.permute(0, 2, 1, 3)  # (batch, n, n_heads, out_features)
        
        if self.concat:
            out = out.reshape(batch_size, n_nodes, -1)
        else:
            out = out.mean(dim=2)
        
        return out


class GNNLayer(nn.Module):
    """
    General GNN Layer with multiple aggregation options
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregation: str = 'mean',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.aggregation = aggregation
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(in_features * 2, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, in_features)
            adj: (batch, n_nodes, n_nodes)
        """
        batch_size, n_nodes, in_features = x.shape
        
        # Create pairwise features for messages
        x_i = x.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (batch, n, n, f)
        x_j = x.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (batch, n, n, f)
        
        # Message: m_ij = MLP([h_i || h_j])
        messages = self.message_mlp(torch.cat([x_i, x_j], dim=-1))  # (batch, n, n, out)
        
        # Mask messages for non-edges
        messages = messages * adj.unsqueeze(-1)
        
        # Aggregate
        if self.aggregation == 'mean':
            degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            aggregated = messages.sum(dim=2) / degree
        elif self.aggregation == 'max':
            messages = messages.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
            aggregated = messages.max(dim=2)[0]
            aggregated = aggregated.masked_fill(aggregated == float('-inf'), 0)
        elif self.aggregation == 'sum':
            aggregated = messages.sum(dim=2)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Update
        updated = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        updated = self.norm(updated)
        updated = self.dropout(updated)
        
        return updated


class SpatialGNN(nn.Module):
    """
    Spatial GNN for Tree Packing
    
    Graph construction based on spatial proximity
    """
    
    def __init__(
        self,
        input_dim: int = 4,        # (x, y, sin(θ), cos(θ))
        hidden_dim: int = 128,
        output_dim: int = 256,
        n_layers: int = 3,
        k_neighbors: int = 5,      # k-NN graph
        use_gat: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.k_neighbors = k_neighbors
        self.use_gat = use_gat
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            
            if use_gat:
                self.gnn_layers.append(
                    GraphAttentionLayer(in_dim, out_dim // 4, n_heads=4, concat=True)
                )
            else:
                self.gnn_layers.append(
                    GNNLayer(in_dim, out_dim, aggregation='mean')
                )
        
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Empty set
        self.empty_embedding = nn.Parameter(torch.randn(output_dim))
    
    def build_knn_graph(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        k-NN graph oluştur
        
        Args:
            coords: (batch, n, 2) - x, y coordinates
            mask: (batch, n) - valid nodes
        
        Returns:
            adj: (batch, n, n) - adjacency matrix
        """
        batch_size, n_nodes, _ = coords.shape
        device = coords.device
        
        # Pairwise distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, n, n, 2)
        dist = (diff ** 2).sum(dim=-1).sqrt()  # (batch, n, n)
        
        # Mask invalid nodes
        if mask is not None:
            invalid_mask = ~(mask.unsqueeze(2) & mask.unsqueeze(1))
            dist = dist.masked_fill(invalid_mask, float('inf'))
        
        # Self-loops mask
        eye = torch.eye(n_nodes, device=device).unsqueeze(0)
        dist = dist.masked_fill(eye.bool(), float('inf'))
        
        # k-NN: find k smallest distances
        k = min(self.k_neighbors, n_nodes - 1)
        _, indices = dist.topk(k, dim=-1, largest=False)  # (batch, n, k)
        
        # Create adjacency matrix
        adj = torch.zeros(batch_size, n_nodes, n_nodes, device=device)
        
        # Scatter 1s at neighbor positions
        src = torch.arange(n_nodes, device=device).unsqueeze(0).unsqueeze(-1)
        src = src.expand(batch_size, -1, k)
        adj.scatter_(2, indices, 1.0)
        
        # Make symmetric
        adj = (adj + adj.transpose(1, 2)).clamp(max=1)
        
        # Add self-loops
        adj = adj + eye
        
        return adj
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n, input_dim) - Node features
            mask: (batch, n)
        
        Returns:
            output: (batch, output_dim)
        """
        batch_size, n_nodes, _ = x.shape
        device = x.device
        
        if mask is None:
            mask = torch.ones(batch_size, n_nodes, dtype=torch.bool, device=device)
        
        set_sizes = mask.sum(dim=1)
        empty_mask = set_sizes == 0
        
        # Build graph (using x, y coordinates)
        coords = x[..., :2]  # (batch, n, 2)
        adj = self.build_knn_graph(coords, mask)
        
        # Input embedding
        h = self.input_embed(x)
        
        # GNN layers
        for layer in self.gnn_layers:
            h = layer(h, adj)
        
        # Masked mean pooling (readout)
        h_masked = h * mask.unsqueeze(-1).float()
        h_sum = h_masked.sum(dim=1)
        h_mean = h_sum / (set_sizes.unsqueeze(-1).float() + 1e-8)
        
        # Output projection
        output = self.readout(h_mean)
        
        # Handle empty sets
        if empty_mask.any():
            output[empty_mask] = self.empty_embedding
        
        return output


class GNNStateEncoder(nn.Module):
    """
    Complete GNN-based State Encoder
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 256,
        n_layers: int = 3,
        k_neighbors: int = 5,
        include_global: bool = True
    ):
        super().__init__()
        
        self.include_global = include_global
        
        self.gnn = SpatialGNN(
            input_dim=4,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            k_neighbors=k_neighbors
        )
        
        if include_global:
            self.global_encoder = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(output_dim + 64, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
    
    def preprocess(self, coords: torch.Tensor) -> torch.Tensor:
        x, y, theta = coords[..., 0], coords[..., 1], coords[..., 2]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        return torch.stack([x_norm, y_norm, torch.sin(theta), torch.cos(theta)], dim=-1)
    
    def forward(
        self,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        x = self.preprocess(coordinates)
        graph_embedding = self.gnn(x, mask)
        
        if self.include_global and global_features is not None:
            global_enc = self.global_encoder(global_features)
            combined = torch.cat([graph_embedding, global_enc], dim=-1)
            return self.fusion(combined)
        
        return graph_embedding
```

## 5.3 GNN Özet

```
✅ Avantajlar:
- Spatial ilişkileri explicit olarak modeller
- k-NN ile yerel bağlam
- "Yakın ağaçlar" kavramı doğal
- Mesafe bilgisi edge feature olarak kullanılabilir

❌ Dezavantajlar:
- Graph construction ekstra adım
- k parametresi seçimi
- Uzak ağaçlar arası bilgi için çok layer gerekli
- Fully connected değil (information bottleneck)
```

---

# Bölüm 6: Karşılaştırma ve Öneriler

## 6.1 Tüm Mimarilerin Karşılaştırması

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           MİMARİ KARŞILAŞTIRMA                                │
├──────────────────┬────────────┬────────────┬────────────┬────────────────────┤
│                  │  DeepSets  │Transformer │ Set Trans. │       GNN          │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Permutation Inv. │     ✅     │     ✅     │     ✅     │        ✅          │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Variable Length  │     ✅     │     ✅     │     ✅     │        ✅          │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Pairwise Interact│     ❌     │     ✅     │     ✅     │        ✅          │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Complexity       │    O(n)    │   O(n²)    │   O(nm)    │      O(nk)         │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Implementation   │   Simple   │   Medium   │   Complex  │      Medium        │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Param Count      │    Low     │    High    │   Medium   │      Medium        │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Spatial Aware    │   Implicit │  Implicit  │  Implicit  │     Explicit       │
├──────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Best For         │Small sets, │ Any size,  │Large sets, │Spatial structure   │
│                  │baseline    │best perf.  │efficiency  │important           │
└──────────────────┴────────────┴────────────┴────────────┴────────────────────┘

n: eleman sayısı
m: inducing point sayısı (Set Transformer)
k: neighbor sayısı (GNN)
```

## 6.2 Tree Packing için Öneriler

### Senaryo 1: Az ağaç (n < 50)
```
Öneri: Transformer

Neden:
- O(n²) kabul edilebilir (50² = 2500)
- En güçlü representation
- Her ağaç herkesi görür
```

### Senaryo 2: Çok ağaç (n > 100)
```
Öneri: Set Transformer veya GNN

Neden:
- O(n²) çok yavaş
- Inducing points veya k-NN ile verimli
```

### Senaryo 3: Hızlı prototyping
```
Öneri: DeepSets

Neden:
- Basit implementasyon
- Hızlı training
- İyi baseline
```

### Senaryo 4: Spatial clustering önemli
```
Öneri: GNN

Neden:
- Yakınlık explicit olarak modellenir
- Cluster detection
- Local context
```

## 6.3 Final Öneri: Hybrid Approach

```python
class HybridStateEncoder(nn.Module):
    """
    Hybrid: Transformer + Local GNN features
    
    En iyi iki dünya:
    - Transformer: Global context
    - GNN: Local spatial features
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        k_neighbors: int = 5
    ):
        super().__init__()
        
        # Local features (GNN)
        self.local_gnn = SpatialGNN(
            input_dim=4,
            hidden_dim=64,
            output_dim=64,
            n_layers=2,
            k_neighbors=k_neighbors
        )
        
        # Global context (Transformer)
        self.global_transformer = TransformerSetEncoder(
            input_dim=4 + 64,  # Original + GNN features
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
    
    def forward(self, coords, mask=None, global_features=None):
        # Preprocess
        x = self.preprocess(coords)
        
        # Local GNN features
        local_features = self.local_gnn.get_node_features(x, mask)
        
        # Concatenate
        x_enhanced = torch.cat([x, local_features], dim=-1)
        
        # Global Transformer
        return self.global_transformer(x_enhanced, mask)
```

---

# Bölüm 7: Tam Kullanım Örneği

```python
def create_state_encoder(
    encoder_type: str = 'transformer',
    d_model: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function for state encoders
    """
    if encoder_type == 'deepsets':
        return DeepSetsEncoder(
            hidden_dim=d_model // 2,
            output_dim=d_model,
            **kwargs
        )
    elif encoder_type == 'transformer':
        return TransformerStateEncoder(
            d_model=d_model,
            **kwargs
        )
    elif encoder_type == 'set_transformer':
        return SetTransformerEncoder(
            d_model=d_model,
            **kwargs
        )
    elif encoder_type == 'gnn':
        return GNNStateEncoder(
            hidden_dim=d_model // 2,
            output_dim=d_model,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# Kullanım örneği
def example_usage():
    batch_size = 32
    max_trees = 50
    
    # Simulated data
    coordinates = torch.randn(batch_size, max_trees, 3)  # (x, y, θ)
    mask = torch.ones(batch_size, max_trees, dtype=torch.bool)
    mask[:, 30:] = False  # Son 20 pozisyon geçersiz
    
    global_features = torch.randn(batch_size, 4)  # n_placed, n_remaining, bbox, eff
    
    # Test each encoder
    encoders = {
        'deepsets': create_state_encoder('deepsets'),
        'transformer': create_state_encoder('transformer'),
        'set_transformer': create_state_encoder('set_transformer'),
        'gnn': create_state_encoder('gnn')
    }
    
    for name, encoder in encoders.items():
        output = encoder(coordinates, mask, global_features)
        print(f"{name}: output shape = {output.shape}")


if __name__ == "__main__":
    example_usage()
```

---

# Özet

| Mimari | Ne Zaman Kullan |
|--------|-----------------|
| **DeepSets** | Baseline, hızlı prototyping, basitlik |
| **Transformer** | En iyi performans, orta boyutlu setler |
| **Set Transformer** | Büyük setler, verimlilik gerekli |
| **GNN** | Spatial ilişkiler kritik, clustering |
| **Hybrid** | Maximum performans, complexity kabul edilebilir |

**Tree Packing için en iyi başlangıç:** Transformer (CLS token pooling ile)
