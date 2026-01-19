# 📦 Bin Packing Optimization via DRL - Paper Analizi ve Tree Packing'e Adaptasyonu

## Wang & Dong (2024) - arXiv:2403.12420

Bu doküman, paper'daki state representation yaklaşımını detaylı şekilde açıklar ve Tree Packing problemine nasıl adapte edebileceğimizi gösterir.

---

# Bölüm 1: Paper'ın Temel Yaklaşımı

## 1.1 Problem Tanımı

Paper 2D ve 3D Bin Packing Problem'i çözüyor:
- **Input:** n adet dikdörtgen (2D) veya küp (3D) nesne
- **Output:** Optimal yerleştirme sırası (packing order)
- **Hedef:** Space utilization'ı maksimize et, kutu sayısını minimize et

## 1.2 İki Aşamalı Yaklaşım

Paper problemi iki aşamaya ayırıyor:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      İKİ AŞAMALI YAKLAŞIM                           │
│                                                                     │
│   AŞAMA 1: SIRALAMA (DRL ile öğrenilen)                            │
│   ════════════════════════════════════                              │
│   Input: Nesnelerin boyut bilgileri                                │
│   Output: Optimal yerleştirme sırası                               │
│   Yöntem: Modified Pointer Network                                  │
│                                                                     │
│                          │                                          │
│                          ▼                                          │
│                                                                     │
│   AŞAMA 2: YERLEŞTİRME (Heuristic - Height Map Based)              │
│   ════════════════════════════════════════════════════              │
│   Input: Sıralı nesneler + Height Map                              │
│   Output: Her nesnenin (x, y, z) pozisyonu                         │
│   Yöntem: Bottom-Left + Minimum Height                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Kritik İçgörü:** Paper pozisyonu öğrenmek yerine SIRAYI öğreniyor. Pozisyon deterministic bir heuristic ile belirleniyor.

---

# Bölüm 2: State Representation (Girdi Yapısı)

## 2.1 Input Structure

Paper'daki state representation çok basit ve elegant:

### 2D BPP için:
```
X = [s₁, s₂, ..., sₙ]  →  2×n matrix

Her nesne sᵢ = (lᵢ, hᵢ)ᵀ  →  2D vector (length, height)

Örnek (n=5 nesne):
┌                     ┐
│ l₁  l₂  l₃  l₄  l₅ │   ← Uzunluklar
│ h₁  h₂  h₃  h₄  h₅ │   ← Yükseklikler
└                     ┘
```

### 3D BPP için:
```
X = [s₁, s₂, ..., sₙ]  →  3×n matrix

Her nesne sᵢ = (lᵢ, wᵢ, hᵢ)ᵀ  →  3D vector (length, width, height)

Örnek (n=5 nesne):
┌                     ┐
│ l₁  l₂  l₃  l₄  l₅ │   ← Uzunluklar
│ w₁  w₂  w₃  w₄  w₅ │   ← Genişlikler
│ h₁  h₂  h₃  h₄  h₅ │   ← Yükseklikler
└                     ┘
```

## 2.2 Output Structure

```
Y = {ρ₁, ρ₂, ..., ρₙ}  →  Permutation

Örnek: Y = {3, 1, 5, 4, 2}
"Sırasıyla Object 3, Object 1, Object 5, Object 4, Object 2 yerleştir"
```

## 2.3 Height Map (Placement State)

Height Map, kutunun mevcut doluluk durumunu temsil eder:

### 2D Height Map:
```
H₂ ∈ ℤᴸ  →  1D array (kutu uzunluğu boyunca)

Örnek: L=5 birimlik kutu
┌─────────────────────┐
│    ████             │  h=5
│    ████  ██████     │  h=3
│    ████  ██████     │  h=2
│    ████  ██████     │  
│████████████████     │  
└─────────────────────┘
 0   1   2   3   4      ← X pozisyonu

H₂ = [5, 5, 2, 3, 3]
```

### 3D Height Map:
```
H₃ ∈ ℤᴸˣᵂ  →  2D matrix (kutu tabanı)

Örnek: 5×5 kutu tabanı
┌─────────────────────┐
│  3   3   2   2   0  │
│  3   3   2   2   0  │
│  5   5   4   4   0  │
│  5   5   4   4   0  │
│  1   1   1   0   0  │
└─────────────────────┘
Her hücre o pozisyondaki yığın yüksekliğini gösterir
```

---

# Bölüm 3: Network Mimarisi (Modified Pointer Network)

## 3.1 Genel Mimari

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODIFIED POINTER NETWORK                         │
│                                                                     │
│   INPUT: X = [s₁, s₂, ..., sₙ]  (Nesne boyutları)                  │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────┐                          │
│   │           ENCODER                    │                          │
│   │     (1D Convolution Layer)          │                          │
│   │                                      │                          │
│   │   Input channels: M (2 veya 3)      │                          │
│   │   Output: n × d_h matrix            │                          │
│   │   (n nesne, d_h dimensional embed)  │                          │
│   └─────────────────┬───────────────────┘                          │
│                     │                                               │
│          e₁, e₂, ..., eₙ  (encoding states)                        │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────┐                          │
│   │           DECODER (GRU)             │                          │
│   │                                      │                          │
│   │   t=0: y₀ = 0⃗  (zero vector)        │                          │
│   │   t=1: y₁ = s_ρ₁ (seçilen nesne)    │                          │
│   │   ...                                │                          │
│   │   Hidden state: h₁, h₂, ..., hₙ     │                          │
│   └─────────────────┬───────────────────┘                          │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────┐                          │
│   │       ATTENTION MODULE              │                          │
│   │                                      │                          │
│   │   u_j^(t+1) = vᵀ tanh(W₁eⱼ + W₂hₜ₊₁)│                          │
│   │                                      │                          │
│   │   P(ρₜ₊₁=j) = softmax(u_j^(t+1))    │                          │
│   │   + Masking (seçilmiş nesneler)     │                          │
│   └─────────────────┬───────────────────┘                          │
│                     │                                               │
│                     ▼                                               │
│   OUTPUT: Y = {ρ₁, ρ₂, ..., ρₙ}  (Packing Order)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.2 Encoder Detayları

```python
class PackingEncoder(nn.Module):
    """
    1D Convolution based encoder
    
    Neden 1D Conv?
    - Her nesnenin boyut vektörünü bağımsız olarak encode eder
    - Parametre sayısı nesne sayısından bağımsız
    - Tüm nesneler aynı weight'leri paylaşır (weight sharing)
    """
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # input_dim: 2 (2D) veya 3 (3D)
        # hidden_dim: embedding dimension (örn: 128)
        
        self.conv = nn.Conv1d(
            in_channels=input_dim,    # 2 veya 3
            out_channels=hidden_dim,  # d_h
            kernel_size=1             # Her nesneyi bağımsız encode et
        )
    
    def forward(self, x):
        """
        Input: x shape (batch, input_dim, n_objects)
               Örnek 2D: (batch, 2, n)  → her sütun bir nesne [l, h]
        
        Output: encodings shape (batch, n_objects, hidden_dim)
        """
        # Conv1d: (batch, in_channels, seq_len) → (batch, out_channels, seq_len)
        encoded = self.conv(x)  # (batch, d_h, n)
        
        # Transpose for decoder: (batch, n, d_h)
        encoded = encoded.permute(0, 2, 1)
        
        return encoded  # Her nesne için d_h boyutlu embedding
```

## 3.3 Decoder (GRU) Detayları

```python
class PackingDecoder(nn.Module):
    """
    GRU-based decoder with attention
    
    Neden GRU?
    - Daha önce seçilen nesneleri hatırlaması gerekiyor
    - Sıralı seçim yapıyor (autoregressive)
    - LSTM'den daha basit, benzer performans
    """
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # GRU: seçilen nesnenin boyutlarını input olarak alır
        self.gru = nn.GRU(
            input_size=input_dim,     # 2 veya 3 (nesne boyutları)
            hidden_size=hidden_dim,   # d_h
            batch_first=True
        )
        
        # Initial hidden state projection (optional)
        self.init_hidden = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, selected_object, hidden_state):
        """
        Tek bir decoding step
        
        Input:
            selected_object: (batch, input_dim) - son seçilen nesnenin boyutları
            hidden_state: (1, batch, hidden_dim) - GRU hidden state
        
        Output:
            output: (batch, hidden_dim) - attention için kullanılacak
            new_hidden: (1, batch, hidden_dim) - güncellenen hidden state
        """
        # selected_object'i (batch, 1, input_dim) yap
        x = selected_object.unsqueeze(1)
        
        output, new_hidden = self.gru(x, hidden_state)
        
        return output.squeeze(1), new_hidden
```

## 3.4 Attention Module

```python
class AttentionModule(nn.Module):
    """
    Pointer Network tarzı attention
    
    Formül:
    u_j^(t+1) = vᵀ tanh(W₁ eⱼ + W₂ hₜ₊₁)
    P(ρₜ₊₁ = j) = softmax(u) * mask
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Encoder için
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Decoder için
        self.v = nn.Linear(hidden_dim, 1, bias=False)            # Score için
    
    def forward(self, encoder_outputs, decoder_hidden, mask):
        """
        Input:
            encoder_outputs: (batch, n_objects, hidden_dim) - tüm nesnelerin encoding'leri
            decoder_hidden: (batch, hidden_dim) - GRU'nun çıktısı
            mask: (batch, n_objects) - True = seçilebilir, False = zaten seçilmiş
        
        Output:
            probs: (batch, n_objects) - her nesnenin seçilme olasılığı
        """
        batch_size, n_objects, hidden_dim = encoder_outputs.shape
        
        # W₁ eⱼ: her nesne için
        encoder_proj = self.W1(encoder_outputs)  # (batch, n, hidden)
        
        # W₂ hₜ₊₁: decoder hidden state
        decoder_proj = self.W2(decoder_hidden)   # (batch, hidden)
        decoder_proj = decoder_proj.unsqueeze(1) # (batch, 1, hidden)
        
        # tanh(W₁ eⱼ + W₂ hₜ₊₁)
        combined = torch.tanh(encoder_proj + decoder_proj)  # (batch, n, hidden)
        
        # vᵀ tanh(...)
        scores = self.v(combined).squeeze(-1)  # (batch, n)
        
        # Masking: seçilmiş nesnelere -inf
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax → probability distribution
        probs = F.softmax(scores, dim=-1)
        
        return probs
```

## 3.5 Complete Pointer Network

```python
class ModifiedPointerNetwork(nn.Module):
    """
    Paper'daki tam Modified Pointer Network implementasyonu
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim: 2 (2D BPP) veya 3 (3D BPP)
        hidden_dim: embedding boyutu (örn: 128)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Components
        self.encoder = PackingEncoder(input_dim, hidden_dim)
        self.decoder = PackingDecoder(input_dim, hidden_dim)
        self.attention = AttentionModule(hidden_dim)
        
        # Initial decoder input (zero vector için)
        self.start_token = nn.Parameter(torch.zeros(input_dim))
    
    def forward(self, objects, greedy=False):
        """
        Full forward pass - packing order üret
        
        Input:
            objects: (batch, input_dim, n_objects)
                     Her sütun bir nesnenin boyutları
            greedy: True = en yüksek olasılıklı nesneyi seç
                    False = olasılığa göre sample et
        
        Output:
            order: (batch, n_objects) - seçim sırası
            log_probs: (batch, n_objects) - her seçimin log probability'si
        """
        batch_size = objects.shape[0]
        n_objects = objects.shape[2]
        device = objects.device
        
        # ═══════════════════════════════════════════════════════════
        # ENCODING
        # ═══════════════════════════════════════════════════════════
        encoder_outputs = self.encoder(objects)  # (batch, n, hidden)
        
        # ═══════════════════════════════════════════════════════════
        # DECODING (autoregressive)
        # ═══════════════════════════════════════════════════════════
        
        # Initialize
        hidden = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        mask = torch.ones(batch_size, n_objects, dtype=torch.bool, device=device)
        
        # İlk input: zero vector (paper'da y₀ = 0⃗)
        decoder_input = self.start_token.expand(batch_size, -1)
        
        # Outputs
        order = []
        log_probs = []
        
        # n adım: her adımda bir nesne seç
        for t in range(n_objects):
            # Decoder step
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            
            # Attention → probability distribution
            probs = self.attention(encoder_outputs, decoder_output, mask)
            
            # Select next object
            if greedy:
                selected = probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                selected = dist.sample()
            
            # Log probability
            log_prob = torch.log(probs.gather(1, selected.unsqueeze(1)) + 1e-8)
            
            # Store
            order.append(selected)
            log_probs.append(log_prob.squeeze(1))
            
            # Update mask (seçilen nesneyi maskele)
            mask = mask.scatter(1, selected.unsqueeze(1), False)
            
            # Next decoder input: seçilen nesnenin boyutları
            # objects: (batch, input_dim, n)
            # selected: (batch,)
            selected_expanded = selected.unsqueeze(1).unsqueeze(1).expand(-1, self.input_dim, -1)
            decoder_input = objects.gather(2, selected_expanded).squeeze(2)
        
        order = torch.stack(order, dim=1)      # (batch, n)
        log_probs = torch.stack(log_probs, dim=1)  # (batch, n)
        
        return order, log_probs
```

---

# Bölüm 4: Placement Strategy (Height Map Based)

## 4.1 Height Map Güncelleme Algoritması

```python
class HeightMapPlacement:
    """
    Paper'daki Height Map tabanlı placement strategy
    
    Prensip: "Bottom-Left with Minimum Height"
    1. Tüm allowable position'ları bul
    2. En düşük z (height) değerine sahip olanı seç
    3. Eşitlik varsa, en küçük x (sonra y) değerine sahip olanı seç
    """
    
    def __init__(self, box_length, box_width=None, box_height=None):
        """
        2D: sadece box_length ve box_height
        3D: box_length, box_width, box_height
        """
        self.L = box_length
        self.W = box_width  # None for 2D
        self.H = box_height
        
        self.is_3d = box_width is not None
        
        # Height map initialization
        if self.is_3d:
            self.height_map = np.zeros((self.L, self.W), dtype=np.int32)
        else:
            self.height_map = np.zeros(self.L, dtype=np.int32)
    
    def find_allowable_positions_2d(self, obj_length, obj_height):
        """
        2D: Dikdörtgen nesne için allowable position'ları bul
        
        Allowable position: 
        - Nesne kutuya sığmalı
        - Çakışma olmamalı
        - Stability: bottom'un %50'si desteklenmeli
        """
        allowable = []
        
        for x in range(self.L - obj_length + 1):
            # Bu x pozisyonunda nesnenin altındaki height map değerleri
            heights_under = self.height_map[x:x + obj_length]
            
            # Nesnenin oturacağı z pozisyonu
            z = heights_under.max()
            
            # Kutu yüksekliğini aşıyor mu?
            if z + obj_height > self.H:
                continue
            
            # Stability check: en az %50'si desteklenmeli
            supported_count = np.sum(heights_under == z)
            if supported_count < obj_length * 0.5:
                continue
            
            allowable.append((x, z))
        
        return allowable
    
    def find_allowable_positions_3d(self, obj_l, obj_w, obj_h):
        """
        3D: Küp nesne için allowable position'ları bul
        """
        allowable = []
        
        for x in range(self.L - obj_l + 1):
            for y in range(self.W - obj_w + 1):
                # Bu (x,y) pozisyonunda nesnenin altındaki height map
                heights_under = self.height_map[x:x + obj_l, y:y + obj_w]
                
                # Nesnenin oturacağı z pozisyonu
                z = heights_under.max()
                
                # Kutu yüksekliğini aşıyor mu?
                if z + obj_h > self.H:
                    continue
                
                # Stability check
                supported_count = np.sum(heights_under == z)
                total_bottom = obj_l * obj_w
                if supported_count < total_bottom * 0.5:
                    continue
                
                allowable.append((x, y, z))
        
        return allowable
    
    def select_target_position(self, allowable_positions):
        """
        Allowable position'lar arasından target'ı seç
        
        Kriter (öncelik sırasıyla):
        1. Minimum z (height)
        2. Minimum x
        3. Minimum y (3D için)
        """
        if not allowable_positions:
            return None  # Sığmıyor, yeni kutu gerekli
        
        if self.is_3d:
            # Sort by (z, y, x)
            sorted_positions = sorted(allowable_positions, key=lambda p: (p[2], p[1], p[0]))
        else:
            # Sort by (z, x)
            sorted_positions = sorted(allowable_positions, key=lambda p: (p[1], p[0]))
        
        return sorted_positions[0]
    
    def place_object_2d(self, obj_length, obj_height):
        """
        2D nesne yerleştir ve height map güncelle
        """
        allowable = self.find_allowable_positions_2d(obj_length, obj_height)
        target = self.select_target_position(allowable)
        
        if target is None:
            return None  # Yeni kutu gerekli
        
        x, z = target
        
        # Height map güncelle
        new_height = z + obj_height
        self.height_map[x:x + obj_length] = np.maximum(
            self.height_map[x:x + obj_length],
            new_height
        )
        
        return (x, z)
    
    def place_object_3d(self, obj_l, obj_w, obj_h):
        """
        3D nesne yerleştir ve height map güncelle
        """
        allowable = self.find_allowable_positions_3d(obj_l, obj_w, obj_h)
        target = self.select_target_position(allowable)
        
        if target is None:
            return None  # Yeni kutu gerekli
        
        x, y, z = target
        
        # Height map güncelle
        new_height = z + obj_h
        self.height_map[x:x + obj_l, y:y + obj_w] = np.maximum(
            self.height_map[x:x + obj_l, y:y + obj_w],
            new_height
        )
        
        return (x, y, z)
    
    def reset(self):
        """Height map'i sıfırla (yeni kutu)"""
        if self.is_3d:
            self.height_map = np.zeros((self.L, self.W), dtype=np.int32)
        else:
            self.height_map = np.zeros(self.L, dtype=np.int32)
    
    def compute_compactness(self, placed_objects_volume):
        """
        Compactness = Toplam nesne hacmi / (Taban alanı × Max height)
        """
        max_height = self.height_map.max()
        if max_height == 0:
            return 1.0
        
        if self.is_3d:
            region_volume = self.L * self.W * max_height
        else:
            region_volume = self.L * max_height
        
        return placed_objects_volume / region_volume
    
    def compute_pyramid(self, placed_objects_volume):
        """
        Pyramid = Toplam nesne hacmi / Height map toplamı
        
        Height map toplamı = nesnelerin tabana projeksiyonu
        """
        projection_volume = self.height_map.sum()
        if projection_volume == 0:
            return 1.0
        
        return placed_objects_volume / projection_volume
```

---

# Bölüm 5: Reward ve Loss Functions

## 5.1 Reward Function

```python
def compute_reward(boxes_info, alpha=0.5, beta=0.5):
    """
    Paper'daki reward function
    
    R = 5 × [α × (1 - Avg_Compactness) + β × (1 - Avg_Pyramid)]
    
    Neden bu formül?
    - Compactness yüksek → iyi (1 - C düşük → R yüksek)
    - Pyramid yüksek → iyi (1 - P düşük → R yüksek)
    - Usage number of boxes implicit olarak minimize ediliyor
      (daha iyi packing → daha az kutu → daha yüksek C ve P)
    
    Args:
        boxes_info: Her kutu için (compactness, pyramid) listesi
        alpha, beta: Hyperparameters (default: 0.5, 0.5)
    """
    k = len(boxes_info)  # Kullanılan kutu sayısı
    
    if k == 0:
        return 0
    
    total_compactness = sum(info['compactness'] for info in boxes_info)
    total_pyramid = sum(info['pyramid'] for info in boxes_info)
    
    avg_compactness = total_compactness / k
    avg_pyramid = total_pyramid / k
    
    reward = 5 * (alpha * (1 - avg_compactness) + beta * (1 - avg_pyramid))
    
    # NOT: Paper'da reward minimize edilecek şekilde tanımlanmış
    # Biz maximize etmek istiyorsak:
    # reward = 5 * (alpha * avg_compactness + beta * avg_pyramid)
    
    return reward
```

## 5.2 Actor-Critic Loss

```python
class ActorCriticTrainer:
    """
    Paper'daki on-policy actor-critic framework
    """
    
    def __init__(self, actor_network, critic_network, actor_lr=5e-4, critic_lr=5e-4):
        self.actor = actor_network    # Modified Pointer Network
        self.critic = critic_network  # Value estimator
        
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    def compute_loss(self, objects, orders, log_probs, rewards):
        """
        Actor Loss:
        L_actor = (R - V_θ(X)) × Σ log P(ρ_{t+1} | ρ_0, ..., ρ_t)
        
        Critic Loss:
        L_critic = (R - V_θ(X))²
        
        Args:
            objects: (batch, input_dim, n) - nesne boyutları
            orders: (batch, n) - üretilen packing order
            log_probs: (batch, n) - her seçimin log probability'si
            rewards: (batch,) - her instance için hesaplanan reward
        """
        # Critic prediction
        expected_reward = self.critic(objects)  # (batch, 1)
        expected_reward = expected_reward.squeeze(1)
        
        # Advantage
        advantage = rewards - expected_reward.detach()
        
        # Actor loss (policy gradient)
        # Σ log P(ρ_{t+1} | ρ_0, ..., ρ_t)
        total_log_prob = log_probs.sum(dim=1)  # (batch,)
        actor_loss = -(advantage * total_log_prob).mean()
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(expected_reward, rewards)
        
        return actor_loss, critic_loss
    
    def train_step(self, objects_batch):
        """
        Tek bir training step
        """
        # Forward pass: packing order üret
        orders, log_probs = self.actor(objects_batch, greedy=False)
        
        # Placement + Reward hesapla
        rewards = []
        for i in range(objects_batch.shape[0]):
            objects_i = objects_batch[i]  # (input_dim, n)
            order_i = orders[i]           # (n,)
            
            reward = self.evaluate_packing(objects_i, order_i)
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=objects_batch.device)
        
        # Loss hesapla
        actor_loss, critic_loss = self.compute_loss(
            objects_batch, orders, log_probs, rewards
        )
        
        # Update
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        (actor_loss + critic_loss).backward()
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def evaluate_packing(self, objects, order):
        """
        Verilen sırayla placement yap ve reward hesapla
        """
        # objects: (input_dim, n)
        # order: (n,)
        
        placer = HeightMapPlacement(box_length=10, box_height=10)
        
        boxes_info = []
        current_box_volume = 0
        
        for idx in order:
            obj = objects[:, idx].cpu().numpy()
            
            if len(obj) == 2:  # 2D
                l, h = obj
                position = placer.place_object_2d(int(l), int(h))
                obj_volume = l * h
            else:  # 3D
                l, w, h = obj
                position = placer.place_object_3d(int(l), int(w), int(h))
                obj_volume = l * w * h
            
            if position is None:
                # Mevcut kutuyu kaydet
                if current_box_volume > 0:
                    boxes_info.append({
                        'compactness': placer.compute_compactness(current_box_volume),
                        'pyramid': placer.compute_pyramid(current_box_volume)
                    })
                
                # Yeni kutu başlat
                placer.reset()
                position = placer.place_object_2d(int(l), int(h)) if len(obj) == 2 \
                          else placer.place_object_3d(int(l), int(w), int(h))
                current_box_volume = 0
            
            current_box_volume += obj_volume
        
        # Son kutuyu kaydet
        if current_box_volume > 0:
            boxes_info.append({
                'compactness': placer.compute_compactness(current_box_volume),
                'pyramid': placer.compute_pyramid(current_box_volume)
            })
        
        return compute_reward(boxes_info)
```

---

# Bölüm 6: Critic Network

```python
class CriticNetwork(nn.Module):
    """
    Paper'daki Critic Network
    
    Yapı:
    - İlk 1D Conv layer actor'ün encoder'ı ile paylaşımlı
    - Sonraki 3 Conv layer
    - Global pooling + FC → value
    """
    
    def __init__(self, input_dim, hidden_dim, shared_encoder):
        super().__init__()
        
        # Shared encoder (actor ile paylaşımlı)
        self.shared_encoder = shared_encoder
        
        # Additional conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU()
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, objects):
        """
        Input: objects (batch, input_dim, n)
        Output: expected_reward (batch, 1)
        """
        # Shared encoding
        encoded = self.shared_encoder.conv(objects)  # (batch, hidden, n)
        
        # Additional processing
        features = self.conv_layers(encoded)  # (batch, hidden, n)
        
        # Global average pooling
        pooled = features.mean(dim=2)  # (batch, hidden)
        
        # Value
        value = self.value_head(pooled)  # (batch, 1)
        
        return value
```

---

# Bölüm 7: Tree Packing'e Adaptasyon

## 7.1 Farklar ve Benzerlikler

| Aspect | Paper (Bin Packing) | Tree Packing |
|--------|---------------------|--------------|
| Şekil | Dikdörtgen/Küp | Üçgen (Triangle) |
| Boyut | 2D veya 3D | 2D |
| Rotasyon | Yok | Var (0°-360°) |
| Kutu | Sabit boyutlu | Bounding box (minimize) |
| Hedef | Kutu sayısını minimize | Bbox alanını minimize |

## 7.2 Adaptasyon Stratejisi

### State Representation için Değişiklikler:

```python
class TreePackingState:
    """
    Tree Packing için state representation
    
    Paper'dan ilham alarak:
    1. Object features: Her ağacın özellikleri
    2. Height map: Mevcut yerleşim durumu
    3. Global stats: Bbox, efficiency vs.
    """
    
    @staticmethod
    def create_object_features(trees):
        """
        Paper tarzı: Her ağaç için feature vector
        
        2D BPP: [length, height]
        Tree Packing: [area, width, height, aspect_ratio, ...]
        """
        features = []
        for tree in trees:
            # Ağacın bounding box'ı
            bbox = tree.bounds  # (minx, miny, maxx, maxy)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            feature = [
                tree.area,           # Alan
                width,               # Genişlik
                height,              # Yükseklik
                width / height,      # Aspect ratio
                tree.centroid.x,     # Centroid x (normalize edilecek)
                tree.centroid.y,     # Centroid y (normalize edilecek)
            ]
            features.append(feature)
        
        return np.array(features).T  # (feature_dim, n_trees) - paper formatı
    
    @staticmethod
    def create_height_map_2d(placed_trees, grid_resolution=64):
        """
        2D Height Map - Paper'daki gibi ama üçgenler için
        
        Fark: Dikdörtgen yerine üçgen şekilleri kaydediyoruz
        """
        height_map = np.zeros(grid_resolution, dtype=np.float32)
        
        for tree_info in placed_trees:
            polygon = tree_info['polygon']  # Shapely polygon
            
            # Her x pozisyonu için max y'yi bul
            for i in range(grid_resolution):
                x = i * (bbox_width / grid_resolution)
                
                # Bu x'teki dikey çizgi
                line = LineString([(x, 0), (x, bbox_height)])
                
                # Polygon ile kesişim
                intersection = polygon.intersection(line)
                
                if not intersection.is_empty:
                    # Max y değerini al
                    if hasattr(intersection, 'bounds'):
                        max_y = intersection.bounds[3]
                        height_map[i] = max(height_map[i], max_y)
        
        return height_map
```

### Modified Pointer Network Adaptasyonu:

```python
class TreePackingPointerNetwork(nn.Module):
    """
    Tree Packing için Modified Pointer Network
    
    Değişiklikler:
    1. Input dim: 2/3 yerine 6+ (daha zengin tree features)
    2. Rotation handling: Her ağaç için rotation da seçilmeli
    3. Placement: Height map based ama üçgenler için
    """
    
    def __init__(self, tree_feature_dim=6, hidden_dim=128, n_rotations=8):
        super().__init__()
        
        self.tree_feature_dim = tree_feature_dim
        self.hidden_dim = hidden_dim
        self.n_rotations = n_rotations
        
        # Encoder
        self.encoder = nn.Conv1d(tree_feature_dim, hidden_dim, 1)
        
        # Decoder
        self.decoder_gru = nn.GRU(tree_feature_dim, hidden_dim, batch_first=True)
        
        # Attention (tree selection)
        self.tree_attention = AttentionModule(hidden_dim)
        
        # Rotation head (seçilen ağaç için rotation)
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_rotations)
        )
    
    def forward(self, tree_features, greedy=False):
        """
        Input: tree_features (batch, feature_dim, n_trees)
        Output: 
            tree_order: (batch, n_trees) - seçim sırası
            rotations: (batch, n_trees) - her ağacın rotasyonu
            log_probs: (batch, n_trees) - combined log probabilities
        """
        batch_size = tree_features.shape[0]
        n_trees = tree_features.shape[2]
        device = tree_features.device
        
        # Encoding
        encoded = self.encoder(tree_features)  # (batch, hidden, n)
        encoded = encoded.permute(0, 2, 1)     # (batch, n, hidden)
        
        # Decoding
        hidden = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        mask = torch.ones(batch_size, n_trees, dtype=torch.bool, device=device)
        decoder_input = torch.zeros(batch_size, self.tree_feature_dim, device=device)
        
        tree_order = []
        rotations = []
        log_probs = []
        
        for t in range(n_trees):
            # Decoder step
            dec_out, hidden = self.decoder_gru(decoder_input.unsqueeze(1), hidden)
            dec_out = dec_out.squeeze(1)  # (batch, hidden)
            
            # Tree selection via attention
            tree_probs = self.tree_attention(encoded, dec_out, mask)
            
            if greedy:
                selected_tree = tree_probs.argmax(dim=-1)
            else:
                selected_tree = torch.distributions.Categorical(tree_probs).sample()
            
            # Get selected tree's encoding
            selected_encoding = encoded.gather(
                1, selected_tree.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            ).squeeze(1)  # (batch, hidden)
            
            # Rotation selection
            rotation_input = torch.cat([dec_out, selected_encoding], dim=-1)
            rotation_logits = self.rotation_head(rotation_input)
            rotation_probs = F.softmax(rotation_logits, dim=-1)
            
            if greedy:
                selected_rotation = rotation_probs.argmax(dim=-1)
            else:
                selected_rotation = torch.distributions.Categorical(rotation_probs).sample()
            
            # Log probabilities
            tree_log_prob = torch.log(tree_probs.gather(1, selected_tree.unsqueeze(1)) + 1e-8)
            rot_log_prob = torch.log(rotation_probs.gather(1, selected_rotation.unsqueeze(1)) + 1e-8)
            combined_log_prob = tree_log_prob + rot_log_prob
            
            # Store
            tree_order.append(selected_tree)
            rotations.append(selected_rotation)
            log_probs.append(combined_log_prob.squeeze(1))
            
            # Update
            mask = mask.scatter(1, selected_tree.unsqueeze(1), False)
            selected_features = tree_features.gather(
                2, selected_tree.unsqueeze(1).unsqueeze(1).expand(-1, self.tree_feature_dim, -1)
            ).squeeze(2)
            decoder_input = selected_features
        
        return (
            torch.stack(tree_order, dim=1),
            torch.stack(rotations, dim=1),
            torch.stack(log_probs, dim=1)
        )
```

### Tree Placement Strategy:

```python
class TreePlacementStrategy:
    """
    Üçgen ağaçlar için placement strategy
    
    Paper'dan farklı olarak:
    - Sabit kutu yok, bounding box minimize ediliyor
    - Üçgen şekiller için NFP kullanılabilir
    """
    
    def __init__(self, n_rotations=8):
        self.n_rotations = n_rotations
        self.rotation_angles = np.linspace(0, 2*np.pi, n_rotations, endpoint=False)
        self.placed_trees = []
        self.bbox = None
    
    def place_tree(self, tree_polygon, rotation_idx):
        """
        Ağacı belirtilen rotasyonla yerleştir
        """
        # Rotate
        angle = self.rotation_angles[rotation_idx]
        rotated = shapely.affinity.rotate(tree_polygon, angle, origin='centroid', use_radians=True)
        
        if not self.placed_trees:
            # İlk ağaç: origin'e yerleştir
            translated = shapely.affinity.translate(
                rotated,
                xoff=-rotated.bounds[0],
                yoff=-rotated.bounds[1]
            )
            self.placed_trees.append(translated)
            self._update_bbox()
            return (0, 0, angle)
        
        # Allowable positions bul (NFP-based veya grid search)
        best_position = self._find_best_position(rotated)
        
        if best_position is None:
            return None
        
        x, y = best_position
        translated = shapely.affinity.translate(rotated, xoff=x, yoff=y)
        self.placed_trees.append(translated)
        self._update_bbox()
        
        return (x, y, angle)
    
    def _find_best_position(self, polygon, grid_step=1):
        """
        En iyi pozisyonu bul
        
        Kriter: Bounding box artışını minimize et
        """
        best_pos = None
        best_bbox_area = float('inf')
        
        # Search area: mevcut bbox etrafında
        search_minx = self.bbox[0] - polygon.bounds[2]
        search_maxx = self.bbox[2] + 10
        search_miny = self.bbox[1] - polygon.bounds[3]
        search_maxy = self.bbox[3] + 10
        
        for x in np.arange(search_minx, search_maxx, grid_step):
            for y in np.arange(search_miny, search_maxy, grid_step):
                translated = shapely.affinity.translate(polygon, xoff=x, yoff=y)
                
                # Collision check
                collision = False
                for placed in self.placed_trees:
                    if translated.intersects(placed):
                        collision = True
                        break
                
                if collision:
                    continue
                
                # Yeni bbox hesapla
                new_bbox = self._compute_new_bbox(translated)
                new_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                
                if new_area < best_bbox_area:
                    best_bbox_area = new_area
                    best_pos = (x, y)
        
        return best_pos
    
    def _compute_new_bbox(self, new_polygon):
        """Yeni polygon eklenirse bbox ne olur?"""
        all_bounds = [p.bounds for p in self.placed_trees] + [new_polygon.bounds]
        
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        
        return (minx, miny, maxx, maxy)
    
    def _update_bbox(self):
        """Mevcut bbox'ı güncelle"""
        all_bounds = [p.bounds for p in self.placed_trees]
        
        self.bbox = (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds)
        )
    
    def get_efficiency(self):
        """Packing efficiency"""
        if not self.placed_trees:
            return 0
        
        total_area = sum(p.area for p in self.placed_trees)
        bbox_area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        
        return total_area / bbox_area
```

## 7.3 Reward Function Adaptasyonu

```python
def tree_packing_reward(placed_trees, final_bbox, target_bbox=None):
    """
    Tree Packing için reward function
    
    Paper'dan ilham:
    - Compactness → Efficiency
    - Pyramid → Kullanılmıyor (farklı problem)
    - Box count → Bbox area
    """
    if not placed_trees:
        return 0
    
    # Efficiency (paper'daki compactness'a benzer)
    total_tree_area = sum(p.area for p in placed_trees)
    bbox_area = (final_bbox[2] - final_bbox[0]) * (final_bbox[3] - final_bbox[1])
    efficiency = total_tree_area / bbox_area
    
    # Bbox penalty (paper'daki box count yerine)
    if target_bbox is not None:
        bbox_ratio = bbox_area / target_bbox
        bbox_penalty = max(0, bbox_ratio - 1)  # Target'ı aşarsa penalty
    else:
        bbox_penalty = 0
    
    # Combined reward
    reward = efficiency - 0.5 * bbox_penalty
    
    return reward
```

---

# Bölüm 8: Tam Training Pipeline

```python
class TreePackingTrainer:
    """
    Complete training pipeline
    """
    
    def __init__(self, n_trees_range=(10, 50), hidden_dim=128, n_rotations=8):
        self.n_trees_range = n_trees_range
        
        # Networks
        self.actor = TreePackingPointerNetwork(
            tree_feature_dim=6,
            hidden_dim=hidden_dim,
            n_rotations=n_rotations
        )
        
        self.critic = CriticNetwork(
            input_dim=6,
            hidden_dim=hidden_dim,
            shared_encoder=self.actor.encoder
        )
        
        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        
        # Placement
        self.placer = TreePlacementStrategy(n_rotations=n_rotations)
    
    def generate_batch(self, batch_size):
        """Random tree batch üret"""
        trees_batch = []
        
        for _ in range(batch_size):
            n_trees = np.random.randint(*self.n_trees_range)
            trees = generate_random_trees(n_trees)  # Implementation needed
            features = TreePackingState.create_object_features(trees)
            trees_batch.append((trees, features))
        
        return trees_batch
    
    def train_step(self, batch):
        """Single training step"""
        # Prepare batch
        features_list = [item[1] for item in batch]
        max_n = max(f.shape[1] for f in features_list)
        
        # Pad to same length
        padded = []
        for f in features_list:
            if f.shape[1] < max_n:
                pad = np.zeros((f.shape[0], max_n - f.shape[1]))
                f = np.concatenate([f, pad], axis=1)
            padded.append(f)
        
        features_tensor = torch.tensor(np.stack(padded), dtype=torch.float32)
        
        # Forward
        tree_orders, rotations, log_probs = self.actor(features_tensor, greedy=False)
        
        # Evaluate packing
        rewards = []
        for i, (trees, _) in enumerate(batch):
            order = tree_orders[i].cpu().numpy()
            rots = rotations[i].cpu().numpy()
            
            self.placer = TreePlacementStrategy()  # Reset
            
            for j in range(len(trees)):
                if order[j] >= len(trees):
                    continue
                tree = trees[order[j]]
                self.placer.place_tree(tree, rots[j])
            
            reward = tree_packing_reward(
                self.placer.placed_trees,
                self.placer.bbox
            )
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Critic prediction
        values = self.critic(features_tensor).squeeze()
        
        # Losses
        advantage = rewards - values.detach()
        
        actor_loss = -(advantage * log_probs.sum(dim=1)).mean()
        critic_loss = F.mse_loss(values, rewards)
        
        # Update
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        
        (actor_loss + critic_loss).backward()
        
        self.actor_optim.step()
        self.critic_optim.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_reward': rewards.mean().item()
        }
    
    def train(self, n_epochs=100, batch_size=50, log_interval=10):
        """Full training loop"""
        for epoch in range(n_epochs):
            batch = self.generate_batch(batch_size)
            info = self.train_step(batch)
            
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}: "
                      f"Actor Loss = {info['actor_loss']:.4f}, "
                      f"Critic Loss = {info['critic_loss']:.4f}, "
                      f"Reward = {info['mean_reward']:.4f}")
```

---

# Bölüm 9: Özet ve Sonuç

## Paper'ın Temel Katkıları:

1. **Sequence-to-Sequence Formulation:** Packing problemini sıralama problemi olarak formüle etme
2. **Modified Pointer Network:** 1D Conv encoder + GRU decoder + Attention
3. **Height Map Placement:** Deterministic, top-down uyumlu placement
4. **Actor-Critic Training:** On-policy, compactness+pyramid based reward

## Tree Packing için Adaptasyon Özeti:

1. **Feature Engineering:** 2D/3D boyutlar yerine zengin tree features
2. **Rotation Handling:** Ekstra rotation head eklenmeli
3. **Placement Strategy:** Height map yerine NFP-based veya grid search
4. **Reward:** Compactness+Pyramid yerine Efficiency+Bbox penalty

## Önerilen Yaklaşım:

1. Paper'daki Pointer Network mimarisini kullan
2. Tree features için zengin representation
3. Rotation'ı discrete olarak handle et (paper tarzı)
4. Placement için NFP + Bottom-Left kombinasyonu
5. Reward: Efficiency-based + Bbox minimization

Bu yaklaşım, paper'ın güçlü yanlarını (sequence learning, attention) alırken Tree Packing'in özel gereksinimlerini (rotation, irregular shapes, bbox minimization) karşılıyor.
