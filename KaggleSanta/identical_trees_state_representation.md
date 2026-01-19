# 🌲 Identical Trees için State Representation

## Tüm Ağaçlar Aynı Olduğunda Optimal Yaklaşım

---

# Bölüm 1: Problem Analizi

## 1.1 Durum Değerlendirmesi

Tüm ağaçlar aynı (identical) olduğunda:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IDENTICAL TREES                                  │
│                                                                     │
│   Ağaç 1 = Ağaç 2 = Ağaç 3 = ... = Ağaç n                          │
│                                                                     │
│   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐                           │
│   │ △ │ = │ △ │ = │ △ │ = │ △ │ = │ △ │                           │
│   └───┘   └───┘   └───┘   └───┘   └───┘                           │
│                                                                     │
│   Aynı: şekil, boyut, alan                                         │
│   Farklı olabilecek: rotasyon, pozisyon                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 1.2 Ne Değişir?

### Paper'daki Yaklaşımın Gereksiz Kısımları:

| Özellik | Paper'da | Bizim Durumumuzda |
|---------|----------|-------------------|
| Object features (l, w, h) | Her nesne farklı | Hepsi aynı → **GEREKSIZ** |
| Packing ORDER | Önemli (farklı boyutlar) | **ÖNEMSİZ** (hepsi aynı) |
| Pointer Network | Sıralama öğrenir | **GEREKSIZ** |
| Placement | Height map based | **SADECE BU ÖNEMLİ** |

### Kritik Fark:
```
Paper: "HANGİ nesneyi NEREYE koyayım?" → 2 karar
Bizim: "NEREYE ve HANGİ AÇIYLA koyayım?" → Sadece placement kararı
```

## 1.3 Basitleştirilmiş Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│              BASİTLEŞTİRİLMİŞ PROBLEM                               │
│                                                                     │
│   Input:                                                            │
│   - n: Ağaç sayısı (integer)                                       │
│   - tree_shape: Tek bir ağacın geometrisi (Polygon)                │
│                                                                     │
│   Her adımda karar:                                                 │
│   - Position: (x, y) ∈ ℝ²                                          │
│   - Rotation: θ ∈ [0, 2π) veya discrete {0°, 45°, 90°, ...}        │
│                                                                     │
│   Hedef: n ağacı minimum bounding box'a yerleştir                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

# Bölüm 2: State Representation (Identical Trees için)

## 2.1 State'in Bileşenleri

Ağaç özellikleri yerine **PLACEMENT STATE** önemli:

```python
class IdenticalTreesState:
    """
    Identical trees için state representation
    
    Ağaç özellikleri sabit olduğundan, state sadece:
    1. Mevcut yerleşim durumu (spatial configuration)
    2. Kalan ağaç sayısı
    3. Global metrikler (bbox, efficiency)
    """
    
    def __init__(self, tree_polygon, n_trees, grid_size=64):
        self.tree_polygon = tree_polygon  # Referans ağaç (hepsi aynı)
        self.n_trees = n_trees
        self.grid_size = grid_size
        
        # Sabit ağaç özellikleri (bir kez hesapla)
        self.tree_area = tree_polygon.area
        self.tree_bounds = tree_polygon.bounds
        self.tree_width = self.tree_bounds[2] - self.tree_bounds[0]
        self.tree_height = self.tree_bounds[3] - self.tree_bounds[1]
    
    def get_state(self, placed_polygons, current_bbox):
        """
        Mevcut state'i döndür
        
        Returns:
            Dict containing:
            - occupancy_grid: 2D grid (yerleşim durumu)
            - placement_mask: Geçerli pozisyonlar
            - global_features: Skaler özellikler
        """
        state = {
            # 1. Spatial representation
            'occupancy_grid': self._create_occupancy_grid(placed_polygons, current_bbox),
            
            # 2. Valid placement mask
            'placement_mask': self._create_placement_mask(placed_polygons, current_bbox),
            
            # 3. Global features
            'global_features': self._compute_global_features(placed_polygons, current_bbox),
            
            # 4. Remaining count (tek bir sayı)
            'n_remaining': self.n_trees - len(placed_polygons)
        }
        
        return state
```

## 2.2 Occupancy Grid (Ana State Representation)

```python
def _create_occupancy_grid(self, placed_polygons, current_bbox):
    """
    2D occupancy grid oluştur
    
    Grid-based representation:
    - Her hücre: 0 (boş) veya 1 (dolu)
    - CNN ile encode edilebilir
    
    ┌────────────────────────────┐
    │ 0  0  0  0  0  0  0  0  0 │
    │ 0  0  1  1  0  0  1  1  0 │
    │ 0  1  1  1  1  0  1  1  0 │
    │ 0  1  1  1  1  1  1  1  0 │
    │ 1  1  1  1  1  1  1  1  1 │
    └────────────────────────────┘
    """
    grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
    
    if not placed_polygons:
        return grid
    
    # Bbox boyutlarını al
    minx, miny, maxx, maxy = current_bbox
    width = maxx - minx
    height = maxy - miny
    
    # Biraz padding ekle (gelecek yerleşimler için)
    padding = max(self.tree_width, self.tree_height)
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    width = maxx - minx
    height = maxy - miny
    
    # Grid çözünürlüğü
    cell_width = width / self.grid_size
    cell_height = height / self.grid_size
    
    # Her polygon için grid'i doldur
    for polygon in placed_polygons:
        # Polygon'un bounding box'ı
        px_min, py_min, px_max, py_max = polygon.bounds
        
        # Grid koordinatlarına çevir
        i_min = max(0, int((px_min - minx) / cell_width))
        i_max = min(self.grid_size - 1, int((px_max - minx) / cell_width))
        j_min = max(0, int((py_min - miny) / cell_height))
        j_max = min(self.grid_size - 1, int((py_max - miny) / cell_height))
        
        # Detaylı doluluk kontrolü
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                # Hücrenin merkezi
                cell_x = minx + (i + 0.5) * cell_width
                cell_y = miny + (j + 0.5) * cell_height
                
                if polygon.contains(Point(cell_x, cell_y)):
                    grid[j, i] = 1.0  # Note: j=row, i=col
    
    return grid
```

## 2.3 Multi-Channel Grid (Daha Zengin Representation)

```python
def _create_multi_channel_grid(self, placed_polygons, current_bbox):
    """
    Çok kanallı grid - daha zengin bilgi
    
    Channels:
    0. Occupancy (0/1)
    1. Distance to nearest obstacle
    2. Distance to bbox edge
    3. Height map (paper'dan)
    4. Potential placement quality heatmap
    """
    n_channels = 5
    grid = np.zeros((n_channels, self.grid_size, self.grid_size), dtype=np.float32)
    
    # Bbox hesapla
    if placed_polygons:
        all_bounds = [p.bounds for p in placed_polygons]
        minx = min(b[0] for b in all_bounds) - self.tree_width
        miny = min(b[1] for b in all_bounds) - self.tree_height
        maxx = max(b[2] for b in all_bounds) + self.tree_width
        maxy = max(b[3] for b in all_bounds) + self.tree_height
    else:
        # İlk ağaç için başlangıç alanı
        minx, miny = 0, 0
        maxx = self.tree_width * 3
        maxy = self.tree_height * 3
    
    width = maxx - minx
    height = maxy - miny
    cell_w = width / self.grid_size
    cell_h = height / self.grid_size
    
    # Channel 0: Occupancy
    for polygon in placed_polygons:
        self._fill_polygon_in_grid(grid[0], polygon, minx, miny, cell_w, cell_h)
    
    # Channel 1: Distance to nearest obstacle
    if placed_polygons:
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[0, j, i] == 0:  # Boş hücre
                    cell_center = Point(minx + (i + 0.5) * cell_w, 
                                       miny + (j + 0.5) * cell_h)
                    min_dist = min(polygon.distance(cell_center) 
                                  for polygon in placed_polygons)
                    grid[1, j, i] = min_dist / max(width, height)  # Normalize
    
    # Channel 2: Distance to current bbox edge
    if current_bbox:
        bx_min, by_min, bx_max, by_max = current_bbox
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_x = minx + (i + 0.5) * cell_w
                cell_y = miny + (j + 0.5) * cell_h
                
                # Bbox'a olan mesafe (içindeyse negatif, dışındaysa pozitif)
                dist_to_left = cell_x - bx_min
                dist_to_right = bx_max - cell_x
                dist_to_bottom = cell_y - by_min
                dist_to_top = by_max - cell_y
                
                min_dist = min(dist_to_left, dist_to_right, 
                              dist_to_bottom, dist_to_top)
                grid[2, j, i] = min_dist / max(width, height)
    
    # Channel 3: Height map (1D'den 2D'ye genişlet)
    height_map_1d = self._compute_height_map_1d(placed_polygons, minx, maxx)
    for i in range(self.grid_size):
        idx = int(i * len(height_map_1d) / self.grid_size)
        idx = min(idx, len(height_map_1d) - 1)
        grid[3, :, i] = height_map_1d[idx] / maxy if maxy > 0 else 0
    
    # Channel 4: Placement quality heatmap (precomputed veya learned)
    # Her pozisyona bir ağaç koyulursa bbox ne kadar artar?
    grid[4] = self._compute_placement_quality_heatmap(
        placed_polygons, minx, miny, cell_w, cell_h
    )
    
    return grid, (minx, miny, maxx, maxy)  # Grid + koordinat bilgisi

def _fill_polygon_in_grid(self, grid_channel, polygon, minx, miny, cell_w, cell_h):
    """Polygon'u grid'e çiz"""
    px_min, py_min, px_max, py_max = polygon.bounds
    
    i_min = max(0, int((px_min - minx) / cell_w))
    i_max = min(self.grid_size - 1, int((px_max - minx) / cell_w))
    j_min = max(0, int((py_min - miny) / cell_h))
    j_max = min(self.grid_size - 1, int((py_max - miny) / cell_h))
    
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            cell_center = Point(minx + (i + 0.5) * cell_w, 
                               miny + (j + 0.5) * cell_h)
            if polygon.contains(cell_center):
                grid_channel[j, i] = 1.0
```

## 2.4 Global Features

```python
def _compute_global_features(self, placed_polygons, current_bbox):
    """
    Skaler global özellikler
    
    Bunlar ağaçtan bağımsız, duruma bağlı
    """
    n_placed = len(placed_polygons)
    n_remaining = self.n_trees - n_placed
    
    if n_placed == 0:
        return np.array([
            0.0,  # current_efficiency
            0.0,  # bbox_area (normalized)
            1.0,  # remaining_ratio
            0.0,  # placed_ratio
            0.0,  # avg_distance_between_trees
            0.0,  # compactness
        ], dtype=np.float32)
    
    # Bbox metrics
    minx, miny, maxx, maxy = current_bbox
    bbox_area = (maxx - minx) * (maxy - miny)
    total_tree_area = n_placed * self.tree_area
    
    # Efficiency
    efficiency = total_tree_area / bbox_area if bbox_area > 0 else 0
    
    # Theoretical minimum bbox (tüm ağaçlar için)
    theoretical_min = self.n_trees * self.tree_area
    normalized_bbox = bbox_area / theoretical_min if theoretical_min > 0 else 1
    
    # Compactness (paper'dan)
    max_height = maxy - miny
    compactness_region = (maxx - minx) * max_height
    compactness = total_tree_area / compactness_region if compactness_region > 0 else 0
    
    # Average pairwise distance (clustering metric)
    if n_placed >= 2:
        centroids = [p.centroid for p in placed_polygons]
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distances.append(centroids[i].distance(centroids[j]))
        avg_distance = np.mean(distances) / np.sqrt(bbox_area) if distances else 0
    else:
        avg_distance = 0
    
    return np.array([
        efficiency,
        normalized_bbox,
        n_remaining / self.n_trees,  # remaining_ratio
        n_placed / self.n_trees,      # placed_ratio
        avg_distance,
        compactness,
    ], dtype=np.float32)
```

---

# Bölüm 3: Action Space

## 3.1 Sadece Placement Kararı

Identical trees için action space:

```python
class IdenticalTreesActionSpace:
    """
    Action = (position, rotation)
    
    Hangi ağaç? → Önemsiz, hepsi aynı
    """
    
    def __init__(self, n_positions=64, n_rotations=8):
        """
        Discrete action space:
        - n_positions × n_positions grid pozisyonları
        - n_rotations rotation seçenekleri
        
        Total actions = n_positions² × n_rotations
        """
        self.n_positions = n_positions  # Her eksen için
        self.n_rotations = n_rotations
        
        self.rotation_angles = np.linspace(0, 2*np.pi, n_rotations, endpoint=False)
        
        # Veya continuous:
        # action = (x, y, θ) ∈ [0,1] × [0,1] × [0, 2π]
    
    def discrete_to_continuous(self, action_idx, grid_bounds):
        """
        Discrete action index'i continuous (x, y, θ)'ya çevir
        """
        minx, miny, maxx, maxy = grid_bounds
        
        # Action encoding: action_idx = rot_idx * n_pos² + y_idx * n_pos + x_idx
        n_pos_sq = self.n_positions ** 2
        
        rot_idx = action_idx // n_pos_sq
        remainder = action_idx % n_pos_sq
        y_idx = remainder // self.n_positions
        x_idx = remainder % self.n_positions
        
        # Continuous coordinates
        x = minx + (x_idx + 0.5) * (maxx - minx) / self.n_positions
        y = miny + (y_idx + 0.5) * (maxy - miny) / self.n_positions
        theta = self.rotation_angles[rot_idx]
        
        return x, y, theta
    
    def get_valid_actions(self, state, placed_polygons, tree_polygon):
        """
        Geçerli action'ları bul (collision-free placements)
        """
        valid = []
        grid_bounds = state['grid_bounds']
        
        for action_idx in range(self.n_positions ** 2 * self.n_rotations):
            x, y, theta = self.discrete_to_continuous(action_idx, grid_bounds)
            
            # Ağacı bu pozisyon ve rotasyonla yerleştir
            placed = self._place_tree(tree_polygon, x, y, theta)
            
            # Collision check
            if self._is_valid_placement(placed, placed_polygons):
                valid.append(action_idx)
        
        return valid
```

## 3.2 Hierarchical Action Space (Daha Verimli)

```python
class HierarchicalActionSpace:
    """
    İki aşamalı action selection:
    
    1. Coarse: Hangi bölgeye? (grid cell selection)
    2. Fine: Tam pozisyon + rotasyon
    
    Bu, action space'i dramatik olarak küçültür
    """
    
    def __init__(self, coarse_grid=8, fine_grid=8, n_rotations=8):
        self.coarse_grid = coarse_grid  # 8×8 = 64 bölge
        self.fine_grid = fine_grid      # Her bölgede 8×8 = 64 pozisyon
        self.n_rotations = n_rotations
        
        # Coarse action: 64
        # Fine action: 64 × 8 = 512
        # Total sequence: 2 karar vs 64×64×8 = 32768 tek karar
    
    def select_action(self, policy_network, state):
        """
        İki aşamalı action selection
        """
        # Stage 1: Coarse region selection
        coarse_logits = policy_network.coarse_head(state)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        coarse_action = torch.distributions.Categorical(coarse_probs).sample()
        
        # Stage 2: Fine position + rotation within region
        region_state = self._get_region_state(state, coarse_action)
        fine_logits = policy_network.fine_head(region_state)
        fine_probs = F.softmax(fine_logits, dim=-1)
        fine_action = torch.distributions.Categorical(fine_probs).sample()
        
        return coarse_action, fine_action
```

---

# Bölüm 4: Network Architecture (Identical Trees için)

## 4.1 Simplified Architecture

```python
class IdenticalTreesPolicy(nn.Module):
    """
    Identical trees için basitleştirilmiş policy network
    
    Pointer Network GEREKSIZ çünkü sıralama önemsiz
    Sadece CNN + spatial reasoning
    """
    
    def __init__(
        self,
        grid_size=64,
        n_channels=5,
        n_global_features=6,
        hidden_dim=256,
        n_rotations=8
    ):
        super().__init__()
        
        self.n_rotations = n_rotations
        
        # ════════════════════════════════════════════════════════════
        # ENCODER: Spatial features from occupancy grid
        # ════════════════════════════════════════════════════════════
        
        self.cnn_encoder = nn.Sequential(
            # Input: (batch, n_channels, 64, 64)
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 32×32
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 16×16
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 8×8
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 4×4
            
            nn.Flatten(),  # → 256 * 4 * 4 = 4096
        )
        
        cnn_output_dim = 256 * 4 * 4
        
        # Global features encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(n_global_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # ════════════════════════════════════════════════════════════
        # ACTION HEAD: Position + Rotation
        # ════════════════════════════════════════════════════════════
        
        # Position heatmap (deconvolution ile)
        self.position_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 4 * 4),
            nn.ReLU(),
            Reshape(-1, 256, 4, 4),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → 8×8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # → 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # → 32×32
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_rotations, kernel_size=4, stride=2, padding=1),  # → 64×64
            # Output: (batch, n_rotations, 64, 64)
            # Her (x, y) için her rotation'ın score'u
        )
        
        # ════════════════════════════════════════════════════════════
        # VALUE HEAD: Critic
        # ════════════════════════════════════════════════════════════
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, grid, global_features, valid_mask=None):
        """
        Forward pass
        
        Args:
            grid: (batch, n_channels, H, W) - occupancy + other channels
            global_features: (batch, n_global) - efficiency, bbox, etc.
            valid_mask: (batch, n_rotations, H, W) - valid placement mask
        
        Returns:
            action_logits: (batch, n_rotations * H * W)
            value: (batch, 1)
        """
        batch_size = grid.shape[0]
        
        # Encode
        cnn_features = self.cnn_encoder(grid)
        global_encoded = self.global_encoder(global_features)
        
        # Fuse
        combined = torch.cat([cnn_features, global_encoded], dim=-1)
        features = self.fusion(combined)
        
        # Position + Rotation logits
        position_rotation_map = self.position_decoder(features)
        # Shape: (batch, n_rotations, 64, 64)
        
        # Flatten for action selection
        action_logits = position_rotation_map.view(batch_size, -1)
        # Shape: (batch, n_rotations * 64 * 64)
        
        # Apply validity mask
        if valid_mask is not None:
            valid_mask_flat = valid_mask.view(batch_size, -1)
            action_logits = action_logits.masked_fill(~valid_mask_flat, float('-inf'))
        
        # Value
        value = self.value_head(features)
        
        return action_logits, value
    
    def get_action(self, grid, global_features, valid_mask=None, deterministic=False):
        """
        Action selection
        """
        action_logits, value = self.forward(grid, global_features, valid_mask)
        
        # Softmax
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(action_probs).sample()
        
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)
        
        return action, log_prob.squeeze(1), value


class Reshape(nn.Module):
    """Helper module for reshaping in Sequential"""
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)
```

## 4.2 Daha Basit Alternatif: Sadece Position (Rotation Ayrı)

```python
class SimplePositionPolicy(nn.Module):
    """
    Daha basit versiyon: Pozisyon ve rotasyon ayrı head'ler
    """
    
    def __init__(self, grid_size=64, n_channels=5, hidden_dim=256, n_rotations=8):
        super().__init__()
        
        self.grid_size = grid_size
        self.n_rotations = n_rotations
        
        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Position head: 64×64 grid üzerinde probability
        self.position_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, grid_size * grid_size)
        )
        
        # Rotation head: pozisyon seçildikten sonra
        self.rotation_head = nn.Sequential(
            nn.Linear(256 + 2, 128),  # +2 for selected position
            nn.ReLU(),
            nn.Linear(128, n_rotations)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, grid, selected_position=None):
        """
        Two-stage forward:
        1. Position selection
        2. Rotation selection (given position)
        """
        features = self.encoder(grid)
        
        # Position logits
        position_logits = self.position_head(features)
        
        # Rotation logits (if position given)
        if selected_position is not None:
            # Normalize position to [0, 1]
            pos_normalized = selected_position / self.grid_size
            rotation_input = torch.cat([features, pos_normalized], dim=-1)
            rotation_logits = self.rotation_head(rotation_input)
        else:
            rotation_logits = None
        
        # Value
        value = self.value_head(features)
        
        return position_logits, rotation_logits, value
```

---

# Bölüm 5: Training for Identical Trees

## 5.1 Environment

```python
class IdenticalTreesEnv:
    """
    Identical trees packing environment
    """
    
    def __init__(self, tree_polygon, n_trees, grid_size=64, n_rotations=8):
        self.tree_polygon = tree_polygon
        self.n_trees = n_trees
        self.grid_size = grid_size
        self.n_rotations = n_rotations
        
        self.rotation_angles = np.linspace(0, 2*np.pi, n_rotations, endpoint=False)
        
        self.state_builder = IdenticalTreesState(tree_polygon, n_trees, grid_size)
        
        self.reset()
    
    def reset(self):
        """Environment'ı sıfırla"""
        self.placed_polygons = []
        self.current_bbox = None
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Mevcut state'i al"""
        grid, grid_bounds = self.state_builder._create_multi_channel_grid(
            self.placed_polygons, self.current_bbox
        )
        
        global_features = self.state_builder._compute_global_features(
            self.placed_polygons, self.current_bbox
        )
        
        valid_mask = self._compute_valid_mask(grid_bounds)
        
        return {
            'grid': grid,
            'global_features': global_features,
            'valid_mask': valid_mask,
            'grid_bounds': grid_bounds
        }
    
    def step(self, action):
        """
        Bir adım at
        
        action: integer index encoding (x, y, rotation)
        """
        grid_bounds = self._get_state()['grid_bounds']
        
        # Decode action
        x, y, theta = self._decode_action(action, grid_bounds)
        
        # Place tree
        placed = self._place_tree(x, y, theta)
        
        # Collision check
        if not self._is_valid_placement(placed):
            # Invalid placement
            return self._get_state(), -1.0, False, {'valid': False}
        
        # Add to placed
        self.placed_polygons.append(placed)
        self._update_bbox()
        self.step_count += 1
        
        # Check if done
        done = self.step_count >= self.n_trees
        
        # Compute reward
        reward = self._compute_reward(done)
        
        return self._get_state(), reward, done, {'valid': True}
    
    def _place_tree(self, x, y, theta):
        """Ağacı pozisyon ve rotasyonla yerleştir"""
        # Rotate around centroid
        rotated = shapely.affinity.rotate(
            self.tree_polygon, theta, 
            origin='centroid', use_radians=True
        )
        
        # Translate
        cx, cy = rotated.centroid.x, rotated.centroid.y
        placed = shapely.affinity.translate(rotated, xoff=x-cx, yoff=y-cy)
        
        return placed
    
    def _is_valid_placement(self, polygon):
        """Yerleşim geçerli mi?"""
        for placed in self.placed_polygons:
            if polygon.intersects(placed):
                return False
        return True
    
    def _update_bbox(self):
        """Bounding box güncelle"""
        if not self.placed_polygons:
            self.current_bbox = None
            return
        
        all_bounds = [p.bounds for p in self.placed_polygons]
        self.current_bbox = (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds)
        )
    
    def _compute_reward(self, done):
        """
        Reward hesapla
        
        Her adımda:
        - Efficiency artışı
        - Bbox artış penaltisi
        
        Episode sonunda:
        - Final efficiency bonus
        """
        if not self.placed_polygons:
            return 0
        
        total_area = len(self.placed_polygons) * self.state_builder.tree_area
        bbox_area = (self.current_bbox[2] - self.current_bbox[0]) * \
                   (self.current_bbox[3] - self.current_bbox[1])
        
        efficiency = total_area / bbox_area
        
        if done:
            # Final reward: efficiency-based
            # Theoretical best: efficiency = 1 (impossible for triangles)
            # Good: efficiency > 0.8
            return efficiency * 10
        else:
            # Incremental reward
            return efficiency * 0.1
```

## 5.2 Simplified Training Loop

```python
def train_identical_trees(env, policy, n_episodes=10000):
    """
    Basit training loop
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    for episode in range(n_episodes):
        state = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        
        done = False
        while not done:
            # State tensors
            grid = torch.tensor(state['grid']).unsqueeze(0)
            global_feat = torch.tensor(state['global_features']).unsqueeze(0)
            valid_mask = torch.tensor(state['valid_mask']).unsqueeze(0)
            
            # Get action
            action, log_prob, value = policy.get_action(
                grid, global_feat, valid_mask
            )
            
            # Step
            next_state, reward, done, info = env.step(action.item())
            
            # Store
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            state = next_state
        
        # Compute returns and advantages
        returns = compute_returns(rewards, gamma=0.99)
        advantages = returns - torch.cat(values).squeeze()
        
        # Policy loss
        policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(torch.cat(values).squeeze(), returns)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards):.2f}, "
                  f"Efficiency: {env.state_builder._compute_global_features(env.placed_polygons, env.current_bbox)[0]:.2f}")
```

---

# Bölüm 6: Özet

## Identical Trees için Key Takeaways:

### 1. Gereksiz Olanlar:
- ❌ Object features (hepsi aynı)
- ❌ Packing order (sıra önemsiz)
- ❌ Pointer Network (seçim yok)
- ❌ Tree encoder

### 2. Gerekli Olanlar:
- ✅ Occupancy grid (mevcut yerleşim)
- ✅ Multi-channel spatial features
- ✅ Position + Rotation action space
- ✅ Global features (efficiency, bbox, remaining count)

### 3. Simplified State:
```python
state = {
    'grid': (n_channels, H, W),      # Spatial info
    'global': (n_features,),          # Scalars
    'valid_mask': (n_rot, H, W),     # Valid placements
    'n_remaining': int               # Kalan ağaç sayısı
}
```

### 4. Simplified Action:
```python
action = (x, y, θ)  # Position + Rotation
# Discrete: action_idx ∈ [0, H × W × n_rotations)
```

### 5. Network:
```
CNN Encoder → Features → Position Head → (x, y) selection
                      → Rotation Head → θ selection
                      → Value Head → V(s)
```

Bu yaklaşım paper'dan çok daha basit ve identical trees için optimal!
