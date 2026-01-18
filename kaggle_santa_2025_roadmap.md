# 🎄 Kaggle Santa 2025 - Christmas Tree Packing Challenge

## Kapsamlı Yol Haritası ve Strateji Dokümanı

---

# Bölüm 1: Problem Tanımı ve Genel Bakış

## 1.1 Yarışma Özeti

Bu yarışmada Noel ağacı oyuncaklarını mümkün olan en küçük kare kutuya yerleştirerek Santa'nın onları dünya genelinde verimli bir şekilde postalamasına yardımcı olman gerekiyor. **1 ila 200 ağacı** sığdırabilecek en küçük kare kutunun boyutlarını bulman isteniyor.

### Ödüller
- **Ana Ödül:** Yarışma sonu sıralamasına göre
- **Rudolph Ödülü:** 10.000$ - 17 Kasım 2025 ile 30 Ocak 2026 arasında liderlik tablosunda en uzun süre 1. sırada kalan takıma veriliyor

### Değerlendirme Metriği
Çözümler, her puzzle konfigürasyonu için kare bounding box'ın **normalize edilmiş alanına** göre değerlendiriliyor.

## 1.2 Problem Yapısı

Bu bir **2D Irregular Polygon Packing** problemidir. Ağaçlar muhtemelen üçgen (isosceles triangle) şeklinde ve bunları kare bir alana sığdırman gerekiyor.

**Temel kısıtlar:**
- Ağaçlar üst üste binemez (no overlap)
- Tüm ağaçlar kutunun içinde kalmalı
- Rotasyon muhtemelen serbest (0° - 360°)
- Amaç: Kare kutunun alanını minimize etmek

---

# Bölüm 2: Temel Yol Haritası

## Faz 1: Problemi Anlama (1-2 gün)

### 2.1 Veri Setini İncele
- Kaggle'dan veri setini indir
- Her puzzle için kaç ağaç olduğunu, ağaç şekillerini analiz et
- `sample_submission.csv` formatını anla

### 2.2 Gerekli Kütüphaneler
```python
# Temel araçlar
pip install numpy scipy shapely matplotlib
pip install networkx  # Graf işlemleri için
pip install numba     # Hız optimizasyonu
```

## Faz 2: Baseline Çözüm (3-5 gün)

### 2.3 Geometrik Primitifler
```python
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate

class Tree:
    def __init__(self, vertices):
        self.polygon = Polygon(vertices)
        self.x, self.y = 0, 0
        self.angle = 0
    
    def transform(self, x, y, angle):
        p = rotate(self.polygon, angle, origin='centroid')
        p = translate(p, x, y)
        return p
    
    def overlaps(self, other_tree):
        return self.transform(...).intersects(other_tree.transform(...))
```

### 2.4 Basit Greedy Yerleştirme (Bottom-Left Fill)

Klasik bottom-left algoritması: Parçaları büyükten küçüğe sıralayıp, her parçayı sol-alt köşeye mümkün olduğunca yakın yerleştir.

```python
def bottom_left_fill(trees, box_size):
    placed = []
    for tree in sorted(trees, key=lambda t: -t.area):
        best_pos = None
        for y in range(0, box_size):
            for x in range(0, box_size):
                if can_place(tree, x, y, placed, box_size):
                    if best_pos is None or (y, x) < best_pos:
                        best_pos = (y, x)
        if best_pos:
            tree.x, tree.y = best_pos[1], best_pos[0]
            placed.append(tree)
    return placed
```

## Faz 3: No-Fit Polygon (NFP) İmplementasyonu (1 hafta)

### 2.5 NFP Nedir?

No-fit polygon (NFP), iki parçanın çakışmadan yerleştirilebileceği pozisyonları gösteren güçlü bir geometrik araçtır ve 2D düzensiz paketleme problemlerinde ana akım yöntemdir.

```python
from shapely.ops import unary_union

def compute_nfp(fixed_poly, moving_poly):
    """
    Fixed polygon etrafında moving polygon'un 
    merkez noktasının izleyebileceği yolu hesapla
    """
    # Minkowski sum yaklaşımı
    # veya edge-sliding algoritması
    pass

def inner_fit_polygon(container, piece):
    """
    Container içinde piece'in yerleştirilebileceği 
    tüm geçerli pozisyonları bul
    """
    pass
```

### 2.6 Hızlı NFP Kütüphaneleri
- **pyclipper**: Polygon clipping işlemleri
- **svg-nest**: JavaScript tabanlı ama Python binding'leri var
- Kendi C++ implementasyonunu yaz + pybind11

## Faz 4: Optimizasyon Algoritmaları (2-3 hafta)

### 2.7 Simulated Annealing

```python
import random
import math

def simulated_annealing(trees, initial_box_size, max_iter=100000):
    T = 1.0
    T_min = 0.0001
    alpha = 0.995
    
    current = initial_solution(trees, initial_box_size)
    current_score = evaluate(current)
    best = current.copy()
    best_score = current_score
    
    while T > T_min:
        for _ in range(100):
            # Komşu çözüm üret
            neighbor = perturb(current)  # Küçük hareket/rotasyon
            neighbor_score = evaluate(neighbor)
            
            delta = neighbor_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = neighbor
                current_score = neighbor_score
                
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
        
        T *= alpha
    
    return best, best_score
```

### 2.8 Genetic Algorithm

```python
def genetic_algorithm(trees, population_size=100, generations=500):
    # Kromozom: (tree_order, rotations)
    population = [random_chromosome(trees) for _ in range(population_size)]
    
    for gen in range(generations):
        # Fitness hesapla
        fitness = [evaluate_chromosome(c) for c in population]
        
        # Selection
        parents = tournament_selection(population, fitness)
        
        # Crossover & Mutation
        offspring = []
        for i in range(0, len(parents), 2):
            c1, c2 = crossover(parents[i], parents[i+1])
            offspring.extend([mutate(c1), mutate(c2)])
        
        population = offspring
    
    return best_individual(population)
```

### 2.9 Local Search Operatörleri
```python
def perturb_moves():
    """Farklı pertürbasyon stratejileri"""
    moves = [
        'translate_single',      # Tek ağacı hareket ettir
        'rotate_single',         # Tek ağacı döndür
        'swap_positions',        # İki ağacın yerini değiştir
        'compact_all',           # Tüm ağaçları merkeze sıkıştır
        'jiggle_all',            # Rastgele küçük hareketler
        'rotate_and_slide'       # Döndür ve kaydır
    ]
    return random.choice(moves)
```

## Faz 5: Gelişmiş Teknikler (2 hafta)

### 2.10 Binary Search + Feasibility Check
```python
def find_minimum_box_size(trees):
    # Minimum ve maksimum box size tahminleri
    total_area = sum(t.area for t in trees)
    lo = math.sqrt(total_area)  # Alt sınır
    hi = lo * 2                  # Üst sınır
    
    best_feasible = None
    
    while hi - lo > 0.001:
        mid = (lo + hi) / 2
        if can_pack_in_size(trees, mid):
            best_feasible = mid
            hi = mid
        else:
            lo = mid
    
    return best_feasible
```

### 2.11 Compaction Algoritması
```python
def compact_solution(trees, box_size):
    """Yerleşmiş ağaçları sıkıştır"""
    improved = True
    while improved:
        improved = False
        for tree in trees:
            # Her ağacı merkeze/köşeye doğru it
            for direction in ['left', 'down', 'center']:
                if try_move(tree, direction):
                    improved = True
    return trees
```

## Faz 6: Hibrit Yaklaşımlar (1-2 hafta)

### 2.12 Multi-Start Stratejisi
```python
def multi_start_optimization(trees, n_starts=20):
    best_solution = None
    best_score = float('inf')
    
    for i in range(n_starts):
        # Farklı başlangıç stratejileri
        if i % 4 == 0:
            initial = random_placement(trees)
        elif i % 4 == 1:
            initial = grid_placement(trees)
        elif i % 4 == 2:
            initial = sorted_placement(trees, key='area')
        else:
            initial = sorted_placement(trees, key='height')
        
        solution = simulated_annealing(initial)
        if solution.score < best_score:
            best_solution = solution
            best_score = solution.score
    
    return best_solution
```

## Zaman Çizelgesi Özeti

| Hafta | Aktivite | Hedef |
|-------|----------|-------|
| 1 | Problem anlama + Baseline | İlk submission |
| 2-3 | NFP + Temel optimizasyon | Top 50% |
| 4-5 | SA/GA implementasyonu | Top 25% |
| 6 | Gelişmiş teknikler | Top 10% |
| 7 | Hibrit yaklaşımlar | Top 5% |
| 8 | Performans + Ensemble | Final submission |

---

# Bölüm 3: Reinforcement Learning Yaklaşımı

## 3.1 RL Bu Problem İçin Uygun mu?

**Evet, çok uygun!** Çünkü:
- Sıralı karar verme problemi (her ağacı sırayla yerleştiriyorsun)
- Eylem uzayı tanımlı (pozisyon + rotasyon)
- Reward fonksiyonu net (kutu boyutunu minimize et)
- Benzer problemlerde (Tetris, bin packing) RL başarılı olmuş

## 3.2 Problemin MDP Formülasyonu

```python
# State (Durum)
# - Şu ana kadar yerleştirilmiş ağaçların konumları
# - Kalan ağaçların listesi
# - Mevcut bounding box boyutu
# - Boş alan haritası (occupancy grid veya polygon)

# Action (Eylem)
# - Hangi ağacı yerleştireceğin (kalan ağaçlardan)
# - X, Y koordinatları (continuous veya discretized)
# - Rotasyon açısı θ (0° - 360°)

# Reward (Ödül)
# - Geçerli yerleştirme: +1
# - Overlap: -100 (veya invalid action)
# - Episode sonu: -final_box_area (minimize etmek için)
# - Kompakt yerleştirme bonusu: +α * (1 - wasted_space_ratio)
```

## 3.3 RL Mimarileri

### 3.3.1 DQN Yaklaşımı (Discretized Actions)

```python
import torch
import torch.nn as nn
import numpy as np

class PackingDQN(nn.Module):
    def __init__(self, grid_size=64, n_rotations=8, max_trees=200):
        super().__init__()
        
        # State encoder - CNN for spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 3 channels: occupancy, current_tree, remaining_area
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Tree features encoder
        self.tree_encoder = nn.Sequential(
            nn.Linear(max_trees * 4, 256),  # Her ağaç için: area, width, height, placed_flag
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Action head
        conv_out_size = 128 * (grid_size // 4) ** 2
        self.action_head = nn.Sequential(
            nn.Linear(conv_out_size + 128, 512),
            nn.ReLU(),
            nn.Linear(512, grid_size * grid_size * n_rotations)  # Q-value for each action
        )
    
    def forward(self, grid_state, tree_features):
        spatial = self.conv(grid_state)
        tree_emb = self.tree_encoder(tree_features)
        combined = torch.cat([spatial, tree_emb], dim=1)
        q_values = self.action_head(combined)
        return q_values
```

### 3.3.2 Environment Implementation

```python
class PackingEnvironment:
    def __init__(self, trees, grid_size=64):
        self.trees = trees
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        self.placed = []
        self.remaining = list(range(len(self.trees)))
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.current_box_size = 0
        return self._get_state()
    
    def step(self, action):
        # action = (tree_idx, x, y, rotation)
        tree_idx, x, y, rot = self._decode_action(action)
        
        tree = self.trees[tree_idx]
        placed_poly = self._place_tree(tree, x, y, rot)
        
        # Check validity
        if self._has_overlap(placed_poly) or self._out_of_bounds(placed_poly):
            return self._get_state(), -100, False, {}  # Invalid action
        
        # Update state
        self.placed.append((tree_idx, x, y, rot))
        self.remaining.remove(tree_idx)
        self._update_grid(placed_poly)
        self._update_box_size()
        
        # Check if done
        done = len(self.remaining) == 0
        
        # Calculate reward
        if done:
            reward = -self.current_box_size  # Minimize final box
        else:
            # Intermediate reward: encourage compact placement
            compactness = self._calculate_compactness()
            reward = compactness * 0.1
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        # Return grid + tree features
        return {
            'grid': self.grid.copy(),
            'tree_features': self._encode_trees(),
            'remaining_mask': self._get_remaining_mask()
        }
```

### 3.3.3 PPO/A2C ile Continuous Actions

```python
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, max_trees):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Actor heads
        # Discrete: which tree to place
        self.tree_selector = nn.Linear(256, max_trees)
        
        # Continuous: position (x, y) and rotation (θ)
        self.position_mean = nn.Linear(256, 2)  # x, y
        self.position_std = nn.Linear(256, 2)
        self.rotation_mean = nn.Linear(256, 1)  # θ
        self.rotation_std = nn.Linear(256, 1)
        
        # Critic head
        self.critic = nn.Linear(256, 1)
    
    def forward(self, state, remaining_mask):
        features = self.backbone(state)
        
        # Tree selection (masked)
        tree_logits = self.tree_selector(features)
        tree_logits = tree_logits.masked_fill(~remaining_mask, float('-inf'))
        tree_dist = Categorical(logits=tree_logits)
        
        # Position distribution
        pos_mean = torch.sigmoid(self.position_mean(features))  # [0, 1] normalized
        pos_std = torch.softplus(self.position_std(features)) + 0.01
        pos_dist = Normal(pos_mean, pos_std)
        
        # Rotation distribution
        rot_mean = torch.sigmoid(self.rotation_mean(features)) * 2 * np.pi
        rot_std = torch.softplus(self.rotation_std(features)) + 0.01
        rot_dist = Normal(rot_mean, rot_std)
        
        # Value estimate
        value = self.critic(features)
        
        return tree_dist, pos_dist, rot_dist, value
    
    def get_action(self, state, remaining_mask):
        tree_dist, pos_dist, rot_dist, value = self.forward(state, remaining_mask)
        
        tree_idx = tree_dist.sample()
        position = pos_dist.sample()
        rotation = rot_dist.sample()
        
        log_prob = (tree_dist.log_prob(tree_idx) + 
                   pos_dist.log_prob(position).sum(-1) +
                   rot_dist.log_prob(rotation).sum(-1))
        
        return tree_idx, position, rotation, log_prob, value
```

### 3.3.4 PPO Update

```python
def ppo_update(policy, optimizer, trajectories, clip_epsilon=0.2):
    states, actions, old_log_probs, returns, advantages = trajectories
    
    for _ in range(10):  # PPO epochs
        tree_dist, pos_dist, rot_dist, values = policy(states)
        
        # Calculate new log probs
        new_log_probs = calculate_log_probs(tree_dist, pos_dist, rot_dist, actions)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Entropy bonus (exploration)
        entropy = tree_dist.entropy().mean() + pos_dist.entropy().mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.3.5 Graph Neural Network + RL

Ağaçlar arası ilişkileri modellemek için:

```python
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool

class GNNPackingPolicy(nn.Module):
    """
    Her ağaç bir node, potansiyel komşuluklar edge
    """
    def __init__(self, node_features=8, hidden_dim=128):
        super().__init__()
        
        # Graph layers
        self.gat1 = GATConv(node_features, hidden_dim, heads=4)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4)
        self.gat3 = GATConv(hidden_dim * 4, hidden_dim, heads=1)
        
        # Global context
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        # Per-node action scores
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Score for selecting this tree next
        )
        
        # Position predictor (given selected tree)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # x, y, θ
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message passing
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        x = self.gat3(x, edge_index)
        
        # Global context
        global_feat = global_mean_pool(x, batch)
        global_context = self.global_encoder(global_feat)
        
        # Expand global to all nodes
        global_expanded = global_context[batch]
        node_features = torch.cat([x, global_expanded], dim=1)
        
        # Selection scores
        selection_scores = self.node_scorer(node_features)
        
        return selection_scores, node_features
```

### 3.3.6 Curriculum Learning Stratejisi

```python
class CurriculumScheduler:
    """Kolay problemlerden zor problemlere geçiş"""
    
    def __init__(self):
        self.stage = 0
        self.success_threshold = 0.8
        self.stage_configs = [
            {'n_trees': 3, 'rotations': [0]},
            {'n_trees': 5, 'rotations': [0, 90]},
            {'n_trees': 10, 'rotations': [0, 45, 90, 135]},
            {'n_trees': 20, 'rotations': 'continuous'},
            {'n_trees': 50, 'rotations': 'continuous'},
            {'n_trees': 100, 'rotations': 'continuous'},
            {'n_trees': 200, 'rotations': 'continuous'},
        ]
    
    def get_current_config(self):
        return self.stage_configs[min(self.stage, len(self.stage_configs) - 1)]
    
    def update(self, success_rate):
        if success_rate > self.success_threshold:
            self.stage += 1
            print(f"Advancing to stage {self.stage}")


def train_with_curriculum(policy, n_episodes=100000):
    scheduler = CurriculumScheduler()
    
    for episode in range(n_episodes):
        config = scheduler.get_current_config()
        env = PackingEnvironment(
            n_trees=config['n_trees'],
            allowed_rotations=config['rotations']
        )
        
        # Train episode
        success = run_episode(policy, env)
        
        # Update curriculum
        if episode % 1000 == 0:
            success_rate = evaluate(policy, config)
            scheduler.update(success_rate)
```

---

# Bölüm 4: Grup Teorisi ve Permütasyon Grupları

## 4.1 Neden Grup Teorisi?

Bu problemde çok fazla **simetri** var:
1. **Rotasyonel simetri**: Ağaçlar döndürülebilir
2. **Permütasyon simetrisi**: Ağaçların yerleştirme sırası değiştirilebilir
3. **Yansıma simetrisi**: Çözüm yatay/dikey yansıtılabilir
4. **Öteleme simetrisi**: Tüm çözüm kaydırılabilir

Bu simetrileri exploit ederek arama uzayını **dramatik şekilde küçültebiliriz!**

## 4.2 Temel Kavramlar

### 4.2.1 Simetri Grubu Tanımı

```python
from itertools import permutations
import numpy as np

class SymmetryGroup:
    """
    Çözüm uzayının simetri grubu
    G = D_4 × S_n × T
    - D_4: Kare için dihedral grup (8 eleman: 4 rotasyon + 4 yansıma)
    - S_n: n ağacın permütasyon grubu (n! eleman)
    - T: Öteleme grubu (continuous)
    """
    
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.d4_elements = self._generate_d4()
    
    def _generate_d4(self):
        """Kare için dihedral grup D_4"""
        elements = []
        
        # Rotasyonlar: R_0, R_90, R_180, R_270
        for angle in [0, 90, 180, 270]:
            elements.append(('rotation', angle))
        
        # Yansımalar: horizontal, vertical, diagonal1, diagonal2
        elements.append(('reflect', 'horizontal'))
        elements.append(('reflect', 'vertical'))
        elements.append(('reflect', 'diagonal1'))
        elements.append(('reflect', 'diagonal2'))
        
        return elements
    
    def apply_d4(self, solution, element):
        """D_4 grubunun bir elemanını çözüme uygula"""
        action_type, param = element
        
        if action_type == 'rotation':
            return self._rotate_solution(solution, param)
        else:
            return self._reflect_solution(solution, param)
    
    def _rotate_solution(self, solution, angle):
        """Tüm çözümü angle derece döndür"""
        new_positions = []
        center = solution.box_size / 2
        
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        for tree in solution.trees:
            # Rotate position around center
            x, y = tree.x - center, tree.y - center
            new_x = x * cos_t - y * sin_t + center
            new_y = x * sin_t + y * cos_t + center
            
            # Also rotate the tree's orientation
            new_angle = (tree.angle + angle) % 360
            
            new_positions.append((new_x, new_y, new_angle))
        
        return new_positions
    
    def canonical_form(self, solution):
        """
        Çözümü kanonik forma getir - simetri grubu altında 
        tek bir temsilci seç
        """
        # Tüm D_4 dönüşümlerini uygula
        all_forms = []
        for element in self.d4_elements:
            transformed = self.apply_d4(solution, element)
            # Hash'lenebilir forma çevir
            canonical = self._to_canonical_string(transformed)
            all_forms.append((canonical, transformed))
        
        # Lexicographically en küçüğünü seç
        all_forms.sort(key=lambda x: x[0])
        return all_forms[0][1]
```

### 4.2.2 Orbits ve Stabilizers

```python
class OrbitCalculator:
    """
    Burnside Lemma kullanarak eşdeğer çözümleri say
    |X/G| = (1/|G|) * Σ |X^g|
    """
    
    def __init__(self, group):
        self.group = group
    
    def orbit_size(self, solution):
        """Bir çözümün orbit büyüklüğünü hesapla"""
        orbit = set()
        
        for g in self.group.elements:
            transformed = self.group.apply(solution, g)
            orbit.add(self._hash_solution(transformed))
        
        return len(orbit)
    
    def stabilizer(self, solution):
        """
        Çözümü sabit bırakan grup elemanları
        Stab(x) = {g ∈ G : g·x = x}
        """
        stab = []
        original_hash = self._hash_solution(solution)
        
        for g in self.group.elements:
            transformed = self.group.apply(solution, g)
            if self._hash_solution(transformed) == original_hash:
                stab.append(g)
        
        return stab
    
    def orbit_stabilizer_theorem(self, solution):
        """
        |Orb(x)| * |Stab(x)| = |G|
        """
        orbit = self.orbit_size(solution)
        stabilizer = len(self.stabilizer(solution))
        group_size = len(self.group.elements)
        
        assert orbit * stabilizer == group_size
        return orbit, stabilizer
```

### 4.2.3 Permütasyon Grupları ile Arama Uzayı Küçültme

```python
from sympy.combinatorics import Permutation, PermutationGroup

class PermutationSymmetry:
    """
    Özdeş ağaçlar varsa permütasyon simetrisi kullan
    """
    
    def __init__(self, trees):
        self.trees = trees
        self.equivalence_classes = self._find_equivalent_trees()
        self.symmetry_group = self._build_symmetry_group()
    
    def _find_equivalent_trees(self):
        """
        Aynı boyuttaki ağaçları grupla
        Eğer tree_i ve tree_j aynı şekildeyse, 
        herhangi bir çözümde yer değiştirebilirler
        """
        classes = {}
        for i, tree in enumerate(self.trees):
            # Ağacın "imzası" - boyut ve şekil
            signature = self._tree_signature(tree)
            if signature not in classes:
                classes[signature] = []
            classes[signature].append(i)
        
        return classes
    
    def _tree_signature(self, tree):
        """Ağacın eşsiz imzası (boyut, açılar vs.)"""
        vertices = sorted(tree.vertices)  # Normalize
        return tuple(round(v, 4) for v in vertices)
    
    def _build_symmetry_group(self):
        """
        Özdeş ağaçların oluşturduğu permütasyon grubu
        Eğer k tane özdeş ağaç varsa, S_k simetri grubu oluşur
        """
        generators = []
        
        for signature, indices in self.equivalence_classes.items():
            if len(indices) > 1:
                # S_k grubunun generatörleri: adjacent transpositions
                for i in range(len(indices) - 1):
                    # (i, i+1) transposition
                    perm = list(range(len(self.trees)))
                    perm[indices[i]], perm[indices[i+1]] = perm[indices[i+1]], perm[indices[i]]
                    generators.append(Permutation(perm))
        
        if generators:
            return PermutationGroup(generators)
        else:
            return PermutationGroup([Permutation(range(len(self.trees)))])
    
    def count_distinct_orderings(self):
        """
        Simetri dikkate alınarak kaç farklı sıralama var?
        n! / (k1! * k2! * ... * km!)  
        burada ki = i'inci sınıftaki özdeş ağaç sayısı
        """
        from math import factorial
        
        total = factorial(len(self.trees))
        for indices in self.equivalence_classes.values():
            total //= factorial(len(indices))
        
        return total
    
    def canonical_ordering(self, ordering):
        """
        Bir sıralamanın kanonik temsilcisini bul
        Simetri grubu içinde lexicographically en küçük
        """
        min_ordering = ordering
        
        for perm in self.symmetry_group.generate():
            new_ordering = tuple(perm(i) for i in ordering)
            if new_ordering < min_ordering:
                min_ordering = new_ordering
        
        return min_ordering
```

### 4.2.4 Polya Enumeration Theorem

```python
def polya_enumeration(n_positions, group):
    """
    Pólya sayım teoremi ile farklı konfigürasyon sayısını hesapla
    
    Eğer n ağacı m pozisyona yerleştiriyorsak ve G simetri grubu varsa:
    |X/G| = (1/|G|) * Σ_{g∈G} m^{c(g)}
    
    c(g) = g permütasyonunun cycle sayısı
    """
    from sympy.combinatorics import Permutation
    
    total = 0
    for g in group.generate():
        # g'nin cycle decomposition'ı
        cycles = g.cyclic_form
        n_cycles = len(cycles)
        
        # Her cycle için m^(cycle sayısı)
        total += n_positions ** n_cycles
    
    return total // group.order()
```

### 4.2.5 Symmetry Breaking Constraints

Optimizasyon sırasında simetriyi kır:

```python
class SymmetryBreaker:
    """
    Simetrik çözümleri eleyerek arama uzayını küçült
    """
    
    def __init__(self, n_trees):
        self.n_trees = n_trees
    
    def add_symmetry_breaking_constraints(self, model):
        """
        Constraint programming modeline simetri kırma kısıtları ekle
        """
        # Kısıt 1: İlk ağaç sol-alt çeyrekte olmalı
        model.add_constraint(model.trees[0].x <= model.box_size / 2)
        model.add_constraint(model.trees[0].y <= model.box_size / 2)
        
        # Kısıt 2: Özdeş ağaçlar için sıralama zorla
        for group in self.find_identical_trees():
            for i in range(len(group) - 1):
                # Lexicographic ordering: (x_i, y_i) < (x_{i+1}, y_{i+1})
                t1, t2 = group[i], group[i+1]
                model.add_constraint(
                    (model.trees[t1].x < model.trees[t2].x) |
                    ((model.trees[t1].x == model.trees[t2].x) & 
                     (model.trees[t1].y <= model.trees[t2].y))
                )
        
        # Kısıt 3: Rotasyon simetrisi kırma
        # Çözümün toplam "momenti" pozitif olmalı
        model.add_constraint(
            sum(t.x * t.y for t in model.trees) >= 0
        )
        
        return model
    
    def is_canonical(self, solution):
        """
        Çözüm kanonik formda mı kontrol et
        Sadece kanonik çözümleri değerlendir
        """
        # İlk ağaç sol-alt çeyrekte mi?
        if solution.trees[0].x > solution.box_size / 2:
            return False
        if solution.trees[0].y > solution.box_size / 2:
            return False
        
        # Özdeş ağaçlar sıralı mı?
        for group in self.find_identical_trees():
            positions = [(solution.trees[i].x, solution.trees[i].y) for i in group]
            if positions != sorted(positions):
                return False
        
        return True
```

### 4.2.6 Group-Equivariant Neural Networks

Simetriyi neural network'e yerleştir:

```python
import torch
import torch.nn as nn

class EquivariantLayer(nn.Module):
    """
    Permütasyon-eşdeğer (equivariant) layer
    Girdi ağaçların sırası değişirse, çıktı da aynı şekilde değişir
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Shared weights for all trees (permutation equivariance)
        self.tree_transform = nn.Linear(in_features, out_features)
        
        # Global pooling for context
        self.global_transform = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        # x: (batch, n_trees, features)
        
        # Per-tree transformation
        tree_out = self.tree_transform(x)
        
        # Global context (permutation invariant)
        global_context = x.mean(dim=1, keepdim=True)  # (batch, 1, features)
        global_out = self.global_transform(global_context)
        global_out = global_out.expand(-1, x.size(1), -1)
        
        return tree_out + global_out


class RotationEquivariantConv(nn.Module):
    """
    Rotasyon-eşdeğer convolution
    Girdinin rotasyonu çıktının rotasyonuna eşit
    """
    
    def __init__(self, in_channels, out_channels, n_rotations=8):
        super().__init__()
        self.n_rotations = n_rotations
        
        # Tek bir filtre öğren, rotasyonlarını kullan
        self.base_filter = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
    
    def forward(self, x):
        outputs = []
        
        for r in range(self.n_rotations):
            # Filtreyi r * (360/n_rotations) derece döndür
            angle = r * 360 / self.n_rotations
            rotated_filter = self._rotate_filter(self.base_filter, angle)
            
            out = nn.functional.conv2d(x, rotated_filter, padding=1)
            outputs.append(out)
        
        # Stack rotations as channels
        return torch.stack(outputs, dim=2)  # (batch, channels, rotations, H, W)
```

## 4.3 Hibrit Yaklaşım: RL + Grup Teorisi

```python
class SymmetryAwareRL:
    """
    RL + Grup Teorisi Hibrit Yaklaşım
    """
    
    def __init__(self, n_trees):
        self.symmetry = SymmetryGroup(n_trees)
        self.permutation = PermutationSymmetry(n_trees)
        self.policy = GNNPackingPolicy()
    
    def get_canonical_state(self, state):
        """State'i kanonik forma getir"""
        return self.symmetry.canonical_form(state)
    
    def train_step(self, states, actions, rewards):
        # Simetrik state'leri grupla
        canonical_states = [self.get_canonical_state(s) for s in states]
        
        # Her kanonik state için tek bir update
        unique_experiences = {}
        for s, a, r in zip(canonical_states, actions, rewards):
            key = hash(s)
            if key not in unique_experiences:
                unique_experiences[key] = []
            unique_experiences[key].append((a, r))
        
        # Agregated update
        for key, experiences in unique_experiences.items():
            avg_reward = np.mean([r for _, r in experiences])
            self.policy.update(key, avg_reward)
    
    def evaluate_with_symmetry(self, state):
        """
        Değerlendirme sırasında tüm simetrik versiyonları kontrol et
        """
        best_action = None
        best_value = float('-inf')
        
        for g in self.symmetry.d4_elements:
            transformed_state = self.symmetry.apply_d4(state, g)
            action, value = self.policy.evaluate(transformed_state)
            
            if value > best_value:
                best_value = value
                # Action'ı orijinal koordinat sistemine geri dönüştür
                best_action = self.symmetry.inverse_transform_action(action, g)
        
        return best_action, best_value
```

## 4.4 Yaklaşım Karşılaştırma Tablosu

| Yaklaşım | Avantaj | Dezavantaj | Ne Zaman Kullan |
|----------|---------|------------|-----------------|
| **RL (DQN/PPO)** | Otomatik öğrenme | Yavaş training | Çok puzzle varsa |
| **RL + GNN** | İlişkileri modeller | Karmaşık | Ağaç sayısı değişiyorsa |
| **Grup Teorisi** | Arama uzayını küçültür | Matematik yoğun | Simetri çoksa |
| **Symmetry Breaking** | Hızlandırır | Manuel tasarım | Optimizasyon içinde |
| **Equivariant NN** | Verimli öğrenme | Uygulama zor | RL ile birlikte |

---

# Bölüm 5: Reward Tasarımı (Incremental Bounding Box Yaklaşımı)

## 5.1 Temel Fikir

```
Adım 1: Ağaç₁ yerleştir → BBox₁ → Boşluk₁ = BBox₁.area - Ağaç₁.area → Reward₁ = -Boşluk₁
Adım 2: Ağaç₂ yerleştir → BBox₂ → Boşluk₂ = BBox₂.area - (Ağaç₁ + Ağaç₂).area → Reward₂ = -Boşluk₂
...
```

## 5.2 Potansiyel Problemler

### Problem 1: Greedy Tuzağı

```
Örnek: 3 üçgen ağaç yerleştiriyorsun

Greedy yaklaşım:
┌─────────┐
│ △       │  Adım 1: Küçük bbox ✓
│         │
└─────────┘

┌─────────┐
│ △  △    │  Adım 2: Yine küçük bbox ✓
│         │
└─────────┘

┌─────────────┐
│ △  △        │  Adım 3: Üçüncü ağaç sığmıyor, 
│        △    │  bbox büyümek ZORUNDA!
└─────────────┘

Oysa optimal:
┌───────┐
│ △ ▽ △ │  Ağaçları iç içe geçirerek
│       │  çok daha küçük bbox!
└───────┘
```

**Sonuç:** Her adımda lokal optimum, global optimumu kaçırabilir.

### Problem 2: Reward Shaping Sorunu

```python
# Temel yaklaşım
reward_t = -(bbox_area_t - total_tree_area_t)

# Problem: İlk ağaçlar için reward çok küçük (az boşluk)
# Son ağaçlar için reward çok büyük (çok boşluk)
# Bu dengesizlik öğrenmeyi zorlaştırır
```

## 5.3 Geliştirilmiş Reward Tasarımları

### 5.3.1 Delta-Based Reward

```python
class DeltaRewardEnv:
    """
    Her adımda bbox'ın ne kadar ARTTI'ğına bak
    """
    
    def __init__(self, trees):
        self.trees = trees
        self.reset()
    
    def reset(self):
        self.placed = []
        self.prev_bbox_area = 0
        self.total_tree_area = 0
        return self.get_state()
    
    def step(self, action):
        tree_idx, x, y, theta = action
        tree = self.trees[tree_idx]
        
        # Yerleştir
        placed_tree = place_tree(tree, x, y, theta)
        
        # Validity check
        if self.has_overlap(placed_tree) or self.out_of_bounds(placed_tree):
            return self.get_state(), -10, False, {}  # Penalty
        
        self.placed.append(placed_tree)
        self.total_tree_area += tree.area
        
        # Yeni bbox hesapla
        new_bbox_area = self.compute_bbox_area()
        
        # DELTA reward: bbox ne kadar arttı?
        bbox_increase = new_bbox_area - self.prev_bbox_area
        tree_area = tree.area
        
        # İdeal: bbox artışı = ağaç alanı (yani sıfır boşluk)
        # Gerçek: bbox artışı > ağaç alanı (boşluk var)
        wasted_space = bbox_increase - tree_area
        
        reward = -wasted_space  # Boşluk ne kadar az, o kadar iyi
        
        self.prev_bbox_area = new_bbox_area
        
        done = len(self.placed) == len(self.trees)
        
        return self.get_state(), reward, done, {}
```

### 5.3.2 Efficiency Ratio Reward

```python
class EfficiencyRewardEnv:
    """
    Packing efficiency'yi reward olarak kullan
    Efficiency = Total Tree Area / BBox Area
    """
    
    def step(self, action):
        # ... yerleştirme kodu ...
        
        bbox_area = self.compute_bbox_area()
        efficiency = self.total_tree_area / bbox_area  # 0 ile 1 arası
        
        # Önceki efficiency ile karşılaştır
        prev_efficiency = self.prev_total_area / self.prev_bbox_area if self.prev_bbox_area > 0 else 0
        
        # Efficiency artışı/azalışı
        reward = (efficiency - prev_efficiency) * 100  # Scale up
        
        self.prev_efficiency = efficiency
        
        return self.get_state(), reward, done, {}
```

### 5.3.3 Potential-Based Reward Shaping

```python
class PotentialBasedRewardEnv:
    """
    Potential function: Φ(s) = -optimal_bbox_estimate(s)
    Reward: R(s,a,s') = γΦ(s') - Φ(s) + base_reward
    
    Bu teorik olarak optimal policy'yi değiştirmez!
    """
    
    def potential(self, state):
        """
        Kalan ağaçları yerleştirmek için gereken
        minimum ek alan tahmini
        """
        remaining_area = sum(t.area for t in self.remaining_trees)
        current_bbox = self.compute_bbox_area()
        
        # Alt sınır tahmini: kalan alan + mevcut bbox
        # Üst sınır: sqrt(total_area) kare
        lower_bound = current_bbox + remaining_area * 0.7  # %70 efficiency varsay
        
        return -lower_bound  # Negatif çünkü minimize ediyoruz
    
    def step(self, action):
        old_potential = self.potential(self.get_state())
        
        # ... yerleştirme ...
        
        new_potential = self.potential(self.get_state())
        
        # Shaped reward
        gamma = 0.99
        shaping = gamma * new_potential - old_potential
        
        base_reward = -self.compute_wasted_space()
        
        reward = base_reward + shaping
        
        return self.get_state(), reward, done, {}
```

### 5.3.4 Multi-Objective Reward

```python
class MultiObjectiveRewardEnv:
    """
    Birden fazla hedefi dengele
    """
    
    def step(self, action):
        # ... yerleştirme ...
        
        # Reward bileşenleri
        r_bbox = -self.compute_bbox_area() / self.max_possible_bbox  # Normalized
        r_efficiency = self.total_tree_area / self.compute_bbox_area()
        r_compactness = -self.compute_convex_hull_waste()
        r_balance = -self.compute_center_of_mass_deviation()
        
        # Weighted sum
        reward = (
            0.4 * r_bbox +
            0.3 * r_efficiency +
            0.2 * r_compactness +
            0.1 * r_balance
        )
        
        return self.get_state(), reward, done, {}
```

### 5.3.5 Sparse + Dense Hybrid Reward

```python
class HybridRewardEnv:
    """
    Dense reward (her adım) + Sparse reward (episode sonu)
    """
    
    def __init__(self, trees, dense_weight=0.3, sparse_weight=0.7):
        self.trees = trees
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def step(self, action):
        tree_idx, x, y, theta = action
        
        # Yerleştir
        success = self.place_tree(tree_idx, x, y, theta)
        
        if not success:
            return self.get_state(), -1, False, {}
        
        done = self.all_placed()
        
        # Dense reward: incremental feedback
        dense_reward = self.compute_dense_reward()
        
        if done:
            # Sparse reward: final evaluation
            final_bbox = self.compute_final_bbox()
            optimal_estimate = math.sqrt(self.total_tree_area)  # Teorik alt sınır
            
            # Ne kadar optimal'e yakınsak o kadar iyi
            sparse_reward = optimal_estimate / final_bbox  # 0-1 arası, 1 = perfect
            sparse_reward = sparse_reward ** 2 * 100  # Amplify good solutions
            
            total_reward = self.dense_weight * dense_reward + self.sparse_weight * sparse_reward
        else:
            total_reward = self.dense_weight * dense_reward
        
        return self.get_state(), total_reward, done, {}
    
    def compute_dense_reward(self):
        """Her adım için hızlı feedback"""
        
        # 1. Bbox verimliliği
        bbox_area = self.compute_bbox_area()
        efficiency = self.total_placed_area / bbox_area
        
        # 2. Yerleşim kompaktlığı - ağaçlar birbirine yakın mı?
        if len(self.placed) >= 2:
            avg_distance = self.average_pairwise_distance()
            compactness = 1 / (1 + avg_distance)
        else:
            compactness = 1
        
        # 3. Boş alan dağılımı - büyük boşluklar kötü
        fragmentation = self.compute_fragmentation_penalty()
        
        reward = 0.5 * efficiency + 0.3 * compactness - 0.2 * fragmentation
        
        return reward
```

### 5.3.6 Look-Ahead Reward

```python
class LookAheadRewardEnv:
    """
    Sadece şu anki duruma değil, gelecekteki potansiyele de bak
    """
    
    def step(self, action):
        # Yerleştir
        self.place_tree(action)
        
        # Şu anki durum
        current_bbox = self.compute_bbox_area()
        current_efficiency = self.total_placed_area / current_bbox
        
        # Gelecek potansiyeli: Kalan ağaçlar için ne kadar "iyi" alan kaldı?
        remaining_trees = self.get_remaining_trees()
        usable_space = self.estimate_usable_space()  # Boş alanın ne kadarı kullanılabilir
        required_space = sum(t.area for t in remaining_trees)
        
        future_potential = min(1.0, usable_space / required_space) if required_space > 0 else 1.0
        
        # Reward: şu an + gelecek
        reward = 0.4 * current_efficiency + 0.6 * future_potential
        
        return self.get_state(), reward, done, {}
    
    def estimate_usable_space(self):
        """
        Boş alanın ne kadarı gerçekten kullanılabilir?
        Küçük, izole boşluklar kullanılamaz
        """
        empty_regions = self.find_empty_regions()
        
        usable = 0
        min_tree_area = min(t.area for t in self.get_remaining_trees())
        
        for region in empty_regions:
            if region.area >= min_tree_area:
                # Bu bölgeye en az bir ağaç sığabilir
                usable += region.area
        
        return usable
```

## 5.4 Tam Environment Implementasyonu

```python
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import math

class TreePackingEnv:
    def __init__(self, trees, grid_size=100):
        self.original_trees = trees
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        self.trees = [t.copy() for t in self.original_trees]
        self.placed = []
        self.remaining = list(range(len(self.trees)))
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.total_placed_area = 0
        self.step_count = 0
        return self._get_state()
    
    def step(self, action):
        """
        action: (tree_idx, x, y, theta)
        """
        tree_idx, x, y, theta = action
        
        if tree_idx not in self.remaining:
            return self._get_state(), -5, False, {'error': 'already_placed'}
        
        tree = self.trees[tree_idx]
        
        # Geçerlilik kontrolü
        if not self._is_valid_placement(tree, x, y, theta):
            return self._get_state(), -2, False, {'error': 'invalid'}
        
        # Yerleştir
        self._place_tree(tree_idx, x, y, theta)
        
        # Reward hesapla
        reward = self._compute_reward()
        
        # Done kontrolü
        done = len(self.remaining) == 0
        
        # Final bonus/penalty
        if done:
            final_score = self._compute_final_score()
            reward += final_score
        
        self.step_count += 1
        
        return self._get_state(), reward, done, {}
    
    def _compute_reward(self):
        """
        Geliştirilmiş incremental bounding box reward
        """
        # Mevcut bbox
        bbox = self._compute_bbox()
        bbox_area = bbox['width'] * bbox['height']
        
        # Efficiency
        efficiency = self.total_placed_area / bbox_area if bbox_area > 0 else 0
        
        # Delta-based: Son ağacın katkısı
        if len(self.placed) >= 2:
            prev_bbox = self._compute_bbox(exclude_last=True)
            prev_area = prev_bbox['width'] * prev_bbox['height']
            
            bbox_increase = bbox_area - prev_area
            tree_area = self.trees[self.placed[-1]['idx']].area
            
            # İdeal: bbox_increase == tree_area
            # Gerçek: bbox_increase >= tree_area
            waste_ratio = (bbox_increase - tree_area) / tree_area if tree_area > 0 else 0
            
            incremental_reward = -waste_ratio  # Waste az ise reward yüksek
        else:
            incremental_reward = 0
        
        # Gelecek potansiyeli
        future_potential = self._estimate_future_potential()
        
        # Combine
        reward = (
            0.3 * efficiency +           # Genel verimlilik
            0.4 * incremental_reward +   # Bu adımın kalitesi
            0.3 * future_potential        # Gelecek için iyi pozisyon
        )
        
        return reward
    
    def _estimate_future_potential(self):
        """
        Kalan ağaçları yerleştirmek ne kadar kolay olacak?
        """
        if not self.remaining:
            return 1.0
        
        remaining_area = sum(self.trees[i].area for i in self.remaining)
        
        # Mevcut bbox'ı biraz genişletsek kalan ağaçlar sığar mı?
        bbox = self._compute_bbox()
        current_area = bbox['width'] * bbox['height']
        
        # Teorik minimum: mevcut + kalan alanlar
        theoretical_min = self.total_placed_area + remaining_area
        
        # Tahmini final bbox (basit heuristic)
        estimated_final = current_area + remaining_area * 1.3  # %30 overhead varsay
        
        # Ne kadar optimale yakınız?
        potential = theoretical_min / estimated_final if estimated_final > 0 else 0
        
        return potential
    
    def _compute_final_score(self):
        """
        Episode sonunda büyük bonus/penalty
        """
        bbox = self._compute_bbox()
        final_area = bbox['width'] * bbox['height']
        
        # Teorik optimal: tüm ağaçların alanı
        theoretical_optimal = sum(t.area for t in self.trees)
        
        # Efficiency
        final_efficiency = theoretical_optimal / final_area
        
        # Bonus: 1'e ne kadar yakınsa o kadar iyi
        # efficiency=0.8 → bonus=64, efficiency=0.9 → bonus=81
        bonus = (final_efficiency ** 2) * 100
        
        return bonus
    
    def _get_state(self):
        """
        State representation
        """
        return {
            'grid': self.grid.copy(),
            'placed_trees': [
                {
                    'idx': p['idx'],
                    'x': p['x'],
                    'y': p['y'],
                    'theta': p['theta'],
                    'area': self.trees[p['idx']].area
                }
                for p in self.placed
            ],
            'remaining_trees': [
                {
                    'idx': i,
                    'area': self.trees[i].area,
                    'width': self.trees[i].width,
                    'height': self.trees[i].height
                }
                for i in self.remaining
            ],
            'bbox': self._compute_bbox(),
            'efficiency': self.total_placed_area / max(1, self._compute_bbox_area()),
            'step': self.step_count
        }
    
    def _compute_bbox(self, exclude_last=False):
        """Mevcut yerleşimin bounding box'ını hesapla"""
        trees_to_consider = self.placed[:-1] if exclude_last and len(self.placed) > 1 else self.placed
        
        if not trees_to_consider:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        min_x = min(p['x'] for p in trees_to_consider)
        min_y = min(p['y'] for p in trees_to_consider)
        max_x = max(p['x'] + self.trees[p['idx']].width for p in trees_to_consider)
        max_y = max(p['y'] + self.trees[p['idx']].height for p in trees_to_consider)
        
        return {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
    
    def _compute_bbox_area(self):
        bbox = self._compute_bbox()
        return bbox['width'] * bbox['height']
```

## 5.5 DQN Agent Implementasyonu

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            # Mask invalid actions
            masked_q = q_values.clone()
            masked_q[~valid_actions] = float('-inf')
            return masked_q.argmax().item()
    
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones)
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_packing_agent(trees, n_episodes=10000):
    env = TreePackingEnv(trees)
    agent = DQNAgent(state_dim=..., action_dim=...)
    
    best_score = float('inf')
    scores = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, env.get_valid_actions())
            next_state, reward, done, info = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, done))
            agent.train_step()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Episode sonu
        final_bbox = env._compute_bbox_area()
        scores.append(final_bbox)
        
        if final_bbox < best_score:
            best_score = final_bbox
            print(f"Episode {episode}: New best! BBox = {final_bbox:.2f}")
        
        # Target network update
        if episode % 100 == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
    
    return agent, scores
```

## 5.6 Reward Tasarım Prensipleri Özeti

| Prensip | Açıklama | Örnek |
|---------|----------|-------|
| **Dense + Sparse** | Her adımda feedback + episode sonu bonus | 0.3×dense + 0.7×sparse |
| **Delta-based** | Mutlak değil, değişime bak | bbox_increase - tree_area |
| **Normalized** | Scale sorunlarını önle | efficiency = area/bbox |
| **Future-aware** | Sadece şimdiyi değil, geleceği de düşün | usable_space/required_space |
| **Potential shaping** | Optimal policy'yi koruyarak şekille | γΦ(s') - Φ(s) |

---

# Bölüm 6: Önemli Kaynaklar

## 6.1 Akademik Makaleler

1. **2D Irregular Packing Surveys:**
   - "A Survey on Two-Dimensional Irregular Packing Problems" - NFP yönteminin detaylı açıklaması
   - "Irregular packing using the line and arc no-fit polygon" - Burke et al.

2. **RL for Packing:**
   - "Learning to Pack Rectangles with Deep Reinforcement Learning"
   - "A Deep Reinforcement Learning Approach to Bin Packing"

3. **Grup Teorisi & Optimizasyon:**
   - "Symmetry Breaking in Constraint Programming"
   - "Exploiting Symmetries in Mathematical Programming"

## 6.2 Kütüphaneler

```python
# Geometri
pip install shapely
pip install pyclipper
pip install scipy

# Optimizasyon
pip install networkx
pip install python-constraint
pip install ortools

# RL
pip install gymnasium
pip install stable-baselines3
pip install torch

# Hızlandırma
pip install numba
pip install cython

# Visualization
pip install matplotlib
pip install plotly
```

## 6.3 Yarışma Kaynakları

- Kaggle Notebooks: "Santa 2025 - Getting Started" notebook'larını incele
- Discussion Forum'u düzenli takip et
- Geçmiş Santa yarışmalarının çözümlerini incele

---

# Bölüm 7: Son İpuçları

1. **Erken ve sık submit et** - Leaderboard feedback'i önemli
2. **Küçük ağaç sayılarından başla** - 1-20 ağaç için optimal çözümleri bul
3. **Görselleştirme yap** - Çözümlerini matplotlib ile görselleştir
4. **Rudolph Prize için** - Erken liderliğe geç ve koru
5. **Ensemble kullan** - Farklı algoritmaları birleştir
6. **Simetriyi exploit et** - Arama uzayını küçült

---

**Başarılar!** 🎅🎄
