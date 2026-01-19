# 🧠 Reinforcement Learning Detaylı Rehber
## Tree Packing Problemi İçin Kapsamlı RL Anlatımı

---

# Bölüm 1: Reinforcement Learning Temelleri

## 1.1 RL Nedir?

Reinforcement Learning (Pekiştirmeli Öğrenme), bir **agent**'ın (etmen) bir **environment** (ortam) ile etkileşime girerek, aldığı **reward**'ları (ödüller) maksimize edecek şekilde **policy** (strateji) öğrenmesidir.

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Döngüsü                               │
│                                                             │
│    ┌─────────┐     action (a_t)      ┌─────────────┐       │
│    │         │ ──────────────────►   │             │       │
│    │  Agent  │                       │ Environment │       │
│    │         │ ◄──────────────────   │             │       │
│    └─────────┘   state (s_t+1)       └─────────────┘       │
│                  reward (r_t+1)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 Temel Kavramlar

### State (Durum) - s
Ortamın mevcut durumunu tanımlayan bilgi. Tree packing için:
- Hangi ağaçlar yerleştirildi?
- Nerede yerleştirildiler?
- Hangi ağaçlar kaldı?
- Mevcut bounding box ne?

### Action (Eylem) - a
Agent'ın yapabileceği seçimler. Tree packing için:
- Hangi ağacı seç?
- Nereye yerleştir (x, y)?
- Hangi açıyla döndür (θ)?

### Reward (Ödül) - r
Bir eylemin ne kadar iyi olduğunu gösteren sayısal değer.
- Pozitif: İyi eylem
- Negatif: Kötü eylem
- Sıfır: Nötr

### Policy (Strateji) - π
State'den action'a mapping. "Bu durumda ne yapmalıyım?"
```
π(s) → a        # Deterministic policy
π(a|s) → [0,1]  # Stochastic policy (olasılık dağılımı)
```

### Value Function (Değer Fonksiyonu) - V(s)
Bir state'in ne kadar "değerli" olduğu. O state'den başlayarak beklenen toplam reward.
```
V(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s]
```

### Q-Function (Eylem-Değer Fonksiyonu) - Q(s, a)
Bir state'de belirli bir action almanın değeri.
```
Q(s, a) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s, a_t = a]
```

### Discount Factor (İndirim Faktörü) - γ
Gelecekteki ödüllerin şimdiki değeri. 0 ≤ γ ≤ 1
- γ = 0: Sadece anlık ödül önemli
- γ = 1: Tüm gelecek ödüller eşit önemli
- γ = 0.99: Tipik değer, gelecek önemli ama yakın gelecek daha önemli

## 1.3 MDP (Markov Decision Process)

RL problemleri genellikle MDP olarak formüle edilir:

```
MDP = (S, A, P, R, γ)

S: State uzayı (tüm olası durumlar)
A: Action uzayı (tüm olası eylemler)
P: Transition probability P(s'|s,a) - bir state ve action'dan sonraki state olasılığı
R: Reward function R(s,a,s') - ödül fonksiyonu
γ: Discount factor
```

### Tree Packing için MDP Formülasyonu

```python
class TreePackingMDP:
    """
    Tree Packing probleminin MDP formülasyonu
    """
    
    # STATE SPACE (S)
    # ─────────────────────────────────────────────────────────
    # State, aşağıdaki bilgileri içerir:
    state = {
        'grid': np.array,           # H x W occupancy grid (0: boş, 1: dolu)
        'placed_trees': [           # Yerleştirilmiş ağaçların listesi
            {'id': int, 'x': float, 'y': float, 'theta': float},
            ...
        ],
        'remaining_trees': [        # Kalan ağaçların listesi
            {'id': int, 'area': float, 'vertices': list},
            ...
        ],
        'current_bbox': {           # Mevcut bounding box
            'x_min': float, 'y_min': float,
            'x_max': float, 'y_max': float
        },
        'total_placed_area': float, # Yerleştirilen toplam alan
        'step': int                 # Kaçıncı adım
    }
    
    # ACTION SPACE (A)
    # ─────────────────────────────────────────────────────────
    # Discrete version:
    action_discrete = {
        'tree_idx': int,            # [0, n_remaining_trees)
        'position_idx': int,        # [0, n_grid_positions)
        'rotation_idx': int         # [0, n_rotation_angles)
    }
    
    # Continuous version:
    action_continuous = {
        'tree_idx': int,            # [0, n_remaining_trees) - discrete
        'x': float,                 # [0, 1] - normalized position
        'y': float,                 # [0, 1] - normalized position
        'theta': float              # [0, 2π] - rotation angle
    }
    
    # TRANSITION FUNCTION P(s'|s,a)
    # ─────────────────────────────────────────────────────────
    # Deterministic: Verilen state ve action için tek bir sonraki state var
    # P(s'|s,a) = 1 eğer s' = next_state(s, a)
    # P(s'|s,a) = 0 diğer tüm s' için
    
    # REWARD FUNCTION R(s,a,s')
    # ─────────────────────────────────────────────────────────
    def reward(self, state, action, next_state):
        if not is_valid_placement(action):
            return -10  # Invalid action penalty
        
        # Efficiency-based reward
        efficiency = next_state['total_placed_area'] / bbox_area(next_state)
        
        # Done bonus
        if is_terminal(next_state):
            return 100 * (efficiency ** 2)
        
        return efficiency - prev_efficiency
```

---

# Bölüm 2: Value-Based Methods (Değer Tabanlı Yöntemler)

## 2.1 Q-Learning

En temel RL algoritması. Q-değerlerini iteratif olarak günceller.

### Bellman Equation
```
Q(s,a) = R(s,a) + γ * max_a' Q(s', a')
```

### Q-Learning Update Rule
```
Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
         ─────────────────────────────────────────────
                         TD Error
```

### Tabular Q-Learning Implementation

```python
import numpy as np
from collections import defaultdict

class TabularQLearning:
    """
    Basit Q-Learning implementasyonu
    Küçük state/action uzayları için uygun
    """
    
    def __init__(self, n_actions, learning_rate=0.1, gamma=0.99, epsilon=1.0):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.n_actions = n_actions
    
    def state_to_key(self, state):
        """State'i hashable key'e çevir"""
        # Basit örnek: grid'i tuple'a çevir
        return tuple(state['grid'].flatten())
    
    def select_action(self, state, valid_actions=None):
        """
        Epsilon-greedy action selection
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        
        # Exploration: rastgele action
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploitation: en iyi Q-değerine sahip action
        state_key = self.state_to_key(state)
        q_values = self.q_table[state_key]
        
        # Sadece valid action'lar arasından seç
        valid_q = {a: q_values[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Current Q value
        current_q = self.q_table[state_key][action]
        
        # Target Q value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # TD Error
        td_error = target_q - current_q
        
        # Update
        self.q_table[state_key][action] += self.lr * td_error
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return td_error
    
    def train_episode(self, env):
        """Bir episode eğit"""
        state = env.reset()
        total_reward = 0
        
        while True:
            action = self.select_action(state, env.get_valid_actions())
            next_state, reward, done, info = env.step(action)
            
            self.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return total_reward
```

## 2.2 Deep Q-Network (DQN)

Q-table yerine neural network kullanarak Q-değerlerini tahmin eder.
Büyük/continuous state uzayları için gerekli.

### DQN'in Temel Fikirleri

1. **Neural Network ile Q-Function Approximation**
   ```
   Q(s, a; θ) ≈ Q*(s, a)
   ```

2. **Experience Replay**
   - Deneyimleri bir buffer'da sakla
   - Rastgele örnekleyerek eğit
   - Correlation'ı kırar, sample efficiency artırır

3. **Target Network**
   - İki network: online ve target
   - Target network yavaşça güncellenir
   - Training stability sağlar

### DQN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DQNetwork(nn.Module):
    """
    Deep Q-Network mimarisi
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Stability için
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Input: state tensor (batch_size, state_dim)
        Output: Q-values (batch_size, action_dim)
        """
        return self.network(state)


class ConvDQNetwork(nn.Module):
    """
    Convolutional DQN - Grid-based state için
    """
    
    def __init__(self, grid_size, n_channels, action_dim, tree_feature_dim):
        super().__init__()
        
        # CNN for spatial features (grid)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(grid_size, n_channels)
        
        # MLP for tree features
        self.tree_encoder = nn.Sequential(
            nn.Linear(tree_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Combined layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def _get_conv_output_size(self, grid_size, n_channels):
        """Conv layers'ın output boyutunu hesapla"""
        dummy_input = torch.zeros(1, n_channels, grid_size, grid_size)
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        return output.shape[1]
    
    def forward(self, grid, tree_features):
        """
        Input:
            grid: (batch, channels, H, W)
            tree_features: (batch, tree_feature_dim)
        Output:
            Q-values: (batch, action_dim)
        """
        spatial_features = self.conv_layers(grid)
        tree_encoding = self.tree_encoder(tree_features)
        combined = torch.cat([spatial_features, tree_encoding], dim=1)
        return self.fc_layers(combined)
```

### Experience Replay Buffer

```python
class ReplayBuffer:
    """
    Experience Replay Buffer
    Geçmiş deneyimleri saklar ve rastgele örnekleme yapar
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Deneyim ekle"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Rastgele batch örnekle"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay
    Daha "şaşırtıcı" (yüksek TD error) deneyimlere öncelik verir
    """
    
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Yeni deneyim ekle (maksimum priority ile)"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Priority'ye göre örnekle"""
        n = len(self.buffer)
        
        # Priority'leri olasılığa çevir
        priorities = self.priorities[:n] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Örnekle
        indices = np.random.choice(n, batch_size, p=probabilities, replace=False)
        
        # Importance sampling weights
        weights = (n * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Beta'yı artır
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool),
            torch.tensor(weights, dtype=torch.float32),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        """TD error'a göre priority'leri güncelle"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Small constant for stability
            self.max_priority = max(self.max_priority, self.priorities[idx])
```

### Complete DQN Agent

```python
class DQNAgent:
    """
    Complete DQN Agent
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[256, 256],
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        
        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = DQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = DQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network always in eval mode
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training stats
        self.update_count = 0
        self.loss_history = []
    
    def select_action(self, state, valid_actions=None):
        """
        Epsilon-greedy action selection
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze()
            
            # Mask invalid actions
            mask = torch.full((self.action_dim,), float('-inf'))
            mask[valid_actions] = 0
            masked_q = q_values + mask.to(self.device)
            
            return masked_q.argmax().item()
    
    def train_step(self):
        """
        Bir training step
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.q_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            # Online network action seçer
            next_actions = self.q_network(next_states).argmax(1)
            # Target network değeri hesaplar
            next_q = self.target_network(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (~dones)
        
        # Loss (Huber Loss - daha robust)
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.loss_history.append(loss.item())
        return loss.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Deneyimi buffer'a ekle"""
        self.memory.push(state, action, reward, next_state, done)
    
    def save(self, path):
        """Model kaydet"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)
    
    def load(self, path):
        """Model yükle"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
```

## 2.3 DQN Varyantları

### Double DQN
Overestimation bias'ı azaltır.

```python
# Standard DQN:
target = r + γ * max_a' Q_target(s', a')

# Double DQN:
a* = argmax_a' Q_online(s', a')  # Online network action seçer
target = r + γ * Q_target(s', a*)  # Target network değeri hesaplar
```

### Dueling DQN
Q-değerini Value ve Advantage olarak ayırır.

```python
class DuelingDQN(nn.Module):
    """
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    
    V(s): State value - bu durumda olmak ne kadar iyi
    A(s,a): Advantage - bu action ortalamadan ne kadar iyi
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single value
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)  # One per action
        )
    
    def forward(self, state):
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
```

### Noisy DQN
Epsilon-greedy yerine parametric noise ile exploration.

```python
class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noise
    Weight'lere öğrenilebilir noise ekler
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        # Initialize
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Yeni noise sample'la"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorized: outer product
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
```

### Rainbow DQN
Tüm iyileştirmeleri birleştirir:
- Double DQN
- Dueling DQN  
- Prioritized Experience Replay
- Multi-step Learning
- Distributional RL
- Noisy Nets

---

# Bölüm 3: Policy Gradient Methods (Strateji Gradyan Yöntemleri)

## 3.1 Policy Gradient Temelleri

Value-based yöntemlerin aksine, doğrudan policy'yi optimize eder.

### Neden Policy Gradient?
1. **Continuous actions** için doğal
2. **Stochastic policies** öğrenebilir
3. Bazı problemlerde daha **stable**

### Policy Gradient Theorem

```
∇_θ J(θ) = E_π [ ∇_θ log π(a|s;θ) * Q^π(s,a) ]
```

Bu formül şunu söylüyor:
- İyi sonuç veren action'ların olasılığını artır
- Kötü sonuç veren action'ların olasılığını azalt

## 3.2 REINFORCE (Monte Carlo Policy Gradient)

En basit policy gradient algoritması.

```python
class REINFORCE:
    """
    REINFORCE (Monte Carlo Policy Gradient)
    
    Avantajlar:
    - Basit ve anlaşılır
    - Unbiased gradient estimate
    
    Dezavantajlar:
    - Yüksek variance
    - Sadece episodic tasks için
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Episode storage
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Stochastic action selection"""
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        
        # Sample from categorical distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Save log probability for later
        self.saved_log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def compute_returns(self):
        """
        Discounted returns hesapla
        G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
        """
        returns = []
        G = 0
        
        # Backwards iteration
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        
        # Normalize (variance reduction)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self):
        """Episode sonunda policy güncelle"""
        returns = self.compute_returns()
        
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            # -log_prob * G (negative because we want gradient ascent)
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
```

## 3.3 Actor-Critic Methods

Policy gradient'ın variance'ını azaltmak için value function kullanır.

```
Actor: Policy π(a|s;θ) - hangi action'ı alacağına karar verir
Critic: Value function V(s;w) - state'in değerini tahmin eder
```

### Advantage Actor-Critic (A2C)

```python
class ActorCritic(nn.Module):
    """
    Actor-Critic Network
    Shared feature extraction + separate heads
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def get_action(self, state):
        action_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, state_value, entropy


class A2CAgent:
    """
    Advantage Actor-Critic Agent
    
    Advantage = Q(s,a) - V(s) ≈ r + γV(s') - V(s) (TD error)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_steps=5,
        device='cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        self.model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def compute_returns(self, rewards, values, dones, last_value):
        """
        N-step returns hesapla
        """
        returns = []
        R = last_value
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        
        return torch.tensor(returns, device=self.device)
    
    def update(self, states, actions, rewards, dones, last_state):
        """
        A2C update
        """
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        
        # Get values and last value
        with torch.no_grad():
            _, last_value = self.model(last_state.unsqueeze(0).to(self.device))
            last_value = last_value.squeeze().item() if not dones[-1] else 0
        
        # Compute returns
        returns = self.compute_returns(rewards, None, dones, last_value)
        
        # Forward pass
        log_probs, values, entropies = self.model.evaluate(states, actions)
        values = values.squeeze()
        
        # Advantage
        advantages = returns - values.detach()
        
        # Losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
```

## 3.4 Proximal Policy Optimization (PPO)

En popüler ve robust policy gradient algoritması.

### PPO'nun Temel Fikri
Policy'yi çok fazla değiştirmeden güncelle (trust region).

```python
class PPO:
    """
    Proximal Policy Optimization
    
    Ana fikir: Policy update'i kısıtla, çok büyük değişiklikler yapma
    
    Clipped objective:
    L = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
    
    r(θ) = π(a|s;θ) / π(a|s;θ_old)  (probability ratio)
    A = advantage
    ε = clip range (typically 0.2)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        mini_batch_size=64,
        device='cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        self.model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def compute_gae(self, rewards, values, dones, last_value):
        """
        Generalized Advantage Estimation (GAE)
        
        GAE-λ: Bias-variance trade-off kontrolü
        λ = 0: TD(0), low variance, high bias
        λ = 1: Monte Carlo, high variance, low bias
        λ = 0.95: Typical value
        """
        advantages = []
        gae = 0
        
        # Add last value
        values = list(values) + [last_value]
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], device=self.device, dtype=torch.float32)
        
        return advantages, returns
    
    def update(self, rollout_buffer):
        """
        PPO Update
        
        1. Rollout buffer'dan veri al
        2. Advantage hesapla
        3. Birden fazla epoch policy'yi güncelle
        """
        states = rollout_buffer['states']
        actions = rollout_buffer['actions']
        old_log_probs = rollout_buffer['log_probs']
        rewards = rollout_buffer['rewards']
        dones = rollout_buffer['dones']
        values = rollout_buffer['values']
        last_value = rollout_buffer['last_value']
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Forward pass
                log_probs, values_pred, entropies = self.model.evaluate(mb_states, mb_actions)
                
                # Probability ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values_pred.squeeze(), mb_returns)
                
                # Entropy bonus
                entropy_loss = -entropies.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
    
    def collect_rollout(self, env, n_steps):
        """
        N adım boyunca rollout topla
        """
        rollout_buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
        state = env.get_state()
        
        for _ in range(n_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                action_probs, value = self.model(state_tensor.unsqueeze(0))
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_state, reward, done, _ = env.step(action.item())
            
            rollout_buffer['states'].append(state_tensor)
            rollout_buffer['actions'].append(action.item())
            rollout_buffer['log_probs'].append(log_prob)
            rollout_buffer['rewards'].append(reward)
            rollout_buffer['dones'].append(done)
            rollout_buffer['values'].append(value.item())
            
            state = next_state
            
            if done:
                state = env.reset()
        
        # Last value for bootstrapping
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            _, last_value = self.model(state_tensor.unsqueeze(0))
            rollout_buffer['last_value'] = last_value.item() if not done else 0
        
        return rollout_buffer
```

## 3.5 Continuous Action Space için PPO

Tree packing'de pozisyon ve rotasyon continuous olabilir.

```python
class ContinuousActorCritic(nn.Module):
    """
    Continuous action space için Actor-Critic
    
    Actor: Gaussian policy
    - mean ve std output'lar
    - Action = mean + std * N(0,1)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head - outputs mean and log_std
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Action bounds
        self.action_low = torch.tensor([0.0, 0.0, 0.0])   # x, y, theta
        self.action_high = torch.tensor([1.0, 1.0, 2*np.pi])
    
    def forward(self, state):
        features = self.shared(state)
        
        # Actor
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
        else:
            # Sample from Gaussian
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
        
        # Squash to action bounds
        action = self.squash_action(action)
        
        return action, value
    
    def squash_action(self, action):
        """Action'ı bounds içine sıkıştır"""
        return self.action_low + (torch.tanh(action) + 1) / 2 * (self.action_high - self.action_low)
    
    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Unsquash action for log_prob calculation
        action_unsquashed = self.unsquash_action(action)
        
        log_prob = dist.log_prob(action_unsquashed).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy
    
    def unsquash_action(self, action):
        """Squashed action'ı orijinal space'e geri çevir"""
        normalized = (action - self.action_low) / (self.action_high - self.action_low) * 2 - 1
        return torch.atanh(torch.clamp(normalized, -0.999, 0.999))
```

---

# Bölüm 4: Graph Neural Networks for RL

## 4.1 Neden GNN?

Tree packing'de:
- Her ağaç bir **node**
- Ağaçlar arası ilişkiler **edge**
- Değişken sayıda ağaç olabilir

GNN bu yapıyı doğal olarak modeller.

## 4.2 GNN Temelleri

### Message Passing Framework

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class SimpleGNNLayer(MessagePassing):
    """
    Basit Graph Neural Network Layer
    
    Message Passing:
    1. Her node komşularından mesaj alır
    2. Mesajları aggregate eder (sum, mean, max)
    3. Kendi feature'ını günceller
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # Mean aggregation
        
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neighbor = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index):
        """
        x: Node features (num_nodes, in_channels)
        edge_index: Edge connections (2, num_edges)
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Start propagation
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        """
        Mesaj fonksiyonu: komşu feature'larını dönüştür
        x_j: Source node features
        """
        return self.lin_neighbor(x_j)
    
    def update(self, aggr_out, x):
        """
        Update fonksiyonu: aggregate edilmiş mesajlarla güncelle
        aggr_out: Aggregated messages
        x: Original node features
        """
        self_features = self.lin_self(x)
        return self.activation(self_features + aggr_out)
```

### Graph Attention Network (GAT)

```python
from torch_geometric.nn import GATConv

class GATLayer(nn.Module):
    """
    Graph Attention Network
    
    Attention mekanizması ile komşuların önemini öğrenir
    """
    
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        
        self.gat = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=True  # Multi-head outputs concatenated
        )
        
        self.norm = nn.LayerNorm(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        out = self.gat(x, edge_index)
        out = self.norm(out)
        out = self.dropout(out)
        return out
```

## 4.3 GNN-based Policy Network

```python
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

class GNNPackingPolicy(nn.Module):
    """
    Graph Neural Network based policy for tree packing
    
    Node features:
    - Tree özellikleri (area, width, height, aspect_ratio)
    - Yerleşim durumu (placed/remaining)
    - Pozisyon (eğer yerleştirilmişse)
    
    Edge features:
    - Mesafe (eğer ikisi de yerleştirilmişse)
    - Boyut benzerliği
    """
    
    def __init__(
        self,
        node_feature_dim=8,
        edge_feature_dim=4,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        
        # Initial node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * n_heads
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=n_heads, dropout=dropout, concat=True)
            )
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim * n_heads, hidden_dim)
        
        # Global graph encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Tree selection head (which tree to place next)
        self.selection_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Position prediction head (where to place)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # x, y, theta
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        """
        Forward pass
        
        data: PyG Data object with:
            - x: Node features
            - edge_index: Edge connections
            - batch: Batch indices
            - remaining_mask: Which trees are remaining
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        remaining_mask = data.remaining_mask
        
        # Initial encoding
        x = self.node_encoder(x)
        
        # GAT message passing
        for gat_layer in self.gat_layers:
            x = F.elu(gat_layer(x, edge_index))
        
        # Final projection
        x = self.final_proj(x)
        
        # Global graph representation
        global_mean = global_mean_pool(x, batch)
        global_max = global_max_pool(x, batch)
        global_repr = torch.cat([global_mean, global_max], dim=-1)
        global_context = self.global_encoder(global_repr)
        
        # Expand global context to all nodes
        global_expanded = global_context[batch]
        
        # Node representations with global context
        node_repr = torch.cat([x, global_expanded], dim=-1)
        
        # Selection scores (only for remaining trees)
        selection_scores = self.selection_head(node_repr).squeeze(-1)
        selection_scores = selection_scores.masked_fill(~remaining_mask, float('-inf'))
        
        # Position predictions (per node)
        position_preds = self.position_head(node_repr)
        
        # Value (graph-level)
        value = self.value_head(global_context)
        
        return selection_scores, position_preds, value
    
    def get_action(self, data, deterministic=False):
        """
        Action selection
        """
        selection_scores, position_preds, value = self.forward(data)
        
        # Select tree
        if deterministic:
            tree_idx = selection_scores.argmax().item()
        else:
            probs = F.softmax(selection_scores, dim=-1)
            dist = torch.distributions.Categorical(probs)
            tree_idx = dist.sample().item()
        
        # Get position for selected tree
        position = position_preds[tree_idx]
        
        # Apply activation for bounded output
        x = torch.sigmoid(position[0])
        y = torch.sigmoid(position[1])
        theta = torch.sigmoid(position[2]) * 2 * np.pi
        
        return tree_idx, (x.item(), y.item(), theta.item()), value


def create_graph_from_state(state, k_neighbors=5):
    """
    State'den PyG graph oluştur
    """
    trees = state['all_trees']
    n_trees = len(trees)
    
    # Node features
    node_features = []
    for i, tree in enumerate(trees):
        is_placed = i in state['placed_indices']
        features = [
            tree['area'] / 1000,        # Normalized area
            tree['width'] / 100,         # Normalized width
            tree['height'] / 100,        # Normalized height
            tree['width'] / tree['height'],  # Aspect ratio
            float(is_placed),            # Placement status
            tree.get('x', 0) / 100 if is_placed else 0,  # Position x
            tree.get('y', 0) / 100 if is_placed else 0,  # Position y
            tree.get('theta', 0) / (2*np.pi) if is_placed else 0  # Rotation
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # Edge index (k-nearest neighbors based on area similarity)
    edge_index = []
    areas = [t['area'] for t in trees]
    
    for i in range(n_trees):
        # Sort other trees by area similarity
        similarities = [(j, abs(areas[i] - areas[j])) for j in range(n_trees) if i != j]
        similarities.sort(key=lambda x: x[1])
        
        # Connect to k nearest neighbors
        for j, _ in similarities[:k_neighbors]:
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Remaining mask
    remaining_mask = torch.tensor(
        [i not in state['placed_indices'] for i in range(n_trees)],
        dtype=torch.bool
    )
    
    return Data(x=x, edge_index=edge_index, remaining_mask=remaining_mask)
```

---

# Bölüm 5: Curriculum Learning

## 5.1 Curriculum Learning Nedir?

İnsan öğrenmesi gibi: Kolay problemlerden zor problemlere geçiş.

```
Kolay → Orta → Zor

3 ağaç → 10 ağaç → 50 ağaç → 200 ağaç
```

## 5.2 Curriculum Stratejileri

### 5.2.1 Automatic Curriculum

```python
class AutomaticCurriculum:
    """
    Başarı oranına göre otomatik zorluk ayarlama
    """
    
    def __init__(
        self,
        initial_difficulty=0.1,
        target_success_rate=0.7,
        adjustment_rate=0.1,
        min_difficulty=0.0,
        max_difficulty=1.0
    ):
        self.difficulty = initial_difficulty
        self.target_success_rate = target_success_rate
        self.adjustment_rate = adjustment_rate
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        
        self.recent_successes = deque(maxlen=100)
    
    def get_task_params(self):
        """
        Mevcut zorluk seviyesine göre task parametreleri
        """
        # Linear interpolation
        n_trees = int(3 + self.difficulty * 197)  # 3 to 200
        
        # Rotation complexity
        if self.difficulty < 0.3:
            rotations = [0, 90, 180, 270]  # Only 90-degree
        elif self.difficulty < 0.6:
            rotations = list(range(0, 360, 45))  # 45-degree steps
        else:
            rotations = 'continuous'
        
        # Box size constraint
        if self.difficulty < 0.5:
            box_slack = 1.5  # 50% larger than minimum
        else:
            box_slack = 1.0 + (1.0 - self.difficulty)  # Tighter constraint
        
        return {
            'n_trees': n_trees,
            'allowed_rotations': rotations,
            'box_slack': box_slack
        }
    
    def record_result(self, success):
        """
        Episode sonucunu kaydet ve zorluğu ayarla
        """
        self.recent_successes.append(float(success))
        
        if len(self.recent_successes) >= 20:
            success_rate = np.mean(self.recent_successes)
            
            # Adjust difficulty
            if success_rate > self.target_success_rate + 0.1:
                # Too easy, increase difficulty
                self.difficulty = min(
                    self.max_difficulty,
                    self.difficulty + self.adjustment_rate
                )
            elif success_rate < self.target_success_rate - 0.1:
                # Too hard, decrease difficulty
                self.difficulty = max(
                    self.min_difficulty,
                    self.difficulty - self.adjustment_rate
                )
    
    def get_stats(self):
        return {
            'difficulty': self.difficulty,
            'success_rate': np.mean(self.recent_successes) if self.recent_successes else 0
        }
```

### 5.2.2 Staged Curriculum

```python
class StagedCurriculum:
    """
    Manuel olarak tanımlanmış aşamalar
    """
    
    def __init__(self):
        self.stages = [
            # Stage 0: Very easy
            {
                'name': 'Beginner',
                'n_trees': 3,
                'rotations': [0],
                'success_threshold': 0.9,
                'min_episodes': 100
            },
            # Stage 1: Easy
            {
                'name': 'Easy',
                'n_trees': 5,
                'rotations': [0, 90],
                'success_threshold': 0.85,
                'min_episodes': 200
            },
            # Stage 2: Medium-Easy
            {
                'name': 'Medium-Easy',
                'n_trees': 10,
                'rotations': [0, 45, 90, 135, 180, 225, 270, 315],
                'success_threshold': 0.8,
                'min_episodes': 500
            },
            # Stage 3: Medium
            {
                'name': 'Medium',
                'n_trees': 25,
                'rotations': 'continuous',
                'success_threshold': 0.75,
                'min_episodes': 1000
            },
            # Stage 4: Medium-Hard
            {
                'name': 'Medium-Hard',
                'n_trees': 50,
                'rotations': 'continuous',
                'success_threshold': 0.7,
                'min_episodes': 2000
            },
            # Stage 5: Hard
            {
                'name': 'Hard',
                'n_trees': 100,
                'rotations': 'continuous',
                'success_threshold': 0.65,
                'min_episodes': 5000
            },
            # Stage 6: Very Hard
            {
                'name': 'Very Hard',
                'n_trees': 150,
                'rotations': 'continuous',
                'success_threshold': 0.6,
                'min_episodes': 10000
            },
            # Stage 7: Expert
            {
                'name': 'Expert',
                'n_trees': 200,
                'rotations': 'continuous',
                'success_threshold': 0.5,
                'min_episodes': None  # Final stage
            }
        ]
        
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_successes = 0
    
    def get_current_config(self):
        return self.stages[self.current_stage]
    
    def record_episode(self, success):
        self.stage_episodes += 1
        if success:
            self.stage_successes += 1
        
        # Check for stage advancement
        config = self.stages[self.current_stage]
        
        if config['min_episodes'] is None:
            return  # Final stage
        
        if self.stage_episodes >= config['min_episodes']:
            success_rate = self.stage_successes / self.stage_episodes
            
            if success_rate >= config['success_threshold']:
                self._advance_stage()
    
    def _advance_stage(self):
        if self.current_stage < len(self.stages) - 1:
            print(f"\n🎉 Stage {self.current_stage} completed!")
            print(f"   Success rate: {self.stage_successes/self.stage_episodes:.1%}")
            print(f"   Advancing to: {self.stages[self.current_stage + 1]['name']}\n")
            
            self.current_stage += 1
            self.stage_episodes = 0
            self.stage_successes = 0
```

### 5.2.3 Self-Paced Learning

```python
class SelfPacedCurriculum:
    """
    Agent kendi öğrenme hızına göre örnek seçer
    """
    
    def __init__(self, task_pool, initial_weight=1.0, growth_rate=1.1):
        """
        task_pool: Farklı zorluk seviyelerinde task'ların listesi
        """
        self.task_pool = sorted(task_pool, key=lambda t: t['difficulty'])
        self.weights = [initial_weight] * len(task_pool)
        self.growth_rate = growth_rate
        
        self.task_history = {i: [] for i in range(len(task_pool))}
    
    def sample_task(self):
        """
        Ağırlıklı örnekleme ile task seç
        """
        # Normalize weights to probabilities
        total = sum(self.weights)
        probs = [w / total for w in self.weights]
        
        # Sample
        task_idx = np.random.choice(len(self.task_pool), p=probs)
        return task_idx, self.task_pool[task_idx]
    
    def update(self, task_idx, reward, loss):
        """
        Task sonucuna göre ağırlıkları güncelle
        """
        self.task_history[task_idx].append({
            'reward': reward,
            'loss': loss
        })
        
        # Calculate task "readiness" - am I ready for this task?
        history = self.task_history[task_idx]
        
        if len(history) >= 10:
            recent_rewards = [h['reward'] for h in history[-10:]]
            avg_reward = np.mean(recent_rewards)
            
            # If doing well on this task, increase weight of harder tasks
            if avg_reward > 0:
                for i in range(task_idx + 1, len(self.task_pool)):
                    self.weights[i] *= self.growth_rate
            
            # If struggling, decrease weight of this task
            else:
                self.weights[task_idx] *= 0.9
        
        # Normalize weights
        max_weight = max(self.weights)
        self.weights = [w / max_weight for w in self.weights]
```

---

# Bölüm 6: Training Pipeline

## 6.1 Complete Training Loop

```python
def train_packing_agent(
    agent,
    env_fn,
    curriculum,
    n_episodes=100000,
    eval_freq=1000,
    save_freq=5000,
    log_dir='logs',
    checkpoint_dir='checkpoints'
):
    """
    Complete training pipeline
    """
    from torch.utils.tensorboard import SummaryWriter
    import os
    
    # Setup
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    best_eval_score = float('inf')
    
    for episode in range(n_episodes):
        # Get curriculum config
        config = curriculum.get_current_config()
        
        # Create environment
        env = env_fn(config)
        
        # Run episode
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            if hasattr(agent, 'collect_rollout'):
                # PPO-style: collect rollout then update
                rollout = agent.collect_rollout(env, n_steps=128)
                loss_info = agent.update(rollout)
                
                episode_reward += sum(rollout['rewards'])
                episode_length += len(rollout['rewards'])
                
                if rollout['dones'][-1]:
                    break
            else:
                # DQN-style: step by step
                action = agent.select_action(state, env.get_valid_actions())
                next_state, reward, done, info = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
        
        # Record results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        final_bbox = env.compute_final_bbox()
        success = final_bbox < config.get('target_bbox', float('inf'))
        curriculum.record_episode(success)
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            writer.add_scalar('training/episode_reward', episode_reward, episode)
            writer.add_scalar('training/avg_reward_100', avg_reward, episode)
            writer.add_scalar('training/episode_length', episode_length, episode)
            writer.add_scalar('training/final_bbox', final_bbox, episode)
            writer.add_scalar('curriculum/difficulty', curriculum.difficulty, episode)
            
            if hasattr(agent, 'epsilon'):
                writer.add_scalar('training/epsilon', agent.epsilon, episode)
            
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | "
                  f"BBox: {final_bbox:.2f} | Difficulty: {curriculum.difficulty:.2f}")
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_score = evaluate_agent(agent, env_fn, n_episodes=20)
            writer.add_scalar('eval/mean_bbox', eval_score, episode)
            
            if eval_score < best_eval_score:
                best_eval_score = eval_score
                agent.save(f"{checkpoint_dir}/best_model.pt")
                print(f"New best model! Eval BBox: {eval_score:.2f}")
        
        # Checkpointing
        if episode % save_freq == 0:
            agent.save(f"{checkpoint_dir}/checkpoint_{episode}.pt")
    
    writer.close()
    return agent


def evaluate_agent(agent, env_fn, n_episodes=20, deterministic=True):
    """
    Agent'ı değerlendir
    """
    bbox_scores = []
    
    for _ in range(n_episodes):
        env = env_fn({'n_trees': 50})  # Fixed evaluation difficulty
        state = env.reset()
        
        while True:
            with torch.no_grad():
                action = agent.select_action(state, env.get_valid_actions(), 
                                            deterministic=deterministic)
            state, reward, done, info = env.step(action)
            
            if done:
                break
        
        bbox_scores.append(env.compute_final_bbox())
    
    return np.mean(bbox_scores)
```

## 6.2 Hyperparameter Tuning

```python
def hyperparameter_search(env_fn, n_trials=50):
    """
    Optuna ile hyperparameter search
    """
    import optuna
    
    def objective(trial):
        # Sample hyperparameters
        params = {
            'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-3),
            'gamma': trial.suggest_uniform('gamma', 0.95, 0.999),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
            'epsilon_decay': trial.suggest_uniform('epsilon_decay', 0.99, 0.9999),
        }
        
        # Create agent
        agent = DQNAgent(
            state_dim=...,
            action_dim=...,
            hidden_dims=[params['hidden_dim']] * params['n_layers'],
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            batch_size=params['batch_size'],
            buffer_size=params['buffer_size'],
            epsilon_decay=params['epsilon_decay']
        )
        
        # Short training
        curriculum = AutomaticCurriculum()
        train_packing_agent(agent, env_fn, curriculum, n_episodes=5000)
        
        # Evaluate
        eval_score = evaluate_agent(agent, env_fn)
        
        return eval_score  # Minimize bbox size
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best params: {study.best_params}")
    print(f"Best score: {study.best_value}")
    
    return study.best_params
```

---

# Bölüm 7: Özet ve Best Practices

## 7.1 Algoritma Seçimi

| Problem Özelliği | Önerilen Algoritma |
|------------------|-------------------|
| Discrete actions, küçük state space | Tabular Q-Learning |
| Discrete actions, büyük state space | DQN, Rainbow |
| Continuous actions | PPO, SAC |
| Değişken boyutlu input | GNN + RL |
| Sample efficiency önemli | PPO, SAC |
| Exploration önemli | Noisy DQN, SAC |

## 7.2 Best Practices

1. **Reward Shaping**
   - Dense + Sparse reward kullan
   - Normalize et
   - Potential-based shaping ile bias'tan kaçın

2. **Curriculum Learning**
   - Basit problemlerden başla
   - Automatic difficulty adjustment kullan
   - Success rate'i %70-80 civarında tut

3. **Network Architecture**
   - LayerNorm kullan
   - Residual connections ekle
   - Separate value/policy heads (A2C, PPO)

4. **Training Stability**
   - Gradient clipping
   - Learning rate scheduling
   - Target network (DQN)
   - PPO clipping

5. **Debugging**
   - TensorBoard ile her şeyi logla
   - Küçük problemlerde test et
   - Reward'ı kontrol et (doğru mu?)
   - Action distribution'ı izle

## 7.3 Tree Packing için Özel Öneriler

1. **State Representation**
   - Grid + tree features kombinasyonu
   - GNN için graph yapısı

2. **Action Space**
   - Hierarchical: önce tree seç, sonra pozisyon
   - Position için NFP-guided sampling

3. **Reward**
   - Delta-based bbox reward
   - Efficiency bonus
   - Future potential consideration

4. **Curriculum**
   - 3 → 10 → 25 → 50 → 100 → 200 trees
   - Discrete → continuous rotation
   - Slack → tight box constraint

---

**İyi çalışmalar!** 🚀
