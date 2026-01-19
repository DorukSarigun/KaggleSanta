"""
Relative-Only Transformer Training for Tree Packing.

Key Differences from transformer_training.py:
1. Observation: Only RELATIVE features (no absolute coordinates)
2. Action Space: RELATIVE (dx, dy from last tree + rotation)
3. CNN provides spatial context via occupancy grid

Philosophy:
- Agent learns to place trees RELATIVE to existing placements
- Generalizes better across different starting positions
- CNN (512x512 grid) provides absolute spatial awareness
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import argparse

from environment import TreePackingEnvironment, scale_factor


class RelativeTransformerEnvironment(TreePackingEnvironment):
    """
    Environment with RELATIVE observations and actions.
    
    Observation:
    - relative_features: (n_trees, 6) per tree
    - collision_features: (6,) risk metrics
    - grid: (512, 512) CNN input
    - scalars: (5,) global metrics
    
    Action:
    - dx, dy: relative to last placed tree (or center if first)
    - rotation: absolute [0, 360]
    """
    
    MAX_TREES = 100
    TREE_DIM = 4  # Only relative features: dx, dy, dist, angle
    
    def __init__(self, n_trees: int = 50, max_trees: int = 200):
        super().__init__(n_trees=n_trees, max_trees=max_trees)
        
        from gymnasium import spaces
        
        # RELATIVE action space: dx, dy from last tree + rotation
        # dx, dy in [-5, 5] (relative offset)
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 360.0], dtype=np.float32)
        )
        
        self.observation_space = spaces.Dict({
            "relative_trees": spaces.Box(-np.inf, np.inf, shape=(self.MAX_TREES, self.TREE_DIM), dtype=np.float32),
            "mask": spaces.Box(0, 1, shape=(self.MAX_TREES,), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "collision_features": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "grid": spaces.Box(0, 1, shape=(512, 512), dtype=np.float32),
            "anchor": spaces.Box(0, 10, shape=(2,), dtype=np.float32)  # Last tree position (for action interpretation)
        })
    
    def _get_relative_tree_features(self, tree, anchor_x: float, anchor_y: float) -> np.ndarray:
        """
        Get RELATIVE features for a tree.
        
        Features (4):
        - dx, dy from anchor: 2
        - distance from anchor: 1
        - angle from anchor: 1
        """
        norm = float(scale_factor)
        
        cx = float(tree.center_x) / norm
        cy = float(tree.center_y) / norm
        anchor_x_norm = anchor_x / norm
        anchor_y_norm = anchor_y / norm
        
        dx = cx - anchor_x_norm
        dy = cy - anchor_y_norm
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)  # [-π, π]
        
        return np.array([dx, dy, distance, angle], dtype=np.float32)
    
    def _get_observation(self) -> Dict:
        """Get relative observation."""
        n = len(self.placed_trees)
        norm = float(scale_factor)
        
        # Anchor: last tree position, or center if no trees
        if n > 0:
            last_tree = self.placed_trees[-1]
            anchor_x = float(last_tree.center_x)
            anchor_y = float(last_tree.center_y)
        else:
            anchor_x = 5.0 * norm  # Center
            anchor_y = 5.0 * norm
        
        # Relative tree features
        relative_trees = np.zeros((self.MAX_TREES, self.TREE_DIM), dtype=np.float32)
        mask = np.zeros(self.MAX_TREES, dtype=np.float32)
        
        for i, tree in enumerate(self.placed_trees):
            if i >= self.MAX_TREES:
                break
            relative_trees[i] = self._get_relative_tree_features(tree, anchor_x, anchor_y)
            mask[i] = 1.0
        
        # Scalars
        scalars = np.array([
            n / self.max_trees,
            self.bounding_box_size / (10.0 * norm) if n > 0 else 0,
            self.fill_ratio,
            (self.n_trees - self.current_index) / self.n_trees,
            self.total_tree_area / (norm * norm * 100)
        ], dtype=np.float32)
        
        # Collision features
        collision_features = self._get_collision_features()
        
        # Grid
        grid = self._get_occupancy_grid(size=512)
        
        # Anchor position (normalized)
        anchor = np.array([anchor_x / norm, anchor_y / norm], dtype=np.float32)
        
        return {
            "relative_trees": relative_trees,
            "mask": mask,
            "scalars": scalars,
            "collision_features": collision_features,
            "grid": grid,
            "anchor": anchor
        }
    
    def get_observation(self):
        return self._get_observation()
    
    def step(self, action: np.ndarray):
        """
        Step with RELATIVE action.
        
        Action: [dx, dy, rotation]
        - dx, dy: offset from anchor (last tree or center)
        - rotation: absolute angle
        """
        dx, dy, rotation = action[0], action[1], action[2]
        
        # Get anchor position
        if len(self.placed_trees) > 0:
            last_tree = self.placed_trees[-1]
            anchor_x = float(last_tree.center_x)
            anchor_y = float(last_tree.center_y)
        else:
            norm = float(scale_factor)
            anchor_x = 5.0 * norm  # Center
            anchor_y = 5.0 * norm
        
        # Convert relative to absolute
        norm = float(scale_factor)
        x = (anchor_x / norm) + dx
        y = (anchor_y / norm) + dy
        
        # Clip to valid range
        x = np.clip(x, 0.5, 9.5)
        y = np.clip(y, 0.5, 9.5)
        rotation = np.clip(rotation, 0.0, 360.0)
        
        # Call parent step with absolute coordinates
        return super().step(np.array([x, y, rotation]))


class RelativeRolloutBuffer:
    """Buffer for relative observations."""
    
    def __init__(self):
        self.relative_trees = []
        self.masks = []
        self.scalars = []
        self.collision_features = []
        self.grids = []
        self.anchors = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.collision_labels = []
    
    def add(self, obs: Dict, action, reward, done, log_prob, value, collision_label):
        self.relative_trees.append(obs["relative_trees"].copy())
        self.masks.append(obs["mask"].copy())
        self.scalars.append(obs["scalars"].copy())
        self.collision_features.append(obs["collision_features"].copy())
        self.grids.append(obs["grid"].copy())
        self.anchors.append(obs["anchor"].copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.collision_labels.append(collision_label)
    
    def clear(self):
        self.__init__()
    
    def get_batches(self, batch_size: int, device: str):
        n = len(self.actions)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            
            yield {
                "relative_trees": torch.tensor(np.array([self.relative_trees[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "masks": torch.tensor(np.array([self.masks[i] for i in batch_idx]), dtype=torch.bool, device=device),
                "scalars": torch.tensor(np.array([self.scalars[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "collision_features": torch.tensor(np.array([self.collision_features[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "grids": torch.tensor(np.array([self.grids[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "anchors": torch.tensor(np.array([self.anchors[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "actions": torch.tensor(np.array([self.actions[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "log_probs": torch.tensor(np.array([self.log_probs[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "collision_labels": torch.tensor(np.array([self.collision_labels[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "indices": batch_idx
            }


class RelativeTreeEncoder(nn.Module):
    """
    Encode relative tree features.
    Input: (batch, n_trees, 6)
    Output: (batch, n_trees, embed_dim)
    """
    
    def __init__(self, input_dim: int = 6, embed_dim: int = 64, n_freqs: int = 8):
        super().__init__()
        
        # Fourier encoding for relative features
        fourier_dim = input_dim * 2 * n_freqs  # 6 * 2 * 8 = 96
        
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer('freqs', freqs * np.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_trees, _ = x.shape
        
        # Fourier encoding
        x_expanded = x.unsqueeze(-1) * self.freqs
        fourier = torch.stack([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
        fourier = fourier.reshape(batch_size, n_trees, -1)
        
        return self.projection(fourier)


class AttentionFusion(nn.Module):
    """
    Cross-attention based fusion.
    Query: scalar + grid context
    Key/Value: tree sequence
    """
    
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Project scalar + grid + collision to query
        self.query_proj = nn.Linear(embed_dim * 3, embed_dim)
        
        # Multi-head cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        tree_seq: torch.Tensor,       # (batch, n_trees, embed_dim)
        scalar_embed: torch.Tensor,   # (batch, embed_dim)
        collision_embed: torch.Tensor,# (batch, embed_dim)
        grid_embed: torch.Tensor,     # (batch, embed_dim)
        mask: torch.Tensor = None     # (batch, n_trees)
    ) -> torch.Tensor:
        batch_size = tree_seq.shape[0]
        
        # Sanitize inputs
        scalar_embed = torch.nan_to_num(scalar_embed, nan=0.0)
        collision_embed = torch.nan_to_num(collision_embed, nan=0.0)
        grid_embed = torch.nan_to_num(grid_embed, nan=0.0)
        tree_seq = torch.nan_to_num(tree_seq, nan=0.0)
        
        # Create query from context (scalar + collision + grid)
        context = torch.cat([scalar_embed, collision_embed, grid_embed], dim=-1)
        query = self.query_proj(context).unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Handle empty sequence or all masked
        if mask is not None:
            valid_count = mask.sum(dim=1)
            has_valid = valid_count > 0
            
            if not has_valid.any():
                # No valid trees at all - use context only
                out = self.norm1(query)
                ffn_out = self.ffn(out)
                out = self.norm2(out + ffn_out)
                return out.squeeze(1)
            
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=query,
            key=tree_seq,
            value=tree_seq,
            key_padding_mask=key_padding_mask
        )
        
        # Robust NaN handling
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Residual + LayerNorm
        attn_out = self.norm1(query + attn_out)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        
        # FFN + Residual
        ffn_out = self.ffn(attn_out)
        ffn_out = torch.nan_to_num(ffn_out, nan=0.0)
        out = self.norm2(attn_out + ffn_out)
        out = torch.nan_to_num(out, nan=0.0)
        
        return out.squeeze(1)  # (batch, embed_dim)


class RelativeActorCritic(nn.Module):
    """
    Actor-Critic for relative action space.
    
    Simpler architecture since we only have relative features.
    CNN provides spatial context.
    """
    
    def __init__(
        self,
        tree_dim: int = 4,  # Only 4 relative features now
        embed_dim: int = 128,  # Increased from 64
        n_heads: int = 8,      # Increased from 4
        n_layers: int = 6,     # Increased from 4
        action_dim: int = 3
    ):
        super().__init__()
        
        # Action bounds (relative)
        action_low = [-5.0, -5.0, 0.0]
        action_high = [5.0, 5.0, 360.0]
        
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2)
        
        # Relative tree encoder
        self.tree_encoder = RelativeTreeEncoder(tree_dim, embed_dim)
        
        # Transformer for tree sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Scalar encoder
        self.scalar_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
        # Collision encoder
        self.collision_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
        # Grid encoder (CNN)
        self.grid_encoder = self._build_grid_encoder(embed_dim)
        
        # Attention-based fusion
        self.attention_fusion = AttentionFusion(embed_dim, n_heads)
        
        # State dimension: fusion output only (cross-attention output)
        state_dim = embed_dim
        
        # Actor
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Collision predictor
        self.collision_predictor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _build_grid_encoder(self, embed_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )
    
    def forward(self, relative_trees, scalars, mask, collision_features, grid):
        batch_size = relative_trees.shape[0]
        
        # Encode trees
        tree_embed = self.tree_encoder(relative_trees)  # (batch, n_trees, embed_dim)
        
        # Transformer
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None
        
        tree_seq = self.transformer(tree_embed, src_key_padding_mask=attn_mask)
        
        # Pool trees (mean of valid)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            tree_pool = (tree_seq * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            tree_pool = tree_seq.mean(dim=1)
        
        # Encode other features
        scalar_embed = self.scalar_encoder(scalars)
        collision_embed = self.collision_encoder(collision_features)
        
        # Grid
        grid_input = grid.unsqueeze(1)  # (batch, 1, 512, 512)
        grid_embed = self.grid_encoder(grid_input)
        
        # ATTENTION FUSION (cross-attention)
        state = self.attention_fusion(
            tree_seq, scalar_embed, collision_embed, grid_embed, mask
        )
        state = torch.nan_to_num(state, nan=0.0)
        
        # Actor
        mean = self.actor_mean(state)
        log_std = self.actor_log_std(state)
        log_std = torch.clamp(log_std, -5, 0)
        
        # Critic
        value = self.critic(state)
        
        # Collision predictor
        collision_logit = self.collision_predictor(state)
        
        return mean, log_std, value, collision_logit
    
    def get_action(self, relative_trees, scalars, mask, collision_features, grid, deterministic=False):
        mean, log_std, value, collision_logit = self.forward(
            relative_trees, scalars, mask, collision_features, grid
        )
        
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-6)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action_raw = dist.rsample()
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            action = torch.tanh(action_raw)
        
        action_scaled = action * self.action_scale + self.action_bias
        
        return action_scaled, log_prob, value, collision_logit
    
    def evaluate_actions(self, relative_trees, scalars, actions, mask, collision_features, grid):
        mean, log_std, value, collision_logit = self.forward(
            relative_trees, scalars, mask, collision_features, grid
        )
        
        log_std = torch.nan_to_num(log_std, nan=-2.0)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-6)
        
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        actions_raw = torch.atanh(torch.clamp(actions_unscaled, -0.999, 0.999))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy, collision_logit


class RelativeTrainer:
    """PPO Trainer for Relative policy."""
    
    def __init__(
        self,
        n_trees: int = 50,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 32,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        visualize: bool = True
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.env = RelativeTransformerEnvironment(n_trees=n_trees)
        self.n_trees = n_trees
        
        self.network = RelativeActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = RelativeRolloutBuffer()
        
        self.visualize = visualize
        if visualize:
            self.output_dir = Path("relative_training_outputs")
            self.output_dir.mkdir(exist_ok=True)
            self.rewards_history = []
            self.bbox_history = []
    
    def obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        return {
            "relative_trees": torch.tensor(obs["relative_trees"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "mask": torch.tensor(obs["mask"], dtype=torch.bool, device=self.device).unsqueeze(0),
            "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "collision_features": torch.tensor(obs["collision_features"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "grid": torch.tensor(obs["grid"], dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def collect_rollouts(self, n_steps: int):
        obs, _ = self.env.reset()
        episode_reward = 0.0
        
        for step in range(n_steps):
            obs_t = self.obs_to_tensor(obs)
            
            with torch.no_grad():
                action, log_prob, value, _ = self.network.get_action(
                    obs_t["relative_trees"], obs_t["scalars"], obs_t["mask"],
                    obs_t["collision_features"], obs_t["grid"]
                )
            
            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            episode_reward += reward
            done = terminated or truncated
            
            # Collision label
            collision_label = 1.0 if info.get("collision", False) else 0.0
            
            self.buffer.add(
                obs, action_np, reward, done,
                log_prob.item(), value.item(), collision_label
            )
            
            if done:
                # Store trees for visualization BEFORE reset
                self.last_placed_trees = list(self.env.placed_trees)
                obs, _ = self.env.reset()
                result = (episode_reward, info.get("bbox_size", 0), info.get("n_placed", 0))
                episode_reward = 0.0
                return result
            
            obs = next_obs
        
        return (episode_reward, 0, 0)
    
    def compute_gae(self):
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return returns, advantages
    
    def update(self, returns, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, self.device):
                batch_returns = torch.tensor(returns[batch["indices"]], dtype=torch.float32, device=self.device)
                batch_advantages = torch.tensor(advantages[batch["indices"]], dtype=torch.float32, device=self.device)
                
                log_probs, values, entropy, collision_logits = self.network.evaluate_actions(
                    batch["relative_trees"], batch["scalars"], batch["actions"],
                    batch["masks"], batch["collision_features"], batch["grids"]
                )
                
                ratio = torch.exp(log_probs - batch["log_probs"])
                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                value_loss = nn.functional.mse_loss(values.squeeze(-1), batch_returns)
                entropy_loss = -entropy.mean()
                
                # Collision loss
                collision_labels_safe = torch.clamp(batch["collision_labels"], 0.0, 1.0)
                collision_loss = nn.functional.binary_cross_entropy_with_logits(
                    collision_logits.squeeze(-1), collision_labels_safe
                )
                
                loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss + 0.1 * collision_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save_visualization(self, episode: int):
        """Save visualization of current episode."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        norm = float(scale_factor)
        
        # Use saved trees from last episode (before reset)
        trees = getattr(self, 'last_placed_trees', [])
        
        # Plot 1: Tree placements
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title(f'Episode {episode} - Trees: {len(trees)}')
        
        for tree in trees:
            coords = np.array(tree.polygon.exterior.coords)
            coords = coords / norm
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.5, edgecolor='black')
        
        # Draw bounding box
        if trees:
            from shapely.ops import unary_union
            all_polys = [t.polygon for t in trees]
            bounds = unary_union(all_polys).bounds
            minx, miny, maxx, maxy = [b / norm for b in bounds]
            size = max(maxx - minx, maxy - miny)
            ax.add_patch(plt.Rectangle((minx, miny), size, size, 
                         fill=False, edgecolor='red', linewidth=2))
        
        # Plot 2: Reward history
        ax = axes[1]
        if len(self.rewards_history) > 0:
            ax.plot(self.rewards_history[-100:])
            ax.set_title(f'Reward (last 100) - Avg: {np.mean(self.rewards_history[-100:]):.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        
        # Plot 3: BBox history
        ax = axes[2]
        if len(self.bbox_history) > 0:
            ax.plot(self.bbox_history[-100:])
            ax.set_title(f'BBox (last 100) - Avg: {np.mean(self.bbox_history[-100:]):.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('BBox Size')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'episode_{episode:05d}.png', dpi=100)
        plt.close()
    
    def train(self, n_episodes: int):
        print(f"Relative Training: {n_episodes} episodes, {self.n_trees} trees each")
        print("-" * 50)
        
        for episode in range(n_episodes):
            result = self.collect_rollouts(n_steps=self.n_trees)
            ep_reward, bbox_size, n_placed = result
            
            returns, advantages = self.compute_gae()
            self.update(returns, advantages)
            self.buffer.clear()
            
            if self.visualize:
                self.rewards_history.append(ep_reward)
                self.bbox_history.append(bbox_size)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1:5d} | Reward: {ep_reward:8.2f} | BBox: {bbox_size:.4f} | Trees: {n_placed}")
            
            # Save visualization every 50 episodes
            if self.visualize and (episode + 1) % 50 == 0:
                self.save_visualization(episode + 1)
        
        print("-" * 50)
        print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trees", type=int, default=50)
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    trainer = RelativeTrainer(
        n_trees=args.n_trees,
        lr=args.lr,
        visualize=True
    )
    trainer.train(n_episodes=args.n_episodes)
