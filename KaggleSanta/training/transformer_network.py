# ==================== TRANSFORMER POLICY NETWORK ====================
"""
Transformer-based policy with attention over all placed trees.
No gradient vanishing - every tree has direct access to all others.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for tree sequence."""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TreeEncoder(nn.Module):
    """Encode individual tree features to embedding."""
    
    def __init__(self, tree_dim: int = 6, embed_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(tree_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    
    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tree_features: (batch, n_trees, tree_dim)
        Returns:
            (batch, n_trees, embed_dim)
        """
        return self.encoder(tree_features)


class TransformerActorCritic(nn.Module):
    """
    Transformer-based Actor-Critic for tree placement.
    
    Uses self-attention over all placed trees - no gradient vanishing!
    Every tree can directly attend to every other tree.
    """
    
    def __init__(
        self,
        tree_dim: int = 25,          # 5 center + 20 corner features
        scalar_dim: int = 5,         # Global scalar features
        embed_dim: int = 128,        # Transformer embedding dim
        n_heads: int = 4,            # Number of attention heads
        n_layers: int = 3,           # Number of transformer layers
        max_trees: int = 200,        # Maximum trees to handle
        action_dim: int = 3          # (distance, direction, rotation)
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_trees = max_trees
        self.action_dim = action_dim
        
        # Action bounds (relative)
        action_low = np.array([0.0, -180.0, 0.0])
        action_high = np.array([3.0, 180.0, 360.0])
        
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2)
        
        # Tree encoder
        self.tree_encoder = TreeEncoder(tree_dim, embed_dim)
        
        # Positional encoding (tree order matters!)
        self.pos_encoder = PositionalEncoding(embed_dim, max_trees)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Scalar encoder (global features)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # Combined state dim
        state_dim = embed_dim * 2  # pooled trees + scalars
        
        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def encode_trees(
        self,
        tree_features: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encode sequence of trees using transformer.
        
        Args:
            tree_features: (batch, n_trees, tree_dim)
            mask: (batch, n_trees) - True for valid trees
        Returns:
            (batch, embed_dim) - pooled representation
        """
        batch_size = tree_features.shape[0]
        n_trees = tree_features.shape[1]
        
        # Check for empty input
        if mask is not None:
            valid_count = mask.sum(dim=1)
            if (valid_count == 0).all():
                return torch.zeros(batch_size, self.embed_dim, device=tree_features.device)
        
        # Encode trees
        x = self.tree_encoder(tree_features)  # (batch, n_trees, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask (True = ignore)
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Mean pooling over valid trees
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            sum_x = (x * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_x / count
        else:
            pooled = x.mean(dim=1)
        
        # Handle NaN
        pooled = torch.nan_to_num(pooled, nan=0.0)
        
        return pooled
    
    def forward(
        self,
        tree_features: torch.Tensor,
        scalars: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            tree_features: (batch, n_trees, tree_dim)
            scalars: (batch, scalar_dim)
            mask: (batch, n_trees) - True for valid trees
        
        Returns:
            mean, log_std, value
        """
        # Encode trees
        tree_embed = self.encode_trees(tree_features, mask)  # (batch, embed_dim)
        
        # Encode scalars
        scalar_embed = self.scalar_encoder(scalars)  # (batch, embed_dim)
        
        # Combine
        state = torch.cat([tree_embed, scalar_embed], dim=-1)  # (batch, 2*embed_dim)
        
        # Actor
        mean = self.actor_mean(state)
        log_std = self.actor_log_std(state)
        log_std = torch.clamp(log_std, -5, 0)
        
        # Critic
        value = self.critic(state)
        
        return mean, log_std, value
    
    def get_action(
        self,
        tree_features: torch.Tensor,
        scalars: torch.Tensor,
        mask: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        
        mean, log_std, value = self.forward(tree_features, scalars, mask)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action_raw = dist.rsample()
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            action = torch.tanh(action_raw)
        
        # Scale to action bounds
        action_scaled = action * self.action_scale + self.action_bias
        
        return action_scaled, log_prob, value
    
    def evaluate_actions(
        self,
        tree_features: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        
        mean, log_std, value = self.forward(tree_features, scalars, mask)
        std = torch.exp(log_std)
        
        # Unscale actions
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        actions_raw = torch.atanh(torch.clamp(actions_unscaled, -0.999, 0.999))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy
