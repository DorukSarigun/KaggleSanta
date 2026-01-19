# ==================== RELATIVE ACTION NETWORK ====================
"""
Relative action space: (distance, direction, rotation) from last tree
Pairwise observations: features for tree pairs
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List


class PairwiseEncoder(nn.Module):
    """
    Encodes pairwise features between trees using simple MLP + pooling.
    (Attention'ı kaldırdık - NaN sorunları vardı)
    """
    
    def __init__(self, pair_dim: int = 8, embed_dim: int = 128):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Project and encode each pair independently
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # Post-pooling projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, pair_features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pair_features: (batch, n_pairs, pair_dim)
            mask: (batch, n_pairs) - True for valid pairs
        Returns:
            (batch, embed_dim)
        """
        batch_size = pair_features.shape[0]
        
        # Check if any valid pairs exist
        if mask is not None:
            n_valid = mask.sum(dim=1)  # (batch,)
            if (n_valid == 0).all():
                # No valid pairs - return zeros
                return torch.zeros(batch_size, self.embed_dim, device=pair_features.device)
        
        # Encode each pair
        x = self.pair_encoder(pair_features)  # (batch, n_pairs, embed_dim)
        
        # Mean pooling over valid pairs
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, n_pairs, 1)
            sum_x = (x * mask_expanded).sum(dim=1)  # (batch, embed_dim)
            count = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
            x = sum_x / count
        else:
            x = x.mean(dim=1)
        
        # Handle any NaN
        x = torch.nan_to_num(x, nan=0.0)
        
        return self.output_proj(x)


class LastTreeEncoder(nn.Module):
    """
    Encodes the last placed tree's features.
    """
    
    def __init__(self, feature_dim: int = 6, embed_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, last_tree: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_tree: (batch, feature_dim) - last tree features
        Returns:
            (batch, embed_dim)
        """
        return self.encoder(last_tree)


class RelativeActor(nn.Module):
    """
    Actor for relative actions: (distance, direction, rotation).
    """
    
    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()
        
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.log_std_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, -5, 0)
        return mean, log_std


class RelativeCritic(nn.Module):
    """
    Critic for value estimation.
    """
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_net(state)


class RelativeActorCritic(nn.Module):
    """
    Actor-Critic with relative action space.
    
    Action space:
        - distance: [0.5, 3.0] - distance from last tree
        - direction: [-180, 180] - angle from last tree (degrees)
        - rotation: [0, 360] - tree's own rotation
    
    Observation:
        - pairwise_features: (n_pairs, 8) - features for tree pairs
        - last_tree: (6,) - last tree position/angle
        - scalars: (5,) - global metrics
    """
    
    def __init__(
        self,
        max_pairs: int = 100,
        pair_dim: int = 8,
        action_dim: int = 3
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.max_pairs = max_pairs
        
        # Action bounds (relative)
        # distance: [0.5, 3.0], direction: [-180, 180], rotation: [0, 360]
        action_low = np.array([0.0, -180.0, 0.0])
        action_high = np.array([3.0, 180.0, 360.0])
        
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2)
        
        # Encoders
        self.pairwise_encoder = PairwiseEncoder(pair_dim=pair_dim, embed_dim=128)
        self.last_tree_encoder = LastTreeEncoder(feature_dim=6, embed_dim=64)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU()
        )
        
        # Combined state dim
        state_dim = 128 + 64 + 32  # pairwise + last_tree + scalars = 224
        
        # Actor and Critic
        self.actor = RelativeActor(state_dim, action_dim)
        self.critic = RelativeCritic(state_dim)
    
    def encode_state(
        self,
        pairwise_features: torch.Tensor,
        last_tree: torch.Tensor,
        scalars: torch.Tensor,
        pair_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode all observations into single state vector."""
        
        pair_embed = self.pairwise_encoder(pairwise_features, pair_mask)
        last_embed = self.last_tree_encoder(last_tree)
        scalar_embed = self.scalar_encoder(scalars)
        
        return torch.cat([pair_embed, last_embed, scalar_embed], dim=-1)
    
    def forward(
        self,
        pairwise_features: torch.Tensor,
        last_tree: torch.Tensor,
        scalars: torch.Tensor,
        pair_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: mean, log_std, value
        """
        state = self.encode_state(pairwise_features, last_tree, scalars, pair_mask)
        mean, log_std = self.actor(state)
        value = self.critic(state)
        return mean, log_std, value
    
    def get_action(
        self,
        pairwise_features: torch.Tensor,
        last_tree: torch.Tensor,
        scalars: torch.Tensor,
        pair_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        
        mean, log_std, value = self.forward(pairwise_features, last_tree, scalars, pair_mask)
        
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.rsample()
        
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        
        # Scale to action bounds
        action = torch.tanh(action_raw)
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        pairwise_features: torch.Tensor,
        last_tree: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        pair_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        
        mean, log_std, value = self.forward(pairwise_features, last_tree, scalars, pair_mask)
        std = torch.exp(log_std)
        
        # Unscale actions
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        actions_raw = torch.atanh(torch.clamp(actions_unscaled, -0.999, 0.999))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy
