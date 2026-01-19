"""
Neural Network Architecture for Tree Packing RL Agent.

Coordinate-based approach with Transformer encoder for variable-length tree sequences.
Actor-Critic architecture for PPO/A2C training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
import math



class TreeFeatureEncoder(nn.Module):
    """
    Her ağacın özelliklerini (10 dim) embed eder.
    
    Input: (batch, max_trees, 10)
    Output: (batch, max_trees, embed_dim)
    """
    
    def __init__(self, input_dim: int = 10, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tree_features: (batch, max_trees, 10)
        Returns:
            (batch, max_trees, embed_dim)
        """
        return self.mlp(tree_features)


class PairwiseFeatureEncoder(nn.Module):
    """
    Pairwise rotated rectangle özelliklerini (10 dim) embed eder.
    
    Input: (batch, max_trees, 10)
    Output: (batch, max_trees, embed_dim)
    """
    
    def __init__(self, input_dim: int = 10, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, pairwise_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairwise_features: (batch, max_trees, 10)
        Returns:
            (batch, max_trees, embed_dim)
        """
        return self.mlp(pairwise_features)


class GridEncoder(nn.Module):
    """
    CNN encoder for 256x256 occupancy grid with fill ratios.
    Extracts spatial features from the grid.
    """
    
    def __init__(self, grid_size: int = 256, embed_dim: int = 32):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # (batch, 1, 256, 256) -> (batch, 16, 128, 128)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 16, 128, 128) -> (batch, 32, 64, 64)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 32, 64, 64) -> (batch, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 64, 32, 32) -> (batch, 64, 16, 16)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 64, 16, 16) -> (batch, 64, 8, 8)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 64, 8, 8) -> (batch, 64, 4, 4)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Flatten
            nn.Flatten(),  # (batch, 64 * 4 * 4) = (batch, 1024)
            nn.Linear(1024, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, grid_size, grid_size)
        Returns:
            (batch, embed_dim)
        """
        # Add channel dim: (batch, 1, 256, 256)
        x = grid.unsqueeze(1)
        return self.cnn(x)


class TransformerStateEncoder(nn.Module):
    """
    Transformer encoder for processing variable-length tree sequences.
    Uses attention mask to handle padding.
    
    Input: Combined tree + pairwise embeddings (batch, max_trees, embed_dim)
    Output: Global state representation (batch, state_dim)
    """
    
    def __init__(self, embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2,
                 state_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # No positional encoding - tree order doesn't matter

        # FIX: norm_first=True for better gradient flow and training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN: daha stabil training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Linear(embed_dim, state_dim)
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, max_trees, embed_dim)
            mask: (batch, max_trees) - 1 for valid, 0 for padding
        Returns:
            (batch, state_dim) - Global state representation
        """
        # No positional encoding needed
        x = embeddings
        
        # Create attention mask (True = ignore)
        padding_mask = (mask == 0)  # (batch, max_trees)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1)  # (batch, max_trees, 1)
        x_masked = x * mask_expanded
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        pooled = x_masked.sum(dim=1) / valid_count  # (batch, embed_dim)
        
        # Project to state dim
        return self.output_proj(pooled)  # (batch, state_dim)


class ScalarEncoder(nn.Module):
    """
    Global scalar özellikleri (5 dim) embed eder.
    
    Input: (batch, 5)
    Output: (batch, scalar_embed_dim)
    """
    
    def __init__(self, input_dim: int = 5, embed_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalars: (batch, 5)
        Returns:
            (batch, embed_dim)
        """
        return self.mlp(scalars)


class Actor(nn.Module):
    """
    Policy network - outputs action distribution parameters.

    Action: (x, y, rotation) continuous
    Output: mean and std for Gaussian distribution
    """

    def __init__(self, state_dim: int = 288, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # FIX: State-dependent log_std for adaptive exploration
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            mean: (batch, action_dim)
            log_std: (batch, action_dim)
        """
        features = self.mlp(state)
        mean = self.mean_head(features)
        # FIX: State-dependent std with stability clipping
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent explosion/vanishing

        return mean, log_std


class Critic(nn.Module):
    """
    Value network - estimates state value V(s).
    """
    
    def __init__(self, state_dim: int = 288, hidden_dim: int = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            value: (batch, 1)
        """
        return self.mlp(state)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    """
    
    def __init__(self, max_trees: int = 200, tree_feature_dim: int = 10,
                 pairwise_feature_dim: int = 10, scalar_dim: int = 5,
                 embed_dim: int = 64, state_dim: int = 256, action_dim: int = 3,
                 action_low: np.ndarray = None, action_high: np.ndarray = None):
        super().__init__()
        
        self.max_trees = max_trees
        self.action_dim = action_dim
        
        # Action bounds for scaling
        if action_low is None:
            action_low = np.array([-10.0, -10.0, 0.0])
        if action_high is None:
            action_high = np.array([10.0, 10.0, 360.0])
        
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2)
        
        # Encoders
        self.tree_encoder = TreeFeatureEncoder(tree_feature_dim, embed_dim)
        self.pairwise_encoder = PairwiseFeatureEncoder(pairwise_feature_dim, embed_dim)
        
        # Transformer (input: combined embed_dim * 2)
        combined_embed_dim = embed_dim * 2
        self.transformer = TransformerStateEncoder(
            embed_dim=combined_embed_dim,
            n_heads=4,
            n_layers=2,
            state_dim=state_dim
        )
        
        # Scalar encoder
        scalar_embed_dim = 32
        self.scalar_encoder = ScalarEncoder(scalar_dim, scalar_embed_dim)
        
        # Grid encoder (CNN for 256x256 occupancy grid with fill ratios)
        grid_embed_dim = 32
        self.grid_encoder = GridEncoder(grid_size=256, embed_dim=grid_embed_dim)
        
        # Combined state dim = state_dim + scalar_embed_dim + grid_embed_dim
        combined_state_dim = state_dim + scalar_embed_dim + grid_embed_dim
        
        # Actor and Critic
        self.actor = Actor(combined_state_dim, action_dim)
        self.critic = Critic(combined_state_dim)
    
    def _encode_state(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation into state vector."""
        tree_features = obs["tree_features"]
        pairwise_features = obs["pairwise_features"]
        scalars = obs["scalars"]
        mask = obs["mask"]
        occupancy_grid = obs["occupancy_grid"]
        
        # Encode tree and pairwise features
        tree_embed = self.tree_encoder(tree_features)  # (batch, max_trees, 64)
        pairwise_embed = self.pairwise_encoder(pairwise_features)  # (batch, max_trees, 64)
        
        # Combine
        combined = torch.cat([tree_embed, pairwise_embed], dim=-1)  # (batch, max_trees, 128)
        
        # Transformer encoding
        state = self.transformer(combined, mask)  # (batch, 256)
        
        # Scalar encoding
        scalar_embed = self.scalar_encoder(scalars)  # (batch, 32)
        
        # Grid encoding (CNN)
        grid_embed = self.grid_encoder(occupancy_grid)  # (batch, 32)
        
        # Final state: transformer + scalars + grid
        return torch.cat([state, scalar_embed, grid_embed], dim=-1)  # (batch, 320)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Observation dictionary
        Returns:
            mean: (batch, action_dim)
            log_std: (batch, action_dim)
            value: (batch, 1)
        """
        state = self._encode_state(obs)
        mean, log_std = self.actor(state)
        value = self.critic(state)
        
        return mean, log_std, value
    
    def get_action(self, obs: Dict[str, torch.Tensor], 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: (batch, action_dim) - scaled to action bounds
            log_prob: (batch,) log probability of action
            value: (batch, 1) value estimate
        """
        mean, log_std, value = self.forward(obs)
        
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action_raw = mean
        else:
            action_raw = dist.rsample()  # Reparameterization trick
        
        # Apply tanh and scale to action bounds
        action_tanh = torch.tanh(action_raw)
        action = action_tanh * self.action_scale + self.action_bias
        
        # Log probability with tanh correction
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        log_prob -= torch.log(self.action_scale * (1 - action_tanh.pow(2)) + 1e-6).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs: (batch,) log probabilities
            values: (batch, 1) value estimates
            entropy: (batch,) entropy of action distribution
        """
        mean, log_std, value = self.forward(obs)
        
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        
        # Inverse transform actions to raw space
        action_normalized = (actions - self.action_bias) / self.action_scale
        action_normalized = action_normalized.clamp(-0.999, 0.999)  # Avoid atanh explosion
        action_raw = torch.atanh(action_normalized)
        
        # Log probability
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        log_prob -= torch.log(self.action_scale * (1 - action_normalized.pow(2)) + 1e-6).sum(dim=-1)
        
        # Entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


# ==================== HELPER FUNCTIONS ====================

def obs_to_tensor(obs: Dict[str, np.ndarray], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Convert numpy observation to torch tensors."""
    return {
        key: torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in obs.items()
    }


def create_network(config: dict = None) -> ActorCritic:
    """Factory function to create network with default or custom config."""
    default_config = {
        "max_trees": 200,
        "tree_feature_dim": 10,
        "pairwise_feature_dim": 10,
        "scalar_dim": 5,
        "embed_dim": 64,
        "state_dim": 256,
        "action_dim": 3
    }
    
    if config:
        default_config.update(config)
    
    return ActorCritic(**default_config)
