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


class DirectSpatialEncoding(nn.Module):
    """
    Direct spatial encoding from raw tree coordinates.
    No MLP TreeEncoder - uses Fourier features from ALL coordinates.
    
    Input: (batch, n_trees, 33) = center(2) + rotation(1) + 15 corners × 2(30)
    Output: (batch, n_trees, d_model)
    """
    
    def __init__(self, d_model: int = 128, n_freqs: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs
        
        # 33 coordinates × 2 (sin+cos) × n_freqs frequencies
        fourier_dim = 33 * 2 * n_freqs  # 33 * 2 * 8 = 528
        
        # Project Fourier features to d_model
        self.projection = nn.Linear(fourier_dim, d_model)
        
        # Frequency bands (log-spaced)
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)  # [1, 2, 4, 8, 16, 32, 64, 128]
        self.register_buffer('freqs', freqs * math.pi)
    
    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        """
        Encode tree coordinates directly using Fourier features.
        
        Args:
            tree_features: (batch, n_trees, 33) - all coordinates normalized [0,1]
        Returns:
            (batch, n_trees, d_model) - spatial embeddings
        """
        batch_size, n_trees, n_coords = tree_features.shape
        
        # Expand for frequencies: (batch, n_trees, 33, n_freqs)
        x = tree_features.unsqueeze(-1) * self.freqs
        
        # Sin and cos: (batch, n_trees, 33, n_freqs, 2)
        fourier = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
        
        # Flatten: (batch, n_trees, 33 * n_freqs * 2)
        fourier = fourier.reshape(batch_size, n_trees, -1)
        
        # Project to d_model
        embeddings = self.projection(fourier)
        
        return embeddings


class TreeEncoder(nn.Module):
    """Simple linear projection for backward compatibility."""
    
    def __init__(self, tree_dim: int = 33, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Linear(tree_dim, embed_dim)
    
    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(tree_features)


class GridEncoder(nn.Module):
    """CNN encoder for 256x256 occupancy grid."""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        # CNN: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 8 -> 1
        )
        
        self.fc = nn.Linear(128, embed_dim)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, 256, 256) or (batch, 1, 256, 256)
        Returns:
            (batch, embed_dim)
        """
        if grid.dim() == 3:
            grid = grid.unsqueeze(1)  # Add channel dim
        
        x = self.cnn(grid)  # (batch, 128, 1, 1)
        x = x.flatten(1)    # (batch, 128)
        return self.fc(x)   # (batch, embed_dim)


class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention to fuse tree, scalar, and grid embeddings.
    
    Query: scalar + grid embeddings (context)
    Key/Value: tree sequence embeddings (what to attend to)
    """
    
    def __init__(self, embed_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Project scalar and grid to query tokens
        self.query_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Multi-head cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        tree_seq: torch.Tensor,      # (batch, n_trees, embed_dim)
        scalar_embed: torch.Tensor,  # (batch, embed_dim)
        grid_embed: torch.Tensor,    # (batch, embed_dim)
        mask: torch.Tensor = None    # (batch, n_trees) - True for valid
    ) -> torch.Tensor:
        """
        Fuse embeddings using cross-attention.
        
        Returns:
            (batch, embed_dim) - fused representation
        """
        batch_size = tree_seq.shape[0]
        
        # Create query from scalar + grid (context)
        context = torch.cat([scalar_embed, grid_embed], dim=-1)  # (batch, 2*embed_dim)
        query = self.query_proj(context).unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Handle empty tree sequence (no trees placed yet)
        if mask is not None:
            valid_count = mask.sum(dim=1)
            if (valid_count == 0).all():
                # No valid trees - just return query through FFN
                out = self.norm1(query)
                ffn_out = self.ffn(out)
                out = self.norm2(out + ffn_out)
                return out.squeeze(1)
            key_padding_mask = ~mask  # Invert: True for padding
        else:
            key_padding_mask = None
        
        # Cross-attention: query attends to tree sequence
        attn_out, _ = self.cross_attn(
            query=query,
            key=tree_seq,
            value=tree_seq,
            key_padding_mask=key_padding_mask
        )
        
        # Handle NaN from attention
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        
        # Residual + LayerNorm
        attn_out = self.norm1(query + attn_out)
        
        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)
        
        return out.squeeze(1)  # (batch, embed_dim)


class TransformerActorCritic(nn.Module):
    """
    Transformer-based Actor-Critic for tree placement.
    
    Uses self-attention over all placed trees - no gradient vanishing!
    Every tree can directly attend to every other tree.
    """
    
    def __init__(
        self,
        tree_dim: int = 33,          # FULL: center (2) + rot (1) + 15 corners × 2 = 33
        scalar_dim: int = 5,         # Global scalar features
        embed_dim: int = 128,        # Transformer embedding dim
        n_heads: int = 8,            # Number of attention heads
        n_layers: int = 6,           # Number of transformer layers
        max_trees: int = 200,        # Maximum trees to handle
        action_dim: int = 3          # (x, y, rotation) - ABSOLUTE
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_trees = max_trees
        self.action_dim = action_dim
        
        # Action bounds (ABSOLUTE coordinates)
        action_low = np.array([0.0, 0.0, 0.0])      # x, y, rotation
        action_high = np.array([10.0, 10.0, 360.0])
        
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2)
        
        # Direct spatial encoding from all coordinates (no TreeEncoder MLP)
        self.spatial_encoder = DirectSpatialEncoding(embed_dim, n_freqs=8)
        
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
        
        # Grid encoder (CNN for occupancy grid)
        self.grid_encoder = GridEncoder(embed_dim)
        
        # Cross-attention fusion (replaces concat)
        self.fusion = CrossAttentionFusion(embed_dim, n_heads=4)
        
        # State dim after fusion (single embed_dim, not 3*embed_dim)
        state_dim = embed_dim
        
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
        
        # Add spatial positional encoding (based on tree coordinates)
        x = self.pos_encoder(x, tree_features)
        
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
        mask: torch.Tensor = None,
        grid: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            tree_features: (batch, n_trees, tree_dim)
            scalars: (batch, scalar_dim)
            mask: (batch, n_trees) - True for valid trees
            grid: (batch, 256, 256) - occupancy grid
        
        Returns:
            mean, log_std, value
        """
        # Direct spatial encoding from raw coordinates (no MLP)
        tree_seq = self.spatial_encoder(tree_features)  # (batch, n_trees, embed_dim)
        
        # Apply transformer encoder for tree-to-tree attention
        if mask is not None:
            attn_mask = ~mask  # True = ignore
        else:
            attn_mask = None
        tree_seq = self.transformer(tree_seq, src_key_padding_mask=attn_mask)
        tree_seq = torch.nan_to_num(tree_seq, nan=0.0)  # Handle NaN from transformer
        
        # Encode scalars
        scalar_embed = self.scalar_encoder(scalars)  # (batch, embed_dim)
        
        # Encode grid
        if grid is not None:
            grid_embed = self.grid_encoder(grid)  # (batch, embed_dim)
        else:
            grid_embed = torch.zeros_like(scalar_embed)
        
        # Cross-attention fusion: scalar+grid query attends to tree sequence
        state = self.fusion(tree_seq, scalar_embed, grid_embed, mask)  # (batch, embed_dim)
        
        # Handle NaN in state
        state = torch.nan_to_num(state, nan=0.0)
        
        # Actor
        mean = self.actor_mean(state)
        mean = torch.nan_to_num(mean, nan=0.0)  # Prevent NaN in distribution
        
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
        grid: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        
        mean, log_std, value = self.forward(tree_features, scalars, mask, grid)
        
        # Handle NaN
        log_std = torch.nan_to_num(log_std, nan=-2.0)
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
        
        # Scale to action bounds
        action_scaled = action * self.action_scale + self.action_bias
        
        return action_scaled, log_prob, value
    
    def evaluate_actions(
        self,
        tree_features: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor = None,
        grid: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        
        mean, log_std, value = self.forward(tree_features, scalars, mask, grid)
        
        # Handle NaN in log_std before exp
        log_std = torch.nan_to_num(log_std, nan=-2.0)  # Default to std~0.13
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-6)  # Ensure positive
        
        # Unscale actions
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        actions_raw = torch.atanh(torch.clamp(actions_unscaled, -0.999, 0.999))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy
