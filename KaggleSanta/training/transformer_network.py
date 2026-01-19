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


class AbsoluteSpatialEncoding(nn.Module):
    """
    Fourier encoding for ABSOLUTE tree coordinates only.
    
    Input: (batch, n_trees, 33) = center(2) + rotation(1) + 15 corners × 2(30)
    Output: (batch, n_trees, d_model)
    """
    
    def __init__(self, d_model: int = 96, n_freqs: int = 8, abs_dim: int = 33):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs
        
        # 33 coordinates × 2 (sin+cos) × n_freqs frequencies
        fourier_dim = abs_dim * 2 * n_freqs  # 33 * 2 * 8 = 528
        
        # Project Fourier features to d_model
        self.projection = nn.Linear(fourier_dim, d_model)
        
        # Frequency bands (log-spaced)
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer('freqs', freqs * math.pi)
    
    def forward(self, abs_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            abs_features: (batch, n_trees, 33) - absolute coords
        Returns:
            (batch, n_trees, d_model)
        """
        batch_size, n_trees, _ = abs_features.shape
        
        x = abs_features.unsqueeze(-1) * self.freqs
        fourier = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
        fourier = fourier.reshape(batch_size, n_trees, -1)
        
        return self.projection(fourier)


class HybridRelativeEncoding(nn.Module):
    """
    Hybrid encoding for RELATIVE features with separate Polar and Cartesian branches.
    
    Input: (batch, n_trees, 6) = [dx, dy, dist, angle, dx_bbox, dy_bbox]
    - Cartesian: dx, dy, dx_bbox, dy_bbox (4 features)
    - Polar: dist, angle (2 features)
    
    Output: (batch, n_trees, d_model)
    """
    
    def __init__(self, d_model: int = 32, n_freqs: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs
        
        # Cartesian branch: dx, dy, dx_bbox, dy_bbox (4 features)
        cartesian_fourier_dim = 4 * 2 * n_freqs  # 4 * 2 * 8 = 64
        self.cartesian_proj = nn.Linear(cartesian_fourier_dim, d_model // 2)
        
        # Polar branch: dist, angle (2 features) - higher frequencies for fine detail
        polar_fourier_dim = 2 * 2 * n_freqs  # 2 * 2 * 8 = 32
        self.polar_proj = nn.Linear(polar_fourier_dim, d_model // 2)
        
        # Frequency bands - standard for Cartesian
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer('cart_freqs', freqs * math.pi)
        
        # Higher frequencies for Polar (fine-grained distance/angle)
        polar_freqs = 2.0 ** torch.linspace(1, n_freqs, n_freqs)  # [2, 4, ..., 256]
        self.register_buffer('polar_freqs', polar_freqs * math.pi)
    
    def forward(self, rel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_features: (batch, n_trees, 6) - [dx, dy, dist, angle, dx_bbox, dy_bbox]
        Returns:
            (batch, n_trees, d_model)
        """
        batch_size, n_trees, _ = rel_features.shape
        
        # Split features
        dx, dy = rel_features[..., 0:1], rel_features[..., 1:2]
        dist, angle = rel_features[..., 2:3], rel_features[..., 3:4]
        dx_bbox, dy_bbox = rel_features[..., 4:5], rel_features[..., 5:6]
        
        # Cartesian features: (batch, n_trees, 4)
        cartesian = torch.cat([dx, dy, dx_bbox, dy_bbox], dim=-1)
        
        # Polar features: (batch, n_trees, 2)
        polar = torch.cat([dist, angle], dim=-1)
        
        # Fourier encode Cartesian
        cart_x = cartesian.unsqueeze(-1) * self.cart_freqs
        cart_fourier = torch.stack([torch.sin(cart_x), torch.cos(cart_x)], dim=-1)
        cart_fourier = cart_fourier.reshape(batch_size, n_trees, -1)
        cart_embed = self.cartesian_proj(cart_fourier)
        
        # Fourier encode Polar (higher frequencies)
        polar_x = polar.unsqueeze(-1) * self.polar_freqs
        polar_fourier = torch.stack([torch.sin(polar_x), torch.cos(polar_x)], dim=-1)
        polar_fourier = polar_fourier.reshape(batch_size, n_trees, -1)
        polar_embed = self.polar_proj(polar_fourier)
        
        # Concatenate: (batch, n_trees, d_model)
        return torch.cat([cart_embed, polar_embed], dim=-1)


class CombinedTreeEncoder(nn.Module):
    """
    Combines AbsoluteSpatialEncoding + HybridRelativeEncoding.
    
    Input: (batch, n_trees, 39) = absolute(33) + relative(6)
    Output: (batch, n_trees, embed_dim)
    """
    
    def __init__(self, embed_dim: int = 128, n_freqs: int = 8):
        super().__init__()
        
        # Absolute branch: 33 features → 96D
        self.absolute_encoder = AbsoluteSpatialEncoding(d_model=96, n_freqs=n_freqs, abs_dim=33)
        
        # Relative branch: 6 features → 32D (Polar + Cartesian)
        self.relative_encoder = HybridRelativeEncoding(d_model=32, n_freqs=n_freqs)
        
        # Fusion: 96 + 32 = 128 → embed_dim
        self.fusion = nn.Linear(128, embed_dim)
    
    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tree_features: (batch, n_trees, 39)
        Returns:
            (batch, n_trees, embed_dim)
        """
        # Split absolute (first 33) and relative (last 6)
        abs_features = tree_features[..., :33]
        rel_features = tree_features[..., 33:]
        
        # Encode separately
        abs_embed = self.absolute_encoder(abs_features)  # (batch, n_trees, 96)
        rel_embed = self.relative_encoder(rel_features)  # (batch, n_trees, 32)
        
        # Fuse
        combined = torch.cat([abs_embed, rel_embed], dim=-1)  # (batch, n_trees, 128)
        
        return self.fusion(combined)  # (batch, n_trees, embed_dim)


class TreeEncoder(nn.Module):
    """Simple linear projection for backward compatibility."""
    
    def __init__(self, tree_dim: int = 39, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Linear(tree_dim, embed_dim)
    
    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(tree_features)


class SpatialContextEncoder(nn.Module):
    """
    Encode spatial context (bbox) using Fourier features.
    Same approach as DirectSpatialEncoding for trees.
    
    Input: bbox(4) = [minx, miny, maxx, maxy]
    Output: (batch, embed_dim)
    """
    
    def __init__(self, input_dim: int = 4, embed_dim: int = 64, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        
        # 4 coordinates × 2 (sin+cos) × n_freqs = 64 dim
        fourier_dim = input_dim * 2 * n_freqs
        
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Frequency bands (log-spaced)
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer('freqs', freqs * math.pi)
    
    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: (batch, 4) - [minx, miny, maxx, maxy]
        Returns:
            (batch, embed_dim) - spatial context embedding
        """
        # Expand for frequencies: (batch, 4, n_freqs)
        x = bbox.unsqueeze(-1) * self.freqs
        
        # Sin and cos: (batch, 4, n_freqs, 2)
        fourier = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
        
        # Flatten: (batch, 4 * n_freqs * 2)
        fourier = fourier.reshape(bbox.shape[0], -1)
        
        return self.projection(fourier)


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm for CNN."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (1x1 conv if dimensions change)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Residual connection
        out = self.relu(out)
        return out


class GridEncoder(nn.Module):
    """CNN encoder for 512x512 occupancy grid with BatchNorm and Residual connections."""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        # Initial conv: 512 -> 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # 512 -> 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with BatchNorm
        self.layer1 = ResidualBlock(32, 64, stride=2)    # 256 -> 128
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 128 -> 64
        self.layer3 = ResidualBlock(128, 128, stride=2)  # 64 -> 32
        self.layer4 = ResidualBlock(128, 128, stride=2)  # 32 -> 16
        self.layer5 = ResidualBlock(128, 128, stride=2)  # 16 -> 8
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 8 -> 1
        self.fc = nn.Linear(128, embed_dim)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, 512, 512) or (batch, 1, 512, 512)
        Returns:
            (batch, embed_dim)
        """
        if grid.dim() == 3:
            grid = grid.unsqueeze(1)  # Add channel dim
        
        x = self.conv1(grid)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


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

    NEW: Collision prediction head for auxiliary learning.
    """

    def __init__(
        self,
        tree_dim: int = 39,          # FULL: center (2) + rot (1) + 15 corners × 2 (30) + relative (6) = 39
        scalar_dim: int = 5,         # Global scalar features
        collision_dim: int = 6,      # Collision features (NEW!)
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
        
        # Combined encoder: Absolute (33D) + Hybrid Relative (6D with Polar+Cartesian)
        self.spatial_encoder = CombinedTreeEncoder(embed_dim, n_freqs=8)
        
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

        # Collision encoder (NEW! - collision-aware features)
        self.collision_encoder = nn.Sequential(
            nn.Linear(collision_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Grid encoder (CNN for occupancy grid)
        self.grid_encoder = GridEncoder(embed_dim)
        
        # Spatial context encoder (bbox = 4 dim -> 64 dim with Fourier features)
        self.spatial_context_encoder = SpatialContextEncoder(input_dim=4, embed_dim=64, n_freqs=8)
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(embed_dim, n_heads=4)

        # State dim after fusion + spatial_context + collision_embed
        state_dim = embed_dim + 64 + 64  # fusion + bbox_context + collision
        
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

        # Collision prediction head (NEW! - auxiliary task)
        # Outputs raw logits - BCEWithLogits applies sigmoid internally
        self.collision_predictor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # No sigmoid! BCEWithLogits handles it
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
        grid: torch.Tensor = None,
        bbox: torch.Tensor = None,
        collision_features: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            tree_features: (batch, n_trees, tree_dim)
            scalars: (batch, scalar_dim)
            mask: (batch, n_trees) - True for valid trees
            grid: (batch, 512, 512) - occupancy grid
            bbox: (batch, 4) - bounding box coords [minx, miny, maxx, maxy]
            collision_features: (batch, collision_dim) - collision-aware features (NEW!)

        Returns:
            mean, log_std, value, collision_prob
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
        fusion_out = self.fusion(tree_seq, scalar_embed, grid_embed, mask)  # (batch, embed_dim)
        
        # Encode bbox spatial context (only bbox now, no free_space)
        batch_size = tree_features.shape[0]
        device = tree_features.device

        if bbox is None:
            bbox = torch.zeros(batch_size, 4, device=device)

        # Fourier encoding for bbox only (4 dim)
        context_embed = self.spatial_context_encoder(bbox)  # (batch, 64)

        # Encode collision features (NEW!)
        if collision_features is None:
            collision_features = torch.zeros(batch_size, 6, device=device)
        collision_embed = self.collision_encoder(collision_features)  # (batch, 64)

        # Concat fusion output with spatial context + collision embeddings
        state = torch.cat([fusion_out, context_embed, collision_embed], dim=-1)  # (batch, embed_dim + 64 + 64)
        
        # Handle NaN in state
        state = torch.nan_to_num(state, nan=0.0)
        
        # Actor
        mean = self.actor_mean(state)
        mean = torch.nan_to_num(mean, nan=0.0)  # Prevent NaN in distribution
        
        log_std = self.actor_log_std(state)
        log_std = torch.clamp(log_std, -5, 0)
        
        # Critic
        value = self.critic(state)

        # Collision prediction (NEW! - auxiliary task)
        collision_prob = self.collision_predictor(state)  # (batch, 1)

        return mean, log_std, value, collision_prob
    
    def get_action(
        self,
        tree_features: torch.Tensor,
        scalars: torch.Tensor,
        mask: torch.Tensor = None,
        grid: torch.Tensor = None,
        bbox: torch.Tensor = None,
        collision_features: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""

        mean, log_std, value, collision_prob = self.forward(
            tree_features, scalars, mask, grid, bbox, collision_features
        )
        
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

        return action_scaled, log_prob, value, collision_prob
    
    def evaluate_actions(
        self,
        tree_features: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor = None,
        grid: torch.Tensor = None,
        bbox: torch.Tensor = None,
        collision_features: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""

        mean, log_std, value, collision_prob = self.forward(
            tree_features, scalars, mask, grid, bbox, collision_features
        )
        
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

        return log_prob, value, entropy, collision_prob
