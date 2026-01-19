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
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import argparse

from environment import TreePackingEnvironment, scale_factor


class WarmupScheduler:
    """Learning rate warmup scheduler for training stability."""

    def __init__(self, optimizer, warmup_steps: int = 1000, base_lr: float = 3e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0

    def step(self):
        """Update learning rate."""
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing after warmup
            progress = (self.step_count - self.warmup_steps) / (10000 - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class ObservationNormalizer:
    """Running statistics for observation normalization."""

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.running_mean = {}
        self.running_std = {}
        self.initialized = {}

    def normalize(self, obs: Dict) -> Dict:
        """Normalize observation using running statistics."""
        normalized = {}

        for key, value in obs.items():
            if key == "mask" or key == "grid":
                # Don't normalize binary mask or grid
                normalized[key] = value
                continue

            # Initialize statistics on first call
            if key not in self.initialized:
                self.running_mean[key] = np.mean(value)
                self.running_std[key] = np.std(value) + 1e-8
                self.initialized[key] = True

            # Update running statistics
            current_mean = np.mean(value)
            current_std = np.std(value) + 1e-8

            self.running_mean[key] = self.momentum * self.running_mean[key] + (1 - self.momentum) * current_mean
            self.running_std[key] = self.momentum * self.running_std[key] + (1 - self.momentum) * current_std

            # Normalize
            normalized[key] = (value - self.running_mean[key]) / self.running_std[key]

        return normalized


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
    TREE_DIM = 14  # Enhanced: base(6) + local_context(2) + angular(1) + boundary(1) + radial(4)
    
    def __init__(self, n_trees: int = 50, max_trees: int = 200, 
                 anchor_mode: str = "bbox_center", action_mode: str = "polar",
                 feature_mode: str = "polar"):
        """
        Args:
            anchor_mode: 'last_tree' or 'bbox_center'
                - last_tree: action offset from last placed tree
                - bbox_center: action offset from bbox center (better for compact packing)
            action_mode: 'cartesian' or 'polar'
                - cartesian: [dx, dy, rotation] where dx,dy ∈ [-5, 5]
                - polar: [r, theta, rotation] where r ∈ [0, 5], theta ∈ [-π, π]
            feature_mode: 'cartesian' or 'polar'
                - cartesian: tree features use (dx, dy)
                - polar: tree features use (r, sin_theta, cos_theta)
        """
        super().__init__(n_trees=n_trees, max_trees=max_trees)
        self.anchor_mode = anchor_mode
        self.action_mode = action_mode
        self.feature_mode = feature_mode
        
        from gymnasium import spaces
        
        if action_mode == "polar":
            # Polar: [r, theta, rotation]
            self.action_space = spaces.Box(
                low=np.array([0.0, -np.pi, 0.0], dtype=np.float32),
                high=np.array([5.0, np.pi, 360.0], dtype=np.float32)
            )
        else:  # cartesian
            # Cartesian: [dx, dy, rotation]
            self.action_space = spaces.Box(
                low=np.array([-5.0, -5.0, 0.0], dtype=np.float32),
                high=np.array([5.0, 5.0, 360.0], dtype=np.float32)
            )
        
        self.observation_space = spaces.Dict({
            "relative_trees": spaces.Box(-np.inf, np.inf, shape=(self.MAX_TREES, self.TREE_DIM), dtype=np.float32),
            "mask": spaces.Box(0, 1, shape=(self.MAX_TREES,), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32),
            "collision_features": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "grid": spaces.Box(0, 1, shape=(512, 512), dtype=np.float32),
            "anchor": spaces.Box(0, 10, shape=(2,), dtype=np.float32)
        })
        
        # Cache for expensive computations
        self._bbox_center = None
        self._last_rotation = 0.0
    
    def _get_relative_tree_features(self, tree, anchor_x: float, anchor_y: float,
                                       tree_idx: int = 0) -> np.ndarray:
        """
        Get ENHANCED RELATIVE features for a tree.

        Features (14):
        Base (6): r, sin_theta, cos_theta, r_bbox, sin_theta_bbox, cos_theta_bbox
        Local Context (2): min_clearance, local_density
        Angular (1): angular_dispersion
        Boundary (1): distance_to_boundary
        Radial (4): radial_zone, normalized_r, normalized_dx, normalized_dy

        All coordinates in [0, 10] range (normalized)
        """
        norm = float(scale_factor)

        # Tree center in [0, 10] range
        cx = float(tree.center_x) / norm
        cy = float(tree.center_y) / norm

        # Anchor already in [0, 10] range (passed from _get_observation)
        # Relative to anchor (board_center = 5, 5)
        dx = cx - anchor_x
        dy = cy - anchor_y
        r = np.sqrt(dx**2 + dy**2)

        # Relative to bbox center (also in [0, 10] range)
        if self._bbox_center is not None:
            dx_bbox = cx - self._bbox_center[0]
            dy_bbox = cy - self._bbox_center[1]
        else:
            dx_bbox = 0.0
            dy_bbox = 0.0
        r_bbox = np.sqrt(dx_bbox**2 + dy_bbox**2)

        # Base features (6)
        if self.feature_mode == "polar":
            # Polar: use sin/cos instead of dx/dy
            sin_theta = dy / (r + 1e-8)
            cos_theta = dx / (r + 1e-8)
            sin_theta_bbox = dy_bbox / (r_bbox + 1e-8)
            cos_theta_bbox = dx_bbox / (r_bbox + 1e-8)

            base_features = [r, sin_theta, cos_theta, r_bbox, sin_theta_bbox, cos_theta_bbox]
        else:
            # Cartesian
            angle = np.arctan2(dy, dx)
            base_features = [dx, dy, r, angle, dx_bbox, dy_bbox]

        # Local context features (2)
        min_clearance, local_density = self._compute_local_context(tree_idx, k=5)

        # Angular dispersion (1)
        angular_std = self._compute_angular_dispersion(tree_idx)

        # Distance to boundary (1)
        dist_to_boundary = min(cx, 10.0 - cx, cy, 10.0 - cy) / 10.0  # Normalized to [0, 1]

        # Radial features (4)
        radial_zone = min(int(r / 2.0), 4) / 5.0  # 0-5 zones, normalized
        normalized_r = min(r / 10.0, 1.0)
        normalized_dx = np.clip(dx / 10.0, -1.0, 1.0)
        normalized_dy = np.clip(dy / 10.0, -1.0, 1.0)

        return np.array(base_features + [
            min_clearance, local_density, angular_std, dist_to_boundary,
            radial_zone, normalized_r, normalized_dx, normalized_dy
        ], dtype=np.float32)
    
    def _compute_local_context(self, tree_idx: int, k: int = 5) -> Tuple[float, float]:
        """
        Compute local context for a tree:
        - min_clearance: minimum distance to any neighbor (normalized)
        - local_density: number of trees within radius / k
        """
        n = len(self.placed_trees)
        norm = float(scale_factor)

        if n < 2 or tree_idx >= n:
            return 1.0, 0.0  # No neighbors, no density

        tree = self.placed_trees[tree_idx]

        # Compute distances to all other trees
        distances = []
        for j, other in enumerate(self.placed_trees):
            if j != tree_idx:
                dist = tree.polygon.distance(other.polygon) / norm
                distances.append(dist)

        if not distances:
            return 1.0, 0.0

        min_clearance = min(distances)
        min_clearance = min(min_clearance, 1.0)  # Clip to [0, 1]

        # Local density: count trees within radius 1.5
        close_count = sum(1 for d in distances if d < 1.5)
        local_density = min(close_count / k, 1.0)  # Normalize

        return min_clearance, local_density

    def _compute_angular_dispersion(self, tree_idx: int) -> float:
        """
        Compute angular dispersion of nearby trees.
        Measures how evenly spread the trees are around this tree.

        Returns:
            float: Standard deviation of angles to nearby trees, normalized to [0, 1]
        """
        n = len(self.placed_trees)
        norm = float(scale_factor)

        if n < 3 or tree_idx >= n:
            return 0.0  # Not enough trees for angular analysis

        tree = self.placed_trees[tree_idx]
        cx = float(tree.center_x) / norm
        cy = float(tree.center_y) / norm

        # Find k=5 nearest neighbors
        angles = []
        distances = []
        for j, other in enumerate(self.placed_trees):
            if j != tree_idx:
                ox = float(other.center_x) / norm
                oy = float(other.center_y) / norm
                dx = ox - cx
                dy = oy - cy
                dist = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                distances.append((dist, angle))

        if not distances:
            return 0.0

        # Sort by distance, take k=5 nearest
        distances.sort(key=lambda x: x[0])
        k = min(5, len(distances))
        nearest_angles = [angle for _, angle in distances[:k]]

        if len(nearest_angles) < 2:
            return 0.0

        # Compute angular standard deviation
        # Convert to unit vectors to handle circular statistics
        sin_sum = sum(np.sin(a) for a in nearest_angles)
        cos_sum = sum(np.cos(a) for a in nearest_angles)
        mean_angle = np.arctan2(sin_sum, cos_sum)

        # Angular deviation
        deviations = [min(abs(a - mean_angle), 2*np.pi - abs(a - mean_angle)) for a in nearest_angles]
        angular_std = np.std(deviations) / np.pi  # Normalize to [0, 1]

        return float(angular_std)
    
    def _update_bbox_center(self):
        """Update cached bbox center."""
        norm = float(scale_factor)
        if len(self.placed_trees) > 0:
            from shapely.ops import unary_union
            all_polys = [t.polygon for t in self.placed_trees]
            bounds = unary_union(all_polys).bounds
            self._bbox_center = (
                (bounds[0] + bounds[2]) / (2 * norm),
                (bounds[1] + bounds[3]) / (2 * norm)
            )
        else:
            self._bbox_center = (5.0, 5.0)  # Center of board
    
    def _get_observation(self) -> Dict:
        """Get relative observation with enhanced features."""
        n = len(self.placed_trees)
        norm = float(scale_factor)
        
        # Update bbox center cache
        self._update_bbox_center()
        
        # Anchor based on anchor_mode (in [0, 10] range)
        if self.anchor_mode == "board_center":
            # Fixed board center
            anchor_x = 5.0
            anchor_y = 5.0
        elif n > 0:
            if self.anchor_mode == "bbox_center":
                anchor_x = self._bbox_center[0]
                anchor_y = self._bbox_center[1]
            else:  # last_tree
                last_tree = self.placed_trees[-1]
                anchor_x = float(last_tree.center_x) / norm
                anchor_y = float(last_tree.center_y) / norm
        else:
            anchor_x = 5.0  # Center
            anchor_y = 5.0
        
        if n > 0:
            self._last_rotation = float(self.placed_trees[-1].angle)
        else:
            self._last_rotation = 0.0
        
        # Relative tree features (now 8D)
        relative_trees = np.zeros((self.MAX_TREES, self.TREE_DIM), dtype=np.float32)
        mask = np.zeros(self.MAX_TREES, dtype=np.float32)
        
        for i, tree in enumerate(self.placed_trees):
            if i >= self.MAX_TREES:
                break
            relative_trees[i] = self._get_relative_tree_features(tree, anchor_x, anchor_y, i)
            mask[i] = 1.0
        
        # Scalars (ENABLED - global context features)
        # Compute direction from last few trees
        dir_sin, dir_cos, momentum = 0.0, 0.0, 0.0
        if n >= 2:
            t1, t2 = self.placed_trees[-2], self.placed_trees[-1]
            dx = (float(t2.center_x) - float(t1.center_x)) / norm
            dy = (float(t2.center_y) - float(t1.center_y)) / norm
            momentum = np.sqrt(dx**2 + dy**2)
            if momentum > 1e-6:
                dir_sin = dy / momentum
                dir_cos = dx / momentum
            momentum = min(momentum / 2.0, 1.0)

        # BBox aspect ratio and offset from board center
        aspect_ratio = 0.5
        if n > 0:
            from shapely.ops import unary_union
            all_polys = [t.polygon for t in self.placed_trees]
            bounds = unary_union(all_polys).bounds
            minx, miny, maxx, maxy = bounds
            width = (maxx - minx) / norm
            height = (maxy - miny) / norm
            if height > 0.01:
                aspect_ratio = (width / height)
                aspect_ratio = min(max(aspect_ratio, 0.1), 10.0) / 10.0
            else:
                aspect_ratio = 0.5

        # BBox center offset from board center (5, 5)
        bbox_offset_x = self._bbox_center[0] - 5.0 if self._bbox_center else 0.0
        bbox_offset_y = self._bbox_center[1] - 5.0 if self._bbox_center else 0.0
        bbox_offset_x = bbox_offset_x / 5.0
        bbox_offset_y = bbox_offset_y / 5.0

        # Additional global features
        progress_ratio = n / self.n_trees
        fill_ratio = self.fill_ratio if n > 0 else 0.0

        scalars = np.array([
            dir_sin, dir_cos, momentum,
            aspect_ratio,
            bbox_offset_x, bbox_offset_y,
            progress_ratio,
            fill_ratio,
            n / self.max_trees,  # Normalized placement count
            (self.n_trees - n) / self.n_trees,  # Remaining ratio
            self.total_tree_area / (1e30 if n > 0 else 1.0)  # Total area normalized
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
    
    def _get_anchor(self) -> Tuple[float, float]:
        """Get current anchor position based on anchor_mode."""
        norm = float(scale_factor)
        
        # Board center is always the fixed center
        if self.anchor_mode == "board_center":
            return 5.0 * norm, 5.0 * norm
        
        if len(self.placed_trees) == 0:
            return 5.0 * norm, 5.0 * norm  # Center for first tree
        
        if self.anchor_mode == "bbox_center":
            # Use bbox center as anchor
            if self._bbox_center is not None:
                return self._bbox_center[0] * norm, self._bbox_center[1] * norm
            else:
                return 5.0 * norm, 5.0 * norm
        else:
            # Default: last_tree
            last_tree = self.placed_trees[-1]
            return float(last_tree.center_x), float(last_tree.center_y)
    
    def step(self, action: np.ndarray):
        """
        Step with RELATIVE action.
        
        Action depends on action_mode:
        - cartesian: [dx, dy, rotation] where dx,dy are offsets
        - polar: [r, theta, rotation] where r is distance, theta is angle
        """
        # Get anchor position based on mode
        anchor_x, anchor_y = self._get_anchor()
        norm = float(scale_factor)
        
        if self.action_mode == "polar":
            # Polar: [r, theta, rotation]
            r, theta, rotation = action[0], action[1], action[2]
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
        else:  # cartesian
            # Cartesian: [dx, dy, rotation]
            dx, dy, rotation = action[0], action[1], action[2]
        
        # Convert relative to absolute
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
    Encode relative tree features with Fourier encoding.
    Input: (batch, n_trees, 14) - enhanced relative features
    Output: (batch, n_trees, embed_dim)
    """

    def __init__(self, input_dim: int = 14, embed_dim: int = 128, n_freqs: int = 8):
        super().__init__()
        
        # Fourier encoding for relative features
        fourier_dim = input_dim * 2 * n_freqs  # 6 * 2 * 8 = 96
        
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),  # 96 -> 128
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
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


class SimpleFusion(nn.Module):
    """
    Multi-modal concatenation-based fusion.
    Concatenates tree_pool + grid + scalar + collision and processes with MLP.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        # 4 modalities * embed_dim = 512 -> embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(
        self,
        tree_pool: torch.Tensor,       # (batch, embed_dim) - pooled trees
        scalar_embed: torch.Tensor,    # (batch, embed_dim) - global context
        collision_embed: torch.Tensor, # (batch, embed_dim) - collision features
        grid_embed: torch.Tensor,      # (batch, embed_dim) - spatial grid
        mask: torch.Tensor = None      # unused, kept for API compatibility
    ) -> torch.Tensor:
        # Sanitize inputs
        tree_pool = torch.nan_to_num(tree_pool, nan=0.0)
        scalar_embed = torch.nan_to_num(scalar_embed, nan=0.0)
        collision_embed = torch.nan_to_num(collision_embed, nan=0.0)
        grid_embed = torch.nan_to_num(grid_embed, nan=0.0)

        # Concatenate all modalities
        concat = torch.cat([tree_pool, grid_embed, scalar_embed, collision_embed], dim=-1)

        # Fuse
        out = self.fusion(concat)
        out = torch.nan_to_num(out, nan=0.0)

        return out  # (batch, embed_dim)


# Residual Block for Grid Encoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class SpatialPyramidPool(nn.Module):
    """
    Spatial Pyramid Pooling - preserves multi-scale spatial information.
    Pools at multiple scales and concatenates.
    """
    
    def __init__(self, pool_sizes: list = [4, 2, 1]):
        super().__init__()
        self.pool_sizes = pool_sizes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        pools = []
        
        for size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, size)  # (batch, channels, size, size)
            pooled = pooled.view(batch_size, -1)     # (batch, channels * size * size)
            pools.append(pooled)
        
        return torch.cat(pools, dim=1)


class SpatialGridEncoder(nn.Module):
    """
    Grid encoder with CoordConv + Spatial Pyramid Pooling.
    
    - CoordConv: Adds x,y coordinate channels so CNN knows pixel positions
    - SPP: Preserves spatial info at 4x4, 2x2, 1x1 scales
    
    Input: (batch, 1, 512, 512)
    Output: (batch, embed_dim)
    """
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        # Stem with Polar CoordConv (4 channels: image + r + sin_θ + cos_θ)
        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, 7, stride=2, padding=3, bias=False),  # 512 -> 256
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # ResBlocks (lighter: 16 -> 32 -> 64 -> 64)
        self.blocks = nn.Sequential(
            ResidualBlock(16, 32, stride=2),   # 256 -> 128
            ResidualBlock(32, 64, stride=2),   # 128 -> 64
            ResidualBlock(64, 64, stride=2),   # 64 -> 32
            ResidualBlock(64, 64, stride=2),   # 32 -> 16
        )
        
        # SPP: 4x4 + 2x2 + 1x1 = 16+4+1 = 21 cells * 64 channels = 1344
        self.spp = SpatialPyramidPool([4, 2, 1])
        
        # Projection to embed_dim (simpler)
        spp_output_dim = 64 * (16 + 4 + 1)  # 1344
        self.projection = nn.Sequential(
            nn.Linear(spp_output_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Cache coordinate grids
        self._cached_coords = None
        self._cached_size = None
    
    def _get_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Generate POLAR coordinate channels (r, sin_θ, cos_θ) from board center."""
        batch_size, _, h, w = x.shape
        device = x.device
        
        # Check cache
        if self._cached_size != (h, w) or self._cached_coords is None or self._cached_coords.device != device:
            # Create coordinate grids in [-1, 1] range centered at (0, 0)
            y_grid = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
            x_grid = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(1, 1, h, w)
            
            # Polar coordinates from center (0, 0)
            r = torch.sqrt(x_grid**2 + y_grid**2)  # [0, sqrt(2)] -> normalize to [0, 1]
            r = r / 1.414  # sqrt(2)
            
            sin_theta = y_grid / (r * 1.414 + 1e-8)  # sin = y/r
            cos_theta = x_grid / (r * 1.414 + 1e-8)  # cos = x/r
            
            self._cached_coords = torch.cat([r, sin_theta, cos_theta], dim=1)  # (1, 3, h, w)
            self._cached_size = (h, w)
        
        # Expand to batch size
        return self._cached_coords.expand(batch_size, -1, -1, -1)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (batch, 1, 512, 512)
        
        # Add polar coordinate channels (r, sin_θ, cos_θ)
        coords = self._get_coord_channels(grid)  # (batch, 3, 512, 512)
        x = torch.cat([grid, coords], dim=1)     # (batch, 4, 512, 512)
        
        # Stem + ResBlocks
        x = self.stem(x)
        x = self.blocks(x)  # (batch, 64, 16, 16)
        
        # SPP
        x = self.spp(x)  # (batch, 1344)
        
        # Project to embed_dim
        x = self.projection(x)  # (batch, embed_dim)
        
        return x

class RelativeActorCritic(nn.Module):
    """
    Enhanced Actor-Critic for relative action space.
    
    Improvements:
    - Learnable positional embeddings
    - Multi-query cross-attention (no bottleneck)
    - ResNet-style grid encoder
    - 8D relative features
    """
    
    MAX_TREES = 100
    
    def __init__(
        self,
        tree_dim: int = 14,  # Enhanced: 14 relative features
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        action_dim: int = 3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
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
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Scalar encoder (ENABLED - global context)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Collision encoder (ENABLED - collision-aware features)
        self.collision_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # ===== Spatial Grid Encoder with CoordConv + SPP =====
        self.grid_encoder = SpatialGridEncoder(embed_dim)
        
        # Simple concat-based fusion
        self.fusion = SimpleFusion(embed_dim)
        
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
    
    def _build_resnet_grid_encoder(self, embed_dim: int) -> nn.Module:
        """Build ResNet-style grid encoder with residual blocks."""
        return nn.Sequential(
            # Stem: 1024 -> 512
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # ResBlocks: 512 -> 256 -> 128 -> 64 -> 32 -> 16
            ResidualBlock(32, 64, stride=2),   # 512 -> 256
            ResidualBlock(64, 128, stride=2),  # 256 -> 128
            ResidualBlock(128, 128, stride=2), # 128 -> 64  
            ResidualBlock(128, 128, stride=2), # 64 -> 32
            ResidualBlock(128, 128, stride=2), # 32 -> 16
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, relative_trees, scalars, mask, collision_features, grid):
        batch_size = relative_trees.shape[0]
        device = relative_trees.device
        
        # Sanitize inputs
        relative_trees = torch.nan_to_num(relative_trees, nan=0.0)
        scalars = torch.nan_to_num(scalars, nan=0.0)
        collision_features = torch.nan_to_num(collision_features, nan=0.0)
        grid = torch.nan_to_num(grid, nan=0.0)
        
        # Encode trees (no positional encoding - set-based problem)
        tree_embed = self.tree_encoder(relative_trees)  # (batch, n_trees, embed_dim)
        tree_embed = torch.nan_to_num(tree_embed, nan=0.0)
        
        # Transformer
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None
        
        tree_seq = self.transformer(tree_embed, src_key_padding_mask=attn_mask)
        tree_seq = torch.nan_to_num(tree_seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pool trees (mean of valid)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            tree_pool = (tree_seq * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            tree_pool = tree_seq.mean(dim=1)
        tree_pool = torch.nan_to_num(tree_pool, nan=0.0)
        
        # Encode other features
        scalar_embed = self.scalar_encoder(scalars)  # ENABLED - global context
        collision_embed = self.collision_encoder(collision_features)  # ENABLED - collision awareness
        
        # Grid
        grid_input = grid.unsqueeze(1)  # (batch, 1, 512, 512)
        grid_embed = self.grid_encoder(grid_input)
        grid_embed = torch.nan_to_num(grid_embed, nan=0.0)
        
        # Simple fusion (concat + MLP)
        state = self.fusion(
            tree_pool, scalar_embed, collision_embed, grid_embed
        )
        state = torch.nan_to_num(state, nan=0.0)
        
        # Actor - with NaN protection
        mean = self.actor_mean(state)
        mean = torch.nan_to_num(mean, nan=0.0)  # Critical guard
        
        log_std = self.actor_log_std(state)
        log_std = torch.nan_to_num(log_std, nan=-2.0)  # Default to exp(-2) = 0.135 std
        log_std = torch.clamp(log_std, -5, 0)
        
        # Critic
        value = self.critic(state)
        value = torch.nan_to_num(value, nan=0.0)
        
        # Collision predictor
        collision_logit = self.collision_predictor(state)
        collision_logit = torch.nan_to_num(collision_logit, nan=0.0)
        
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
        
        # Robust NaN handling for distribution parameters
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=-2.0)
        log_std = torch.clamp(log_std, -5, 0)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-6, max=1.0)
        
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        actions_raw = torch.atanh(torch.clamp(actions_unscaled, -0.999, 0.999))
        actions_raw = torch.nan_to_num(actions_raw, nan=0.0)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        log_prob = torch.nan_to_num(log_prob, nan=-10.0)  # Replace NaN with low prob
        entropy = dist.entropy().sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0)
        
        return log_prob, value, entropy, collision_logit


class RelativeTrainer:
    """
    Enhanced PPO Trainer for Relative policy.
    
    Improvements:
    - Variable-length rollouts
    - Reward running statistics
    - Model checkpointing
    """
    
    def __init__(
        self,
        n_trees: int = 50,
        lr: float = 1e-4,  # Reduced for stability with warmup
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,  # Increased for more stable gradients
        ent_coef: float = 0.03,  # Increased for better exploration
        max_grad_norm: float = 0.5,
        min_rollout_steps: int = 128,
        anchor_mode: str = "bbox_center",
        action_mode: str = "polar",
        feature_mode: str = "polar",
        visualize: bool = True
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.env = RelativeTransformerEnvironment(
            n_trees=n_trees, anchor_mode=anchor_mode, 
            action_mode=action_mode, feature_mode=feature_mode
        )
        self.n_trees = n_trees
        self.min_rollout_steps = min_rollout_steps
        self.anchor_mode = anchor_mode
        self.action_mode = action_mode
        self.min_rollout_steps = min_rollout_steps
        
        self.network = RelativeActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Training stability: Warmup scheduler
        self.scheduler = WarmupScheduler(self.optimizer, warmup_steps=1000, base_lr=lr)

        # Observation normalization
        self.obs_normalizer = ObservationNormalizer(momentum=0.99)

        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"Using LR warmup: {1000} steps, base_lr: {lr}")
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = RelativeRolloutBuffer()
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        self.visualize = visualize
        if visualize:
            self.output_dir = Path("relative_training_outputs")
            self.output_dir.mkdir(exist_ok=True)
            self.rewards_history = []
            self.bbox_history = []
            self.trees_history = []
    
    def update_reward_stats(self, rewards: list):
        """Update running reward statistics for normalization."""
        for r in rewards:
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            self.reward_std = np.sqrt(
                (self.reward_std**2 * (self.reward_count - 1) + delta * delta2) / self.reward_count
            ) if self.reward_count > 1 else 1.0
    
    def obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        return {
            "relative_trees": torch.tensor(obs["relative_trees"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "mask": torch.tensor(obs["mask"], dtype=torch.bool, device=self.device).unsqueeze(0),
            "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "collision_features": torch.tensor(obs["collision_features"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "grid": torch.tensor(obs["grid"], dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def collect_rollouts(self) -> list:
        """
        Collect variable-length rollouts until min_steps is reached.
        Returns list of episode results: (reward, bbox, n_placed)
        """
        total_steps = 0
        episode_results = []
        episode_rewards_list = []
        
        obs, _ = self.env.reset()
        episode_reward = 0.0

        while total_steps < self.min_rollout_steps:
            # Normalize observation for stability
            obs_normalized = self.obs_normalizer.normalize(obs)
            obs_t = self.obs_to_tensor(obs_normalized)
            
            with torch.no_grad():
                action, log_prob, value, _ = self.network.get_action(
                    obs_t["relative_trees"], obs_t["scalars"], obs_t["mask"],
                    obs_t["collision_features"], obs_t["grid"]
                )
            
            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Normalize reward (optional, can be disabled)
            # normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
            normalized_reward = reward  # Keep raw rewards for now
            
            episode_reward += reward
            episode_rewards_list.append(reward)
            done = terminated or truncated
            
            # Collision label
            collision_label = 1.0 if info.get("collision", False) else 0.0
            
            self.buffer.add(
                obs, action_np, normalized_reward, done,
                log_prob.item(), value.item(), collision_label
            )
            
            total_steps += 1
            
            if done:
                # Store trees for visualization BEFORE reset
                self.last_placed_trees = list(self.env.placed_trees)
                episode_results.append((episode_reward, info.get("bbox_size", 0), info.get("n_placed", 0)))
                obs, _ = self.env.reset()
                episode_reward = 0.0
            else:
                obs = next_obs
        
        # Update reward statistics
        self.update_reward_stats(episode_rewards_list)
        
        # Return last result if no episode completed
        if not episode_results:
            episode_results.append((episode_reward, 0, 0))
        
        return episode_results
    
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

        # Update learning rate scheduler (after all epochs)
        self.scheduler.step()
    
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
        """Train the agent for n_episodes worth of updates."""
        print(f"Enhanced Relative Training: {n_episodes} updates")
        print(f"Min rollout steps: {self.min_rollout_steps}, Trees: {self.n_trees}")
        print("-" * 60)
        
        best_bbox = float('inf')
        total_episodes = 0
        
        for update in range(n_episodes):
            # Collect variable-length rollouts
            episode_results = self.collect_rollouts()
            
            # Process all completed episodes
            for ep_reward, bbox_size, n_placed in episode_results:
                total_episodes += 1
                
                if self.visualize:
                    self.rewards_history.append(ep_reward)
                    self.bbox_history.append(bbox_size)
                    self.trees_history.append(n_placed)
                
                # Track best
                if bbox_size > 0 and bbox_size < best_bbox:
                    best_bbox = bbox_size
            
            # PPO Update
            returns, advantages = self.compute_gae()
            self.update(returns, advantages)
            self.buffer.clear()
            
            # Use last episode for logging
            ep_reward, bbox_size, n_placed = episode_results[-1]
            
            if (update + 1) % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-50:]) if self.rewards_history else 0
                avg_bbox = np.mean(self.bbox_history[-50:]) if self.bbox_history else 0
                avg_trees = np.mean(self.trees_history[-50:]) if self.trees_history else 0
                current_lr = self.scheduler.get_lr()
                print(f"Update {update+1:5d} | Episodes: {total_episodes:4d} | "
                      f"Reward: {avg_reward:7.2f} | BBox: {avg_bbox:.4f} | Trees: {avg_trees:.1f} | "
                      f"Best: {best_bbox:.4f} | LR: {current_lr:.6f}")
            
            # Save visualization every 10 updates
            if self.visualize and (update + 1) % 10 == 0:
                self.save_visualization(total_episodes)
                
            # Save checkpoint every 200 updates
            if (update + 1) % 200 == 0:
                self.save_model(f"checkpoint_{update+1}.pt")
        
        print("-" * 60)
        print(f"Training completed! Total episodes: {total_episodes}")
        print(f"Best BBox: {best_bbox:.4f}")
        self.save_model("final_model.pt")
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
        }, path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relative Transformer Training")
    parser.add_argument("--n_trees", type=int, default=50)
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anchor_mode", type=str, default="board_center",
                        choices=["board_center", "bbox_center", "last_tree"],
                        help="Anchor: 'board_center' (fixed 5,5), 'bbox_center', or 'last_tree'")
    parser.add_argument("--action_mode", type=str, default="polar",
                        choices=["polar", "cartesian"],
                        help="Action space: 'polar' [r,theta,rot] or 'cartesian' [dx,dy,rot]")
    parser.add_argument("--feature_mode", type=str, default="polar",
                        choices=["polar", "cartesian"],
                        help="Feature space: 'polar' [r,sin,cos] or 'cartesian' [dx,dy]")
    args = parser.parse_args()
    
    print(f"Config: anchor={args.anchor_mode}, action={args.action_mode}, features={args.feature_mode}")
    
    trainer = RelativeTrainer(
        n_trees=args.n_trees,
        lr=args.lr,
        anchor_mode=args.anchor_mode,
        action_mode=args.action_mode,
        feature_mode=args.feature_mode,
        visualize=True
    )
    trainer.train(n_episodes=args.n_episodes)
