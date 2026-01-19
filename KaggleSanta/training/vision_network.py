# ==================== VISION-ONLY NETWORK (Atari Style) ====================
"""
Sadece 256x256 görsel grid kullanarak öğrenen network.
Atari oyunlarındaki DQN benzeri mimari.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class UNetEncoder(nn.Module):
    """
    U-Net style encoder for 1024x1024 grid.
    Multi-scale feature extraction with skip connections.
    """
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        
        # Encoder path (downsampling) - 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8
        
        # Level 1: 1024 -> 512
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        # Level 2: 512 -> 256
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Level 3: 256 -> 128
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Level 4: 128 -> 64
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Level 5: 64 -> 32
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Level 6: 32 -> 16
        self.enc6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Bottleneck: 16 -> 8
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Global pooling + FC
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, 1024, 1024) fill ratio grid
        Returns:
            (batch, embed_dim)
        """
        x = grid.unsqueeze(1)  # (batch, 1, 1024, 1024)
        
        x = self.enc1(x)   # (batch, 16, 512, 512)
        x = self.enc2(x)   # (batch, 32, 256, 256)
        x = self.enc3(x)   # (batch, 64, 128, 128)
        x = self.enc4(x)   # (batch, 128, 64, 64)
        x = self.enc5(x)   # (batch, 256, 32, 32)
        x = self.enc6(x)   # (batch, 512, 16, 16)
        x = self.bottleneck(x)  # (batch, 512, 8, 8)
        
        x = self.global_pool(x)  # (batch, 512, 4, 4)
        return self.fc(x)


# Alias for backward compatibility
VisionEncoder = UNetEncoder


class VisionActor(nn.Module):
    """Actor network for continuous actions."""
    
    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()
        
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Output in [-1, 1], scaled later
        )
        
        # State-dependent log_std
        self.log_std_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, -5, 0)  # Limit exploration
        return mean, log_std


class VisionCritic(nn.Module):
    """Critic network for value estimation."""
    
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


class VisionActorCritic(nn.Module):
    """
    Vision-only Actor-Critic network.
    Input: 256x256 occupancy grid with fill ratios
    Output: Action (x, y, rotation) + Value
    """
    
    def __init__(
        self,
        action_dim: int = 3,
        embed_dim: int = 512,
        action_low: np.ndarray = None,
        action_high: np.ndarray = None
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Action bounds - 10x10 oyun alanı
        if action_low is None:
            action_low = np.array([0.0, 0.0, 0.0])
        if action_high is None:
            action_high = np.array([10.0, 10.0, 360.0])
        
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2)
        
        # Vision encoder (CNN)
        self.vision_encoder = VisionEncoder(embed_dim)
        
        # Actor and Critic
        self.actor = VisionActor(embed_dim, action_dim)
        self.critic = VisionCritic(embed_dim)
    
    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grid: (batch, 256, 256) fill ratio grid
        Returns:
            mean: (batch, action_dim)
            log_std: (batch, action_dim)
            value: (batch, 1)
        """
        # Encode visual observation
        state = self.vision_encoder(grid)
        
        # Get action distribution
        mean, log_std = self.actor(state)
        
        # Get value
        value = self.critic(state)
        
        return mean, log_std, value
    
    def get_action(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            grid: (batch, 256, 256) grid
        Returns:
            action: (batch, action_dim) scaled action
            log_prob: (batch,) log probability
            value: (batch, 1) state value
        """
        mean, log_std, value = self.forward(grid)
        
        # Sample from Gaussian
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.rsample()  # Reparameterization trick
        
        # Log probability
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        
        # Scale action to actual range
        action = torch.tanh(action_raw)
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob, value
    
    def evaluate_actions(self, grid: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            grid: (batch, 256, 256)
            actions: (batch, action_dim) scaled actions
        Returns:
            log_prob: (batch,)
            value: (batch, 1)
            entropy: (batch,)
        """
        mean, log_std, value = self.forward(grid)
        std = torch.exp(log_std)
        
        # Unscale actions to raw space
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        actions_raw = torch.atanh(torch.clamp(actions_unscaled, -0.999, 0.999))
        
        # Compute log prob and entropy
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


def obs_to_tensor(obs: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy grid observation to torch tensor.
    
    Args:
        obs: (256, 256) numpy array
        device: 'cpu' or 'cuda'
    Returns:
        (1, 256, 256) tensor
    """
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
