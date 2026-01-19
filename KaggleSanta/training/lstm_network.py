# ==================== LSTM SEQUENTIAL POLICY ====================
"""
LSTM-based policy that maintains hidden state across steps.
Captures sequential patterns in tree placement.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class LSTMActorCritic(nn.Module):
    """
    LSTM-based Actor-Critic for sequential tree placement.
    
    Maintains hidden state across steps within an episode.
    Uses relative action space: (distance, direction, rotation)
    """
    
    def __init__(
        self,
        input_dim: int = 14,  # last_tree(5) + relative(4) + scalars(5)
        hidden_dim: int = 256,
        num_layers: int = 2,
        action_dim: int = 3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_dim = action_dim
        
        # Action bounds (relative)
        action_low = np.array([0.0, -180.0, 0.0])
        action_high = np.array([3.0, 180.0, 360.0])
        
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2)
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Hidden state
        self.hidden = None
    
    def reset_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Reset LSTM hidden state at episode start."""
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            features: (batch, input_dim) or (batch, seq_len, input_dim)
            hidden: Optional LSTM hidden state
        
        Returns:
            mean, log_std, value, new_hidden
        """
        # Encode features
        x = self.feature_encoder(features)
        
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, 128)
        
        # LSTM
        if hidden is None:
            hidden = self.hidden
        
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # Take last output
        lstm_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Actor
        mean = self.actor_mean(lstm_out)
        log_std = self.actor_log_std(lstm_out)
        log_std = torch.clamp(log_std, -5, 0)
        
        # Critic
        value = self.critic(lstm_out)
        
        return mean, log_std, value, new_hidden
    
    def get_action(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and update hidden state."""
        
        mean, log_std, value, new_hidden = self.forward(features, self.hidden)
        self.hidden = new_hidden
        
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
        features: torch.Tensor,
        actions: torch.Tensor,
        hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        
        mean, log_std, value, _ = self.forward(features, hidden_states)
        std = torch.exp(log_std)
        
        # Unscale actions
        actions_unscaled = (actions - self.action_bias) / self.action_scale
        actions_raw = torch.atanh(torch.clamp(actions_unscaled, -0.999, 0.999))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy
