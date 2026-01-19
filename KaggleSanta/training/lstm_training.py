# ==================== LSTM SEQUENTIAL TRAINING ====================
"""
Training with LSTM-based policy that maintains state across steps.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple, List, Optional
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import TreePackingEnvironment, ChristmasTree, scale_factor
from lstm_network import LSTMActorCritic


class LSTMEnvironment(TreePackingEnvironment):
    """
    Environment for LSTM training with relative actions.
    Returns simple feature vector (no pairwise).
    """
    
    def __init__(self, n_trees: int = 50, max_trees: int = 200):
        super().__init__(n_trees=n_trees, max_trees=max_trees)
        
        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -180.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 180.0, 360.0], dtype=np.float32)
        )
        
        # Feature: last_tree(5) + relative(4) + scalars(5) = 14
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        self.reference_point = (5.0, 5.0)
    
    def _get_features(self) -> np.ndarray:
        """Get feature vector for LSTM input."""
        norm = float(scale_factor)
        n = len(self.placed_trees)
        
        # Last tree features (5) - area removed
        if self.placed_trees:
            last = self.placed_trees[-1]
            c = last.polygon.centroid
            bounds = last.polygon.bounds
            last_tree = [
                c.x / norm / 10.0,
                c.y / norm / 10.0,
                float(last.angle) / 360.0,
                (bounds[2] - bounds[0]) / norm,
                (bounds[3] - bounds[1]) / norm,
            ]
        else:
            last_tree = [0.5, 0.5, 0.0, 0.0, 0.0]
        
        # Relative features between last 2 trees (4)
        if len(self.placed_trees) >= 2:
            last = self.placed_trees[-1]
            prev = self.placed_trees[-2]
            c_last = last.polygon.centroid
            c_prev = prev.polygon.centroid
            
            dx = (c_last.x - c_prev.x) / norm / 10.0
            dy = (c_last.y - c_prev.y) / norm / 10.0
            distance = np.sqrt(dx**2 + dy**2) * 10.0  # scale back
            angle_diff = (float(last.angle) - float(prev.angle)) / 360.0
            
            relative = [dx, dy, distance, angle_diff]
        else:
            relative = [0.0, 0.0, 0.0, 0.0]
        
        # Scalars (5)
        scalars = [
            n / self.max_trees,
            self.bounding_box_size / (10.0 * norm) if n > 0 else 0,
            self.fill_ratio,
            (self.n_trees - self.current_index) / self.n_trees,
            self.total_tree_area / (norm * norm * 100)
        ]
        
        return np.array(last_tree + relative + scalars, dtype=np.float32)
    
    def _relative_to_absolute(self, distance, direction, rotation):
        """Convert relative action to absolute coordinates."""
        if not self.placed_trees:
            ref_x, ref_y = self.reference_point
        else:
            last = self.placed_trees[-1]
            c = last.polygon.centroid
            norm = float(scale_factor)
            ref_x = c.x / norm
            ref_y = c.y / norm
        
        dir_rad = math.radians(direction)
        x = ref_x + distance * math.cos(dir_rad)
        y = ref_y + distance * math.sin(dir_rad)
        
        margin = 0.5
        x = np.clip(x, margin, 10.0 - margin)
        y = np.clip(y, margin, 10.0 - margin)
        rotation = np.clip(rotation, 0.0, 360.0)
        
        return x, y, rotation
    
    def step(self, action: np.ndarray):
        distance, direction, rotation = action[0], action[1], action[2]
        x, y, rot = self._relative_to_absolute(distance, direction, rotation)
        abs_action = np.array([x, y, rot], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = super().step(abs_action)
        return self._get_features(), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_features(), {}


class LSTMRolloutBuffer:
    """Buffer that stores hidden states for LSTM training."""
    
    def __init__(self):
        self.features = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.hidden_h = []  # LSTM hidden states
        self.hidden_c = []
    
    def add(self, features, action, reward, done, log_prob, value, hidden):
        self.features.append(features.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        # Store detached hidden states
        self.hidden_h.append(hidden[0].detach().cpu().numpy())
        self.hidden_c.append(hidden[1].detach().cpu().numpy())
    
    def clear(self):
        self.features = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.hidden_h = []
        self.hidden_c = []
    
    def get_batches(self, batch_size: int, device: str):
        n = len(self.features)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            
            # Reconstruct hidden states
            h = np.array([self.hidden_h[i] for i in batch_idx])
            c = np.array([self.hidden_c[i] for i in batch_idx])
            
            yield {
                "features": torch.tensor(np.array([self.features[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "actions": torch.tensor(np.array([self.actions[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "log_probs": torch.tensor(np.array([self.log_probs[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "hidden": (
                    torch.tensor(h, dtype=torch.float32, device=device).squeeze(2).permute(1, 0, 2).contiguous(),
                    torch.tensor(c, dtype=torch.float32, device=device).squeeze(2).permute(1, 0, 2).contiguous()
                ),
                "indices": batch_idx
            }


class LSTMTrainer:
    """PPO Trainer for LSTM policy."""
    
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
        device: str = None,
        visualize: bool = True
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.env = LSTMEnvironment(n_trees=n_trees)
        self.n_trees = n_trees
        
        self.network = LSTMActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = LSTMRolloutBuffer()
        
        self.visualize = visualize
        if visualize:
            self.output_dir = Path("lstm_training_outputs")
            self.output_dir.mkdir(exist_ok=True)
            self.rewards_history = []
            self.bbox_history = []
    
    def collect_rollouts(self, n_steps: int):
        features, _ = self.env.reset()
        self.network.reset_hidden(batch_size=1, device=self.device)
        episode_reward = 0.0
        
        for step in range(n_steps):
            feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Save hidden before action
            hidden_before = (self.network.hidden[0].clone(), self.network.hidden[1].clone())
            
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(feat_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            
            next_features, reward, done, truncated, info = self.env.step(action_np)
            
            self.buffer.add(features, action_np, reward, done, log_prob_np, value_np, hidden_before)
            
            episode_reward += reward
            features = next_features
            
            if done or truncated:
                break
        
        return episode_reward, self.env.bounding_box_size
    
    def compute_gae(self, last_value: float):
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values + [last_value])
        dones = np.array(self.buffer.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        return returns, advantages
    
    def update(self, returns: np.ndarray, advantages: np.ndarray):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, self.device):
                batch_returns = torch.tensor(returns[batch["indices"]], dtype=torch.float32, device=self.device)
                batch_advantages = torch.tensor(advantages[batch["indices"]], dtype=torch.float32, device=self.device)
                
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch["features"], batch["actions"], batch["hidden"]
                )
                
                ratio = torch.exp(log_probs - batch["log_probs"])
                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                value_loss = nn.functional.mse_loss(values.squeeze(-1), batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save_visualization(self, episode: int):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        norm = float(scale_factor)
        
        ax = axes[0]
        for tree in self.env.placed_trees:
            coords = np.array(tree.polygon.exterior.coords) / norm
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.6, color='green')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title(f'Episode {episode}\nTrees: {len(self.env.placed_trees)}')
        
        if self.rewards_history:
            axes[1].plot(self.rewards_history)
            axes[1].set_title('Episode Rewards')
        
        if self.bbox_history:
            axes[2].plot(self.bbox_history)
            axes[2].set_title('BBox Size')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'episode_{episode:04d}.png', dpi=100)
        plt.close()
    
    def train(self, n_episodes: int = 1000):
        print(f"LSTM Training: {n_episodes} episodes, {self.n_trees} trees each")
        print("-" * 50)
        
        for episode in range(n_episodes):
            episode_reward, bbox_size = self.collect_rollouts(self.n_trees)
            
            with torch.no_grad():
                features = self.env._get_features()
                feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, _, value, _ = self.network.forward(feat_tensor, self.network.hidden)
                last_value = 0.0 if self.buffer.dones[-1] else value.item()
            
            returns, advantages = self.compute_gae(last_value)
            self.update(returns, advantages)
            self.buffer.clear()
            
            if self.visualize:
                norm = float(scale_factor)
                self.rewards_history.append(episode_reward)
                self.bbox_history.append(bbox_size / norm)
            
            if (episode + 1) % 10 == 0:
                norm = float(scale_factor)
                print(f"Episode {episode + 1:4d} | Reward: {episode_reward:8.2f} | "
                      f"BBox: {bbox_size/norm:.4f} | Trees: {len(self.env.placed_trees)}")
                
                if self.visualize:
                    self.save_visualization(episode)
        
        print("-" * 50)
        print("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM Sequential Training')
    parser.add_argument('--n_trees', type=int, default=50)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_viz', action='store_true')
    
    args = parser.parse_args()
    
    trainer = LSTMTrainer(
        n_trees=args.n_trees,
        lr=args.lr,
        batch_size=args.batch_size,
        visualize=not args.no_viz
    )
    trainer.train(n_episodes=args.n_episodes)
