# ==================== VISION-ONLY TRAINING (Atari Style) ====================
"""
Sadece 256x256 görsel grid kullanarak PPO training.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import threading
from typing import Dict, Tuple, Optional
from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import environment
from environment import TreePackingEnvironment, scale_factor

# Import vision network
from vision_network import VisionActorCritic, obs_to_tensor


class VisionEnvironment(TreePackingEnvironment):
    """
    Vision-only environment wrapper.
    Returns only NxN grid as observation.
    """
    DEFAULT_GRID_SIZE = 1024
    
    def __init__(self, n_trees: int = 50, max_trees: int = 200, grid_size: int = None):
        super().__init__(n_trees=n_trees, max_trees=max_trees)
        
        # Grid size from parameter or class default
        self.grid_size = grid_size if grid_size is not None else self.DEFAULT_GRID_SIZE
        
        # Override action space to [0, 10]
        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 360.0], dtype=np.float32)
        )
        
        # Simple observation space - just the grid
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.grid_size, self.grid_size), 
            dtype=np.float32
        )
    
    def _get_occupancy_grid(self) -> np.ndarray:
        """
        256x256 occupancy grid with fill ratios.
        Covers [0, 10] x [0, 10] area.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        if not self.placed_trees:
            return grid
        
        try:
            from rasterio.features import rasterize
            from rasterio.transform import from_bounds
            
            # 10x10 oyun alanı [0, 10] x [0, 10]
            norm = float(scale_factor)
            minx = 0.0 * norm
            miny = 0.0 * norm
            maxx = 10.0 * norm
            maxy = 10.0 * norm
            
            # High-res rasterize (2x) for fill ratio - 2048x2048 then downsample
            high_res = self.grid_size * 2  # 2048x2048
            transform = from_bounds(minx, miny, maxx, maxy, high_res, high_res)
            
            shapes = [(tree.polygon, 1.0) for tree in self.placed_trees]
            
            high_res_grid = rasterize(
                shapes,
                out_shape=(high_res, high_res),
                transform=transform,
                fill=0.0,
                dtype=np.float32,
                all_touched=True
            )
            
            # Downsample: 2x2 blocks average = fill ratio
            grid = high_res_grid.reshape(
                self.grid_size, 2, self.grid_size, 2
            ).mean(axis=(1, 3))
            
            grid = np.flipud(grid).copy()
            
        except ImportError:
            pass
        
        return grid
    
    def get_observation(self) -> np.ndarray:
        """Return only the grid observation."""
        return self._get_occupancy_grid()
    
    def reset(self, seed=None, options=None):
        """Reset and return grid observation."""
        super().reset(seed=seed, options=options)
        return self.get_observation(), {}
    
    def step(self, action: np.ndarray):
        """Step and return grid observation."""
        # Use parent step logic but return grid
        obs, reward, terminated, truncated, info = super().step(action)
        return self.get_observation(), reward, terminated, truncated, info


class VisionRolloutBuffer:
    """Simple rollout buffer for vision observations."""
    
    def __init__(self):
        self.grids = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, grid, action, reward, done, log_prob, value):
        self.grids.append(grid.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.grids = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def get_batches(self, batch_size: int, device: str):
        n = len(self.grids)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            
            batch_grids = torch.tensor(
                np.array([self.grids[i] for i in batch_indices]),
                dtype=torch.float32, device=device
            )
            
            batch_actions = np.array([self.actions[i] for i in batch_indices])
            batch_log_probs = np.array([self.log_probs[i] for i in batch_indices])
            
            yield (
                batch_grids,
                torch.tensor(batch_actions, dtype=torch.float32, device=device),
                torch.tensor(batch_log_probs, dtype=torch.float32, device=device),
                batch_indices
            )


class VisionTrainer:
    """PPO Trainer for vision-only network."""
    
    def __init__(
        self,
        n_trees: int = 50,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = None,
        visualize: bool = True
    ):
        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Environment
        self.env = VisionEnvironment(n_trees=n_trees)
        self.n_trees = n_trees
        
        # Network
        self.network = VisionActorCritic().to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Buffer
        self.buffer = VisionRolloutBuffer()
        
        # Visualization
        self.visualize = visualize
        if visualize:
            self.output_dir = Path("vision_training_outputs")
            self.output_dir.mkdir(exist_ok=True)
            self.rewards_history = []
            self.bbox_history = []
    
    def collect_rollouts(self, n_steps: int):
        """Collect experience using current policy."""
        grid = self.env.reset()[0]
        episode_reward = 0.0
        
        for step in range(n_steps):
            grid_tensor = obs_to_tensor(grid, self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(grid_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            
            next_grid, reward, done, truncated, info = self.env.step(action_np)
            
            self.buffer.add(grid, action_np, reward, done, log_prob_np, value_np)
            
            episode_reward += reward
            grid = next_grid
            
            if done or truncated:
                break
        
        bbox_size = self.env.bounding_box_size
        return episode_reward, bbox_size
    
    def compute_gae(self, last_value: float):
        """Compute GAE advantages."""
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
        """PPO update step."""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.n_epochs):
            for batch_grids, batch_actions, batch_old_log_probs, batch_indices in \
                    self.buffer.get_batches(self.batch_size, self.device):
                
                batch_returns = torch.tensor(returns[batch_indices], dtype=torch.float32, device=self.device)
                batch_advantages = torch.tensor(advantages[batch_indices], dtype=torch.float32, device=self.device)
                
                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(batch_grids, batch_actions)
                
                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(-1), batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save_visualization(self, episode: int):
        """Save grid and training metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Grid
        grid = self.env.get_observation()
        axes[0].imshow(grid, cmap='Greens', origin='lower', vmin=0, vmax=1)
        axes[0].set_title(f'Episode {episode}\nTrees: {len(self.env.placed_trees)}')
        axes[0].axis('off')
        
        # Reward history
        if self.rewards_history:
            axes[1].plot(self.rewards_history)
            axes[1].set_title('Episode Rewards')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Reward')
        
        # BBox history
        if self.bbox_history:
            axes[2].plot(self.bbox_history)
            axes[2].set_title('BBox Size')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Size')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'episode_{episode:04d}.png', dpi=100)
        plt.close()
    
    def train(self, n_episodes: int = 1000, update_every: int = 1):
        """Main training loop."""
        print(f"Vision Training: {n_episodes} episodes, {self.n_trees} trees each")
        print(f"Update every {update_every} episodes")
        print("-" * 50)
        
        episode_rewards = []
        episode_bboxes = []
        
        for episode in range(n_episodes):
            # Collect rollouts
            episode_reward, bbox_size = self.collect_rollouts(self.n_trees)
            episode_rewards.append(episode_reward)
            episode_bboxes.append(bbox_size)
            
            # Visualization history
            if self.visualize:
                norm = float(scale_factor)
                self.rewards_history.append(episode_reward)
                self.bbox_history.append(bbox_size / norm)
            
            # Update every N episodes
            if (episode + 1) % update_every == 0:
                # Compute returns and advantages
                with torch.no_grad():
                    grid = self.env.get_observation()
                    grid_tensor = obs_to_tensor(grid, self.device)
                    _, _, last_value = self.network.forward(grid_tensor)
                    last_value = 0.0 if self.buffer.dones[-1] else last_value.item()
                
                returns, advantages = self.compute_gae(last_value)
                
                # Update
                self.update(returns, advantages)
                
                # Clear buffer after update
                self.buffer.clear()
                
                # Log and visualize
                norm = float(scale_factor)
                print(f"Episode {episode + 1:4d} | Reward: {episode_reward:8.2f} | "
                      f"BBox: {bbox_size/norm:.4f} | Trees: {len(self.env.placed_trees)}")
                
                if self.visualize and episode % 10 == 0:
                    self.save_visualization(episode)
        
        print("-" * 50)
        print("Training completed!")
        print(f"Visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vision-Only Tree Packing RL Training')
    
    # Training parameters
    parser.add_argument('--n_trees', type=int, default=50, help='Trees per episode')
    parser.add_argument('--n_episodes', type=int, default=1000, help='Total episodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--update_every', type=int, default=1, help='Update every N episodes')
    
    # Environment parameters  
    parser.add_argument('--grid_size', type=int, default=1024, help='Grid resolution (256, 512, 1024)')
    
    # Training options
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Override grid size in VisionEnvironment
    VisionEnvironment.DEFAULT_GRID_SIZE = args.grid_size
    
    trainer = VisionTrainer(
        n_trees=args.n_trees,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        visualize=not args.no_viz
    )
    trainer.train(n_episodes=args.n_episodes, update_every=args.update_every)
