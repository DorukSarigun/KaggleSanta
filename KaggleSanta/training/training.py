"""
PPO Training Loop with Async Visualization for Tree Packing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import threading
from pathlib import Path

# Use non-interactive backend for file saving
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
from typing import Dict, List, Tuple
import time

from environment import TreePackingEnvironment, scale_factor
from networks import ActorCritic, obs_to_tensor, create_network
from shapely.ops import unary_union


class RolloutBuffer:
    """PPO için deneyimleri toplayan buffer."""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99,
                                        gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """GAE ile returns ve advantages hesaplar."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
                # FIX: Use current step's done flag, not 0
                next_done = dones[t]
            else:
                next_value = values[t + 1]
                # FIX: Use current step's done flag
                next_done = dones[t]

            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae

        returns = advantages + values
        return returns, advantages
    
    def get_batches(self, batch_size: int, device: str = 'cpu'):
        """Mini-batch generator."""
        n_samples = len(self.observations)
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # Stack observations - convert to numpy array first for performance
            batch_obs = {
                key: torch.from_numpy(
                    np.stack([self.observations[i][key] for i in batch_indices])
                ).float().to(device)
                for key in self.observations[0].keys()
            }
            
            # Convert actions and log_probs to numpy arrays first
            batch_actions = np.stack([self.actions[i] for i in batch_indices])
            batch_log_probs = np.array([self.log_probs[i] for i in batch_indices])
            
            yield (
                batch_obs,
                torch.from_numpy(batch_actions).float().to(device),
                torch.from_numpy(batch_log_probs).float().to(device),
                batch_indices
            )


class AsyncVisualizer:
    """Async file-based visualization - saves images without blocking training."""
    
    def __init__(self, output_dir: str = "training_outputs", save_every: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_every = save_every
        
        self.rewards_history = deque(maxlen=100)
        self.bbox_history = deque(maxlen=100)
        
        # Thread pool for async saving
        self._pending_threads = []
    
    def update(self, env: TreePackingEnvironment, episode_reward: float = None, 
               bbox_size: float = None, episode: int = 0):
        """Schedule async image save."""
        if episode_reward is not None:
            self.rewards_history.append(episode_reward)
        if bbox_size is not None:
            norm = float(scale_factor)
            self.bbox_history.append(bbox_size / norm)
        
        # Only save every N episodes
        if episode % self.save_every != 0:
            return
        
        # Copy data for thread safety
        tree_data = []
        for tree in env.placed_trees:
            coords = np.array(tree.polygon.exterior.coords).copy()
            tree_data.append(coords)
        
        bbox_bounds = None
        if env.placed_trees:
            union = unary_union([t.polygon for t in env.placed_trees])
            bbox_bounds = np.array(union.bounds).copy()
        
        rewards_copy = list(self.rewards_history)
        bbox_copy = list(self.bbox_history)
        
        # Launch background save thread
        thread = threading.Thread(
            target=self._save_image,
            args=(tree_data, bbox_bounds, rewards_copy, bbox_copy, episode),
            daemon=True
        )
        thread.start()
        self._pending_threads.append(thread)
        
        # Clean up finished threads
        self._pending_threads = [t for t in self._pending_threads if t.is_alive()]
    
    def _save_image(self, tree_data, bbox_bounds, rewards, bbox_sizes, episode):
        """Save visualization to file (runs in background thread)."""
        try:
            norm = float(scale_factor)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left panel: Trees
            ax_trees = axes[0]
            ax_trees.set_title(f'Episode {episode} - Trees: {len(tree_data)}')
            
            for coords in tree_data:
                coords_norm = coords / norm
                ax_trees.fill(coords_norm[:, 0], coords_norm[:, 1], alpha=0.6, color='green')
                ax_trees.plot(coords_norm[:, 0], coords_norm[:, 1], color='darkgreen', linewidth=0.5)
            
            # Bounding box
            if bbox_bounds is not None:
                bounds = bbox_bounds / norm
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                side = max(width, height)
                cx = (bounds[0] + bounds[2]) / 2
                cy = (bounds[1] + bounds[3]) / 2
                
                rect = Rectangle(
                    (cx - side/2, cy - side/2), side, side,
                    fill=False, edgecolor='red', linewidth=2, linestyle='--'
                )
                ax_trees.add_patch(rect)
            
            ax_trees.set_aspect('equal')
            ax_trees.autoscale()
            
            # Right panel: Metrics
            ax_metrics = axes[1]
            ax_metrics.set_title('Training Metrics')
            
            if len(rewards) > 0:
                ax_metrics.plot(rewards, 'b-', label='Episode Reward')
                ax_metrics.set_xlabel('Episode')
                ax_metrics.set_ylabel('Reward', color='b')
                ax_metrics.legend(loc='upper left')
                
                if len(bbox_sizes) > 0:
                    ax2 = ax_metrics.twinx()
                    ax2.plot(bbox_sizes, 'r-', label='BBox Size')
                    ax2.set_ylabel('BBox Size', color='r')
                    ax2.legend(loc='upper right')
            
            plt.tight_layout()
            
            # Save to file
            filepath = self.output_dir / f"episode_{episode:04d}.png"
            fig.savefig(filepath, dpi=100)
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Failed to save visualization: {e}")
    
    def close(self):
        """Wait for pending saves to complete."""
        for thread in self._pending_threads:
            thread.join(timeout=5.0)
        print(f"Visualizations saved to: {self.output_dir.absolute()}")


class PPOTrainer:
    """PPO trainer with live visualization."""
    
    def __init__(self,
                 n_trees: int = 10,
                 max_trees: int = 200,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.05,  # FIX: Increased for better exploration
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 4,
                 batch_size: int = 64,
                 device: str = None,  # Auto-detect GPU if available
                 visualize: bool = True):

        self.n_trees = n_trees
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Auto-detect device (CUDA GPU if available, else CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Enable cuDNN benchmarking for better GPU performance
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

        # Environment
        self.env = TreePackingEnvironment(n_trees=n_trees, max_trees=max_trees)

        # Network
        self.network = create_network({"max_trees": max_trees}).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Buffer
        self.buffer = RolloutBuffer()
        
        # Visualization
        self.visualize = visualize
        if visualize:
            self.viz = AsyncVisualizer(output_dir="training_outputs", save_every=1)
    
    def collect_rollouts(self, n_steps: int) -> Tuple[float, float]:
        """Bir episode boyunca deneyim topla."""
        obs, _ = self.env.reset()
        episode_reward = 0
        
        for step in range(n_steps):
            # Convert observation
            obs_tensor = {
                key: torch.tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
                for key, value in obs.items()
            }
            
            # Get action
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(obs_tensor)
            
            action_np = action.squeeze(0).cpu().numpy()
            log_prob_np = log_prob.item()
            value_np = value.item()
            
            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action_np)
            episode_reward += reward
            
            # Store
            self.buffer.add(obs, action_np, reward, value_np, log_prob_np, done)
            
            # Visualize
            if self.visualize:
                self.viz.update(self.env)
            
            obs = next_obs
            
            if done:
                break
        
        # Last value for GAE
        with torch.no_grad():
            obs_tensor = {
                key: torch.tensor(obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                for key in obs.keys()
            }
            _, _, last_value = self.network.forward(obs_tensor)
            last_value = last_value.item()
        
        return episode_reward, self.env.bounding_box_size
    
    def update(self, returns: np.ndarray, advantages: np.ndarray):
        """PPO policy update."""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            for batch_obs, batch_actions, batch_old_log_probs, batch_indices in \
                    self.buffer.get_batches(self.batch_size, self.device):
                
                batch_returns = torch.tensor(returns[batch_indices], dtype=torch.float32, device=self.device)
                batch_advantages = torch.tensor(advantages[batch_indices], dtype=torch.float32, device=self.device)
                
                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(batch_obs, batch_actions)
                
                # Policy loss (PPO clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def train(self, n_episodes: int = 1_000):
        """Ana training loop."""
        print(f"Training started: {n_episodes} episodes, {self.n_trees} trees each")
        print("-" * 50)
        
        for episode in range(n_episodes):
            # Collect rollouts
            episode_reward, bbox_size = self.collect_rollouts(self.n_trees)

            # FIX: Compute last_value correctly based on terminal state
            with torch.no_grad():
                # Check if last step was terminal
                if len(self.buffer.dones) > 0 and self.buffer.dones[-1]:
                    # Episode ended, terminal state has value 0
                    last_value = 0.0
                else:
                    # Episode not done, estimate value of next state
                    obs = self.env.get_observation()
                    obs_tensor = {
                        key: torch.tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key, value in obs.items()
                    }
                    _, _, last_value_tensor = self.network.forward(obs_tensor)
                    last_value = last_value_tensor.item()

            returns, advantages = self.buffer.compute_returns_and_advantages(
                last_value, self.gamma, self.gae_lambda
            )
            
            # Update policy
            self.update(returns, advantages)
            
            # Update visualization
            if self.visualize:
                self.viz.update(self.env, episode_reward, bbox_size, episode)
            
            # Logging
            if (episode + 1) % 10 == 0:
                norm = float(scale_factor)
                print(f"Episode {episode + 1:4d} | Reward: {episode_reward:8.2f} | "
                      f"BBox: {bbox_size/norm:.4f} | Trees: {len(self.env.placed_trees)}")
            
            # Clear buffer
            self.buffer.clear()
        
        print("-" * 50)
        print("Training completed!")
        
        if self.visualize:
            self.viz.close()
    
    def save(self, path: str):
        """Model kaydet."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Model yükle."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path} to {self.device}")


if __name__ == "__main__":
    trainer = PPOTrainer(
        n_trees=50,          # Her episode'da 10 ağaç
        lr=3e-6,
        n_epochs=20,
        batch_size=32,
        device=None,         # Auto-detect GPU (cuda if available, else cpu)
        visualize=True       # Canlı görselleştirme
    )

    trainer.train(n_episodes=100)
