"""
PPO Training Loop with Live Visualization for Tree Packing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
                next_done = 0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
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
            
            # Stack observations
            batch_obs = {
                key: torch.stack([torch.tensor(self.observations[i][key], dtype=torch.float32) 
                                  for i in batch_indices]).to(device)
                for key in self.observations[0].keys()
            }
            
            yield (
                batch_obs,
                torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.float32, device=device),
                torch.tensor([self.log_probs[i] for i in batch_indices], dtype=torch.float32, device=device),
                batch_indices
            )


class LiveVisualizer:
    """Matplotlib ile canlı ağaç yerleştirme görselleştirmesi."""
    
    def __init__(self, update_frequency: int = 1):
        self.update_frequency = update_frequency
        self.step_count = 0
        
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sol: Ağaç yerleştirme
        self.ax_trees = self.axes[0]
        self.ax_trees.set_title('Ağaç Yerleştirme')
        self.ax_trees.set_aspect('equal')
        
        # Sağ: Training metrikleri
        self.ax_metrics = self.axes[1]
        self.ax_metrics.set_title('Training Metrics')
        
        self.rewards_history = deque(maxlen=100)
        self.bbox_history = deque(maxlen=100)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update(self, env: TreePackingEnvironment, episode_reward: float = None, 
               bbox_size: float = None, episode: int = 0):
        """Görselleştirmeyi güncelle."""
        self.step_count += 1
        
        if self.step_count % self.update_frequency != 0:
            return
        
        # Sol panel: Ağaçlar
        self.ax_trees.clear()
        self.ax_trees.set_title(f'Episode {episode} - Trees: {len(env.placed_trees)}')
        
        norm = float(scale_factor)
        
        for i, tree in enumerate(env.placed_trees):
            # Koordinatları normalize et
            coords = np.array(tree.polygon.exterior.coords) / norm
            self.ax_trees.fill(coords[:, 0], coords[:, 1], alpha=0.6, color='green')
            self.ax_trees.plot(coords[:, 0], coords[:, 1], color='darkgreen', linewidth=0.5)
        
        # Bounding box
        if env.placed_trees:
            union = unary_union([t.polygon for t in env.placed_trees])
            bounds = np.array(union.bounds) / norm
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            side = max(width, height)
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            
            rect = Rectangle(
                (cx - side/2, cy - side/2), side, side,
                fill=False, edgecolor='red', linewidth=2, linestyle='--'
            )
            self.ax_trees.add_patch(rect)
        
        self.ax_trees.set_aspect('equal')
        self.ax_trees.autoscale()
        
        # Sağ panel: Metrikler
        if episode_reward is not None:
            self.rewards_history.append(episode_reward)
        if bbox_size is not None:
            self.bbox_history.append(bbox_size / norm)
        
        if len(self.rewards_history) > 0:
            self.ax_metrics.clear()
            self.ax_metrics.set_title('Training Metrics')
            
            ax1 = self.ax_metrics
            ax1.plot(list(self.rewards_history), 'b-', label='Episode Reward')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward', color='b')
            ax1.legend(loc='upper left')
            
            if len(self.bbox_history) > 0:
                ax2 = ax1.twinx()
                ax2.plot(list(self.bbox_history), 'r-', label='BBox Size')
                ax2.set_ylabel('BBox Size', color='r')
                ax2.legend(loc='upper right')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.close()


class PPOTrainer:
    """PPO trainer with live visualization."""
    
    def __init__(self, 
                 n_trees: int = 10,
                 max_trees: int = 200,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 4,
                 batch_size: int = 64,
                 device: str = 'cpu',
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
        self.device = device
        
        # Environment
        self.env = TreePackingEnvironment(n_trees=n_trees, max_trees=max_trees)
        
        # Network
        self.network = create_network({"max_trees": max_trees}).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Buffer
        self.buffer = RolloutBuffer()
        
        # Visualization
        self.visualize = visualize
        if visualize:
            self.viz = LiveVisualizer(update_frequency=1)
    
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
    
    def train(self, n_episodes: int = 100):
        """Ana training loop."""
        print(f"Training started: {n_episodes} episodes, {self.n_trees} trees each")
        print("-" * 50)
        
        for episode in range(n_episodes):
            # Collect rollouts
            episode_reward, bbox_size = self.collect_rollouts(self.n_trees)
            
            # Compute returns and advantages
            with torch.no_grad():
                obs = self.env.get_observation()
                obs_tensor = {
                    key: torch.tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
                    for key, value in obs.items()
                }
                _, _, last_value = self.network.forward(obs_tensor)
                last_value = last_value.item() if not self.env.current_index >= self.n_trees else 0
            
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
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    trainer = PPOTrainer(
        n_trees=10,          # Her episode'da 10 ağaç
        lr=3e-4,
        n_epochs=4,
        batch_size=32,
        visualize=True       # Canlı görselleştirme
    )
    
    trainer.train(n_episodes=100)
