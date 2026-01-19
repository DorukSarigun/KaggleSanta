# ==================== RELATIVE ACTION TRAINING ====================
"""
Training with relative action space and pairwise observations.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple, List
from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import TreePackingEnvironment, ChristmasTree, scale_factor
from relative_network import RelativeActorCritic


class RelativeEnvironment(TreePackingEnvironment):
    """
    Environment with relative action space.
    
    Action: (distance, direction, rotation)
        - distance: 0.5-3.0 birim (son ağaçtan uzaklık)
        - direction: -180 to 180 derece (son ağaca göre yön)
        - rotation: 0-360 derece (ağacın rotasyonu)
    
    Observation:
        - pairwise_features: Ağaç çiftleri özellikleri
        - last_tree: Son ağaç bilgisi
        - scalars: Global metrikler
    """
    
    MAX_PAIRS = 100  # Maximum pairs to consider
    
    def __init__(self, n_trees: int = 50, max_trees: int = 200):
        super().__init__(n_trees=n_trees, max_trees=max_trees)
        
        # Override action space - relative
        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -180.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 180.0, 360.0], dtype=np.float32)
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            "pairwise_features": spaces.Box(-np.inf, np.inf, shape=(self.MAX_PAIRS, 8), dtype=np.float32),
            "pair_mask": spaces.Box(0, 1, shape=(self.MAX_PAIRS,), dtype=np.float32),
            "last_tree": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        })
        
        # Reference point for first tree
        self.reference_point = (5.0, 5.0)  # Center of play area
    
    def _get_pairwise_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise features for all tree pairs.
        
        Returns:
            features: (MAX_PAIRS, 8) - pairwise features
            mask: (MAX_PAIRS,) - 1 for valid pairs, 0 for padding
        """
        features = np.zeros((self.MAX_PAIRS, 8), dtype=np.float32)
        mask = np.zeros(self.MAX_PAIRS, dtype=np.float32)
        
        n = len(self.placed_trees)
        if n < 2:
            return features, mask
        
        norm = float(scale_factor)
        pair_idx = 0
        
        # Compute features for each pair
        for i in range(n):
            for j in range(i + 1, n):
                if pair_idx >= self.MAX_PAIRS:
                    break
                
                tree_i = self.placed_trees[i]
                tree_j = self.placed_trees[j]
                
                # Get centers
                ci = tree_i.polygon.centroid
                cj = tree_j.polygon.centroid
                
                xi, yi = ci.x / norm, ci.y / norm
                xj, yj = cj.x / norm, cj.y / norm
                
                # Features
                dx = xj - xi
                dy = yj - yi
                distance = np.sqrt(dx**2 + dy**2)
                direction = np.arctan2(dy, dx)  # radians
                
                angle_i = float(tree_i.angle) / 360.0
                angle_j = float(tree_j.angle) / 360.0
                angle_diff = (float(tree_j.angle) - float(tree_i.angle)) / 360.0
                
                # Edge distance (normalized)
                edge_dist = tree_i.polygon.distance(tree_j.polygon) / norm
                
                features[pair_idx] = [
                    dx / 10.0,           # normalized dx
                    dy / 10.0,           # normalized dy
                    distance / 10.0,     # normalized distance
                    direction / np.pi,   # normalized direction [-1, 1]
                    angle_i,             # angle of tree i
                    angle_j,             # angle of tree j
                    angle_diff,          # angle difference
                    np.clip(edge_dist, 0, 5) / 5.0  # normalized edge distance
                ]
                mask[pair_idx] = 1.0
                pair_idx += 1
            
            if pair_idx >= self.MAX_PAIRS:
                break
        
        return features, mask
    
    def _get_last_tree_features(self) -> np.ndarray:
        """
        Get features of the last placed tree.
        
        Returns:
            (6,) array: [x, y, angle, width, height, area]
        """
        if not self.placed_trees:
            # No trees yet - return reference point
            return np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        last = self.placed_trees[-1]
        norm = float(scale_factor)
        
        c = last.polygon.centroid
        bounds = last.polygon.bounds
        
        return np.array([
            c.x / norm / 10.0,                    # normalized x
            c.y / norm / 10.0,                    # normalized y
            float(last.angle) / 360.0,            # normalized angle
            (bounds[2] - bounds[0]) / norm,       # width
            (bounds[3] - bounds[1]) / norm,       # height
            last.polygon.area / (norm * norm)     # normalized area
        ], dtype=np.float32)
    
    def _get_scalars(self) -> np.ndarray:
        """
        Get global scalar features.
        
        Returns:
            (5,) array
        """
        norm = float(scale_factor)
        n = len(self.placed_trees)
        
        return np.array([
            n / self.max_trees,                                    # progress
            self.bounding_box_size / (10.0 * norm) if n > 0 else 0,  # bbox size
            self.fill_ratio,                                        # fill ratio
            (self.n_trees - self.current_index) / self.n_trees,     # remaining ratio
            self.total_tree_area / (norm * norm * 100)              # total area
        ], dtype=np.float32)
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get full observation dictionary."""
        pairwise, mask = self._get_pairwise_features()
        
        return {
            "pairwise_features": pairwise,
            "pair_mask": mask,
            "last_tree": self._get_last_tree_features(),
            "scalars": self._get_scalars()
        }
    
    def _relative_to_absolute(self, distance: float, direction: float, rotation: float) -> Tuple[float, float, float]:
        """
        Convert relative action to absolute coordinates.
        
        Args:
            distance: Distance from last tree (birim)
            direction: Angle from last tree (derece)
            rotation: Tree's own rotation (derece)
        
        Returns:
            (x, y, rotation) absolute coordinates
        """
        if not self.placed_trees:
            # First tree - use reference point
            ref_x, ref_y = self.reference_point
        else:
            # Use last tree's center
            last = self.placed_trees[-1]
            c = last.polygon.centroid
            norm = float(scale_factor)
            ref_x = c.x / norm
            ref_y = c.y / norm
        
        # Convert direction to radians
        dir_rad = math.radians(direction)
        
        # Compute new position
        x = ref_x + distance * math.cos(dir_rad)
        y = ref_y + distance * math.sin(dir_rad)
        
        # Clip to valid range
        margin = 0.5
        x = np.clip(x, margin, 10.0 - margin)
        y = np.clip(y, margin, 10.0 - margin)
        rotation = np.clip(rotation, 0.0, 360.0)
        
        return x, y, rotation
    
    def step(self, action: np.ndarray):
        """
        Step with relative action.
        
        Args:
            action: [distance, direction, rotation]
        """
        distance, direction, rotation = action[0], action[1], action[2]
        
        # Convert to absolute coordinates
        x, y, rot = self._relative_to_absolute(distance, direction, rotation)
        
        # Create absolute action and call parent step
        abs_action = np.array([x, y, rot], dtype=np.float32)
        
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(abs_action)
        
        # Return our observation format
        return self.get_observation(), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed, options=options)
        return self.get_observation(), {}


class RelativeRolloutBuffer:
    """Rollout buffer for relative observations."""
    
    def __init__(self):
        self.pairwise = []
        self.masks = []
        self.last_trees = []
        self.scalars = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, obs: Dict, action, reward, done, log_prob, value):
        self.pairwise.append(obs["pairwise_features"].copy())
        self.masks.append(obs["pair_mask"].copy())
        self.last_trees.append(obs["last_tree"].copy())
        self.scalars.append(obs["scalars"].copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.pairwise = []
        self.masks = []
        self.last_trees = []
        self.scalars = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def get_batches(self, batch_size: int, device: str):
        n = len(self.actions)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            
            yield {
                "pairwise": torch.tensor(np.array([self.pairwise[i] for i in batch_indices]), dtype=torch.float32, device=device),
                "masks": torch.tensor(np.array([self.masks[i] for i in batch_indices]), dtype=torch.bool, device=device),
                "last_trees": torch.tensor(np.array([self.last_trees[i] for i in batch_indices]), dtype=torch.float32, device=device),
                "scalars": torch.tensor(np.array([self.scalars[i] for i in batch_indices]), dtype=torch.float32, device=device),
                "actions": torch.tensor(np.array([self.actions[i] for i in batch_indices]), dtype=torch.float32, device=device),
                "log_probs": torch.tensor(np.array([self.log_probs[i] for i in batch_indices]), dtype=torch.float32, device=device),
                "indices": batch_indices
            }


class RelativeTrainer:
    """PPO Trainer for relative action space."""
    
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
        
        self.env = RelativeEnvironment(n_trees=n_trees)
        self.n_trees = n_trees
        
        self.network = RelativeActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = RelativeRolloutBuffer()
        
        self.visualize = visualize
        if visualize:
            self.output_dir = Path("relative_training_outputs")
            self.output_dir.mkdir(exist_ok=True)
            self.rewards_history = []
            self.bbox_history = []
    
    def obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        return {
            "pairwise": torch.tensor(obs["pairwise_features"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "mask": torch.tensor(obs["pair_mask"], dtype=torch.bool, device=self.device).unsqueeze(0),
            "last_tree": torch.tensor(obs["last_tree"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def collect_rollouts(self, n_steps: int):
        obs, _ = self.env.reset()
        episode_reward = 0.0
        
        for step in range(n_steps):
            obs_t = self.obs_to_tensor(obs)
            
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(
                    obs_t["pairwise"], obs_t["last_tree"], obs_t["scalars"], obs_t["mask"]
                )
            
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            
            next_obs, reward, done, truncated, info = self.env.step(action_np)
            
            self.buffer.add(obs, action_np, reward, done, log_prob_np, value_np)
            
            episode_reward += reward
            obs = next_obs
            
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
                    batch["pairwise"], batch["last_trees"], batch["scalars"],
                    batch["actions"], batch["masks"]
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
        from shapely.ops import unary_union
        from matplotlib.patches import Rectangle
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        norm = float(scale_factor)
        
        # Trees
        ax = axes[0]
        for tree in self.env.placed_trees:
            coords = np.array(tree.polygon.exterior.coords) / norm
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.6, color='green')
            ax.plot(coords[:, 0], coords[:, 1], color='darkgreen', linewidth=0.5)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title(f'Episode {episode}\nTrees: {len(self.env.placed_trees)}')
        
        # Rewards
        if self.rewards_history:
            axes[1].plot(self.rewards_history)
            axes[1].set_title('Episode Rewards')
            axes[1].set_xlabel('Episode')
        
        # BBox
        if self.bbox_history:
            axes[2].plot(self.bbox_history)
            axes[2].set_title('BBox Size')
            axes[2].set_xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'episode_{episode:04d}.png', dpi=100)
        plt.close()
    
    def train(self, n_episodes: int = 1000):
        print(f"Relative Training: {n_episodes} episodes, {self.n_trees} trees each")
        print("-" * 50)
        
        for episode in range(n_episodes):
            episode_reward, bbox_size = self.collect_rollouts(self.n_trees)
            
            with torch.no_grad():
                obs = self.env.get_observation()
                obs_t = self.obs_to_tensor(obs)
                _, _, last_value = self.network.forward(
                    obs_t["pairwise"], obs_t["last_tree"], obs_t["scalars"], obs_t["mask"]
                )
                last_value = 0.0 if self.buffer.dones[-1] else last_value.item()
            
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
    
    parser = argparse.ArgumentParser(description='Relative Action Space Training')
    parser.add_argument('--n_trees', type=int, default=50)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_viz', action='store_true')
    
    args = parser.parse_args()
    
    trainer = RelativeTrainer(
        n_trees=args.n_trees,
        lr=args.lr,
        batch_size=args.batch_size,
        visualize=not args.no_viz
    )
    trainer.train(n_episodes=args.n_episodes)
