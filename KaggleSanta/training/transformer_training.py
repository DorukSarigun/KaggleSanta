# ==================== TRANSFORMER TRAINING ====================
"""
Training with Transformer-based policy.
Self-attention over all placed trees - no gradient vanishing!
"""

import os
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple, List
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import TreePackingEnvironment, ChristmasTree, scale_factor
from transformer_network import TransformerActorCritic


class TransformerEnvironment(TreePackingEnvironment):
    """
    Environment for Transformer training.
    Uses ABSOLUTE coordinates with full geometry features.
    """
    
    MAX_TREES = 100  # Maximum trees to track
    TREE_DIM = 39    # center (2) + rotation (1) + 15 corners × 2 coords (30) + relative (6) = 39
    
    def __init__(self, n_trees: int = 50, max_trees: int = 200):
        super().__init__(n_trees=n_trees, max_trees=max_trees)
        
        from gymnasium import spaces
        # ABSOLUTE action space: x, y, rotation
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 360.0], dtype=np.float32)
        )
        
        self.observation_space = spaces.Dict({
            "trees": spaces.Box(-np.inf, np.inf, shape=(self.MAX_TREES, self.TREE_DIM), dtype=np.float32),
            "mask": spaces.Box(0, 1, shape=(self.MAX_TREES,), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "collision_features": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "grid": spaces.Box(0, 1, shape=(512, 512), dtype=np.float32),  # Higher res for CNN
            "bbox": spaces.Box(0, 1, shape=(4,), dtype=np.float32)  # [minx, miny, maxx, maxy]
        })
    
    
    def _get_tree_features_full(self, tree, last_tree=None, bbox_center=None) -> np.ndarray:
        """
        Get FULL geometry features for a tree with RELATIVE encoding (NO NORMALIZATION):
        - center (x, y): 2 - raw [0, 10]
        - rotation: 1 - raw [0, 360]
        - 15 corners (x, y each): 30 - raw
        - RELATIVE: dx, dy to last tree: 2 - raw
        - RELATIVE: distance, angle to last tree: 2 - raw
        - RELATIVE: dx, dy to bbox center: 2 - raw
        Total: 39 features
        """
        norm = float(scale_factor)
        c = tree.polygon.centroid
        
        # Center (RAW, [0, 10])
        cx = c.x / norm
        cy = c.y / norm
        
        # Rotation (RAW, [0, 360])
        rot = float(tree.angle)
        
        # All 15 corner coordinates (RAW)
        coords = np.array(tree.polygon.exterior.coords[:-1])  # Skip repeated last point
        corners_raw = coords / norm  # (15, 2) raw
        corners_flat = corners_raw.flatten()  # (30,)
        
        # RELATIVE ENCODING (RAW)
        if last_tree is not None:
            last_c = last_tree.polygon.centroid
            dx = (c.x - last_c.x) / norm  # Raw delta
            dy = (c.y - last_c.y) / norm
            dist = np.sqrt(dx**2 + dy**2)  # Raw distance
            angle = np.arctan2(dy, dx)  # Raw radians [-π, π]
        else:
            dx, dy, dist, angle = 0.0, 0.0, 0.0, 0.0
        
        if bbox_center is not None:
            dx_bbox = cx - bbox_center[0]  # Raw delta
            dy_bbox = cy - bbox_center[1]
        else:
            dx_bbox, dy_bbox = cx - 5.0, cy - 5.0  # Relative to center (5, 5)
        
        features = np.concatenate([
            [cx, cy, rot],           # 3: absolute position (raw)
            corners_flat,            # 30: corner coords (raw)
            [dx, dy, dist, angle],   # 4: relative to last tree (raw)
            [dx_bbox, dy_bbox]       # 2: relative to bbox center (raw)
        ])
        return features.astype(np.float32)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation with FULL geometry tree features + relative encoding + occupancy grid."""
        norm = float(scale_factor)
        n = len(self.placed_trees)
        
        # Compute bbox center for relative encoding (RAW)
        if n > 0:
            from shapely.ops import unary_union
            all_polys = [t.polygon for t in self.placed_trees]
            union = unary_union(all_polys)
            minx, miny, maxx, maxy = union.bounds
            bbox_center = ((minx + maxx) / 2 / norm, (miny + maxy) / 2 / norm)  # Raw
            bbox = np.array([
                minx / norm,
                miny / norm,
                maxx / norm,
                maxy / norm
            ], dtype=np.float32)
        else:
            bbox_center = (5.0, 5.0)  # Center of [0, 10]
            bbox = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Tree features (n_trees, 39) - FULL GEOMETRY + RELATIVE
        trees = np.zeros((self.MAX_TREES, self.TREE_DIM), dtype=np.float32)
        mask = np.zeros(self.MAX_TREES, dtype=np.float32)
        
        for i in range(min(n, self.MAX_TREES)):
            last_tree = self.placed_trees[i-1] if i > 0 else None
            trees[i] = self._get_tree_features_full(self.placed_trees[i], last_tree, bbox_center)
            mask[i] = 1.0
        
        # Scalars (5)
        scalars = np.array([
            n / self.max_trees,
            self.bounding_box_size / (10.0 * norm) if n > 0 else 0,
            self.fill_ratio,
            (self.n_trees - self.current_index) / self.n_trees,
            self.total_tree_area / (norm * norm * 100)
        ], dtype=np.float32)
        
        # Occupancy grid (512, 512) - CNN will learn spatial features
        grid = self._get_occupancy_grid(size=512)

        # Collision features (6,) - NEW!
        collision_features = self._get_collision_features()

        return {
            "trees": trees,
            "mask": mask,
            "scalars": scalars,
            "collision_features": collision_features,
            "grid": grid,
            "bbox": bbox
        }
    
    def get_observation(self):
        """Override to return our observation format."""
        return self._get_observation()
    
    def step(self, action: np.ndarray):
        """Step with ABSOLUTE coordinates."""
        x, y, rotation = action[0], action[1], action[2]
        
        # Clip to valid range
        x = np.clip(x, 0.5, 9.5)
        y = np.clip(y, 0.5, 9.5)
        rotation = np.clip(rotation, 0.0, 360.0)
        
        abs_action = np.array([x, y, rotation], dtype=np.float32)
        
        # Call parent step
        _, reward, terminated, truncated, info = super().step(abs_action)
        return self._get_observation(), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_observation(), {}


class TransformerRolloutBuffer:
    """Rollout buffer for transformer observations."""

    def __init__(self):
        self.trees = []
        self.masks = []
        self.scalars = []
        self.collision_features = []
        self.grids = []
        self.bboxes = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.collision_probs = []
        self.collision_labels = []  # Actual collision occurred (0 or 1)
    
    def add(self, obs: Dict, action, reward, done, log_prob, value, collision_prob, collision_label):
        self.trees.append(obs["trees"].copy())
        self.masks.append(obs["mask"].copy())
        self.scalars.append(obs["scalars"].copy())
        self.collision_features.append(obs["collision_features"].copy())
        self.grids.append(obs["grid"].copy())
        self.bboxes.append(obs["bbox"].copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.collision_probs.append(collision_prob)
        self.collision_labels.append(collision_label)
    
    def clear(self):
        self.trees = []
        self.masks = []
        self.scalars = []
        self.collision_features = []
        self.grids = []
        self.bboxes = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.collision_probs = []
        self.collision_labels = []
    
    def get_batches(self, batch_size: int, device: str):
        n = len(self.actions)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            
            yield {
                "trees": torch.tensor(np.array([self.trees[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "masks": torch.tensor(np.array([self.masks[i] for i in batch_idx]), dtype=torch.bool, device=device),
                "scalars": torch.tensor(np.array([self.scalars[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "collision_features": torch.tensor(np.array([self.collision_features[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "grids": torch.tensor(np.array([self.grids[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "bboxes": torch.tensor(np.array([self.bboxes[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "actions": torch.tensor(np.array([self.actions[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "log_probs": torch.tensor(np.array([self.log_probs[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "collision_labels": torch.tensor(np.array([self.collision_labels[i] for i in batch_idx]), dtype=torch.float32, device=device),
                "indices": batch_idx
            }


class TransformerTrainer:
    """PPO Trainer for Transformer policy."""
    
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
        
        self.env = TransformerEnvironment(n_trees=n_trees)
        self.n_trees = n_trees
        
        self.network = TransformerActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = TransformerRolloutBuffer()
        
        self.visualize = visualize
        if visualize:
            self.output_dir = Path("transformer_training_outputs")
            self.output_dir.mkdir(exist_ok=True)
            self.rewards_history = []
            self.bbox_history = []
    
    def obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        return {
            "trees": torch.tensor(obs["trees"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "mask": torch.tensor(obs["mask"], dtype=torch.bool, device=self.device).unsqueeze(0),
            "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "collision_features": torch.tensor(obs["collision_features"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "grid": torch.tensor(obs["grid"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "bbox": torch.tensor(obs["bbox"], dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def collect_rollouts(self, n_steps: int):
        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(n_steps):
            obs_t = self.obs_to_tensor(obs)

            with torch.no_grad():
                action, log_prob, value, collision_prob = self.network.get_action(
                    obs_t["trees"], obs_t["scalars"], obs_t["mask"],
                    obs_t["grid"], obs_t["bbox"], obs_t["collision_features"]
                )

            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            collision_prob_np = collision_prob.cpu().numpy().item()

            next_obs, reward, done, truncated, info = self.env.step(action_np)

            # Collision label: 1 if collision occurred, 0 otherwise
            collision_label = 1.0 if info.get("collision", False) else 0.0

            self.buffer.add(obs, action_np, reward, done, log_prob_np, value_np,
                          collision_prob_np, collision_label)

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

                log_probs, values, entropy, collision_probs = self.network.evaluate_actions(
                    batch["trees"], batch["scalars"], batch["actions"], batch["masks"],
                    batch["grids"], batch["bboxes"], batch["collision_features"]
                )

                ratio = torch.exp(log_probs - batch["log_probs"])
                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                value_loss = nn.functional.mse_loss(values.squeeze(-1), batch_returns)
                entropy_loss = -entropy.mean()

                # Collision prediction loss (NEW! - auxiliary task)
                # Use BCEWithLogits for numerical stability (no sigmoid needed on network output)
                # Also clamp labels to [0,1] in case of NaN
                collision_labels_safe = torch.clamp(batch["collision_labels"], 0.0, 1.0)
                collision_loss = nn.functional.binary_cross_entropy_with_logits(
                    collision_probs.squeeze(-1),  # raw logits, no sigmoid
                    collision_labels_safe,
                    reduction='mean'
                )

                # Total loss: PPO loss + collision prediction loss
                # Collision loss weight: 0.1 (auxiliary task, should not dominate)
                loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss + 0.1 * collision_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save_visualization(self, episode: int):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        norm = float(scale_factor)
        
        ax = axes[0]
        # Draw trees with placement order numbers
        for i, tree in enumerate(self.env.placed_trees):
            coords = np.array(tree.polygon.exterior.coords) / norm
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.6, color='green', edgecolor='darkgreen', linewidth=0.5)
            centroid = tree.polygon.centroid
            cx, cy = centroid.x / norm, centroid.y / norm
            ax.text(cx, cy, str(i+1), fontsize=6, ha='center', va='center', color='white', fontweight='bold')
        
        # Draw SQUARE bounding box (max dimension) - matches Kaggle scoring
        if self.env.placed_trees:
            from shapely.ops import unary_union
            all_polys = [t.polygon for t in self.env.placed_trees]
            union = unary_union(all_polys)
            minx, miny, maxx, maxy = union.bounds
            minx, miny = minx / norm, miny / norm
            maxx, maxy = maxx / norm, maxy / norm
            
            # Square bbox: use max dimension
            width = maxx - minx
            height = maxy - miny
            bbox_size = max(width, height)
            
            # Center the square on the actual bbox center
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            half = bbox_size / 2
            
            bbox = plt.Rectangle((cx - half, cy - half), bbox_size, bbox_size, 
                                  fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(bbox)
            ax.set_title(f'Episode {episode} | Trees: {len(self.env.placed_trees)} | BBox: {bbox_size:.2f}')
        else:
            ax.set_title(f'Episode {episode} | Trees: 0')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if self.rewards_history:
            axes[1].plot(self.rewards_history)
            axes[1].set_title('Episode Rewards')
            axes[1].grid(True, alpha=0.3)
        
        if self.bbox_history:
            axes[2].plot(self.bbox_history)
            axes[2].set_title('BBox Size')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'episode_{episode:04d}.png', dpi=100)
        plt.close()
    
    def train(self, n_episodes: int = 1000):
        print(f"Transformer Training: {n_episodes} episodes, {self.n_trees} trees each")
        print("-" * 50)
        
        for episode in range(n_episodes):
            episode_reward, bbox_size = self.collect_rollouts(self.n_trees)
            
            with torch.no_grad():
                obs = self.env._get_observation()
                obs_t = self.obs_to_tensor(obs)
                _, _, value, _ = self.network.forward(
                    obs_t["trees"], obs_t["scalars"], obs_t["mask"],
                    obs_t["grid"], obs_t["bbox"], obs_t["collision_features"]
                )
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
    
    parser = argparse.ArgumentParser(description='Transformer Training')
    parser.add_argument('--n_trees', type=int, default=50)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_viz', action='store_true')
    
    args = parser.parse_args()
    
    trainer = TransformerTrainer(
        n_trees=args.n_trees,
        lr=args.lr,
        batch_size=args.batch_size,
        visualize=not args.no_viz
    )
    trainer.train(n_episodes=args.n_episodes)
