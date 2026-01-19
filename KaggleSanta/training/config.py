"""
Hyperparameter Configuration for Tree Packing RL System.

Dataclass-based configuration for easy experimentation and reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class EnvironmentConfig:
    """Environment hyperparameters."""
    n_trees: int = 50                    # Number of trees per episode
    max_trees: int = 200                 # Maximum trees (for padding)
    grid_size: int = 256                 # Occupancy grid resolution
    k_neighbors: int = 3                 # Number of nearest neighbors for pairwise features
    action_low: List[float] = field(default_factory=lambda: [-10.0, -10.0, 0.0])
    action_high: List[float] = field(default_factory=lambda: [10.0, 10.0, 360.0])


@dataclass
class NetworkConfig:
    """Neural network hyperparameters."""
    # Feature dimensions
    tree_feature_dim: int = 10
    pairwise_feature_dim: int = 30       # k=3 neighbors * 10 features
    scalar_dim: int = 5
    free_space_dim: int = 4

    # Embedding dimensions
    embed_dim: int = 64
    state_dim: int = 256
    scalar_embed_dim: int = 32
    free_space_embed_dim: int = 32
    grid_embed_dim: int = 64

    # Transformer architecture
    n_heads: int = 8                     # Number of attention heads
    n_layers: int = 4                    # Number of transformer layers
    dropout: float = 0.1

    # Actor-Critic
    hidden_dim: int = 128
    action_dim: int = 3


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99                  # Discount factor
    gae_lambda: float = 0.95             # GAE lambda

    # PPO specific
    clip_epsilon: float = 0.2            # PPO clipping
    entropy_coef: float = 0.05           # Entropy bonus coefficient
    value_coef: float = 0.5              # Value loss coefficient
    max_grad_norm: float = 0.5           # Gradient clipping

    # Training loop
    n_epochs: int = 4                    # PPO epochs per update
    batch_size: int = 64
    n_episodes: int = 1000               # Total training episodes

    # Device
    device: Optional[str] = None         # Auto-detect if None

    def get_device(self) -> torch.device:
        """Get the training device."""
        if self.device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = True
    stages: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])
    episodes_per_stage: int = 200
    advancement_threshold: float = 0.4   # Minimum fill ratio to advance


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    enabled: bool = True
    output_dir: str = "training_outputs"
    save_every: int = 1                  # Save every N episodes


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Experiment metadata
    experiment_name: str = "tree_packing_ppo"
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        assert self.env.k_neighbors >= 1, "k_neighbors must be at least 1"
        assert self.network.n_heads > 0, "n_heads must be positive"
        assert self.network.n_layers > 0, "n_layers must be positive"
        assert self.training.learning_rate > 0, "learning_rate must be positive"

        # Update pairwise_feature_dim based on k_neighbors
        self.network.pairwise_feature_dim = self.env.k_neighbors * 10


# ==================== PRESET CONFIGURATIONS ====================

def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_fast_debug_config() -> Config:
    """Get configuration for fast debugging."""
    return Config(
        env=EnvironmentConfig(n_trees=10, max_trees=50),
        network=NetworkConfig(n_heads=4, n_layers=2, embed_dim=32),
        training=TrainingConfig(
            n_episodes=10,
            batch_size=16,
            n_epochs=2
        ),
        curriculum=CurriculumConfig(enabled=False),
        visualization=VisualizationConfig(save_every=5)
    )


def get_full_training_config() -> Config:
    """Get configuration for full training."""
    return Config(
        env=EnvironmentConfig(n_trees=200, max_trees=200),
        network=NetworkConfig(n_heads=8, n_layers=6, embed_dim=128),
        training=TrainingConfig(
            n_episodes=5000,
            learning_rate=1e-4,
            batch_size=128
        ),
        curriculum=CurriculumConfig(
            enabled=True,
            stages=[10, 25, 50, 100, 150, 200],
            episodes_per_stage=500
        )
    )


def get_curriculum_config() -> Config:
    """Get configuration for curriculum learning."""
    return Config(
        curriculum=CurriculumConfig(
            enabled=True,
            stages=[10, 25, 50, 100, 200],
            episodes_per_stage=300,
            advancement_threshold=0.35
        ),
        training=TrainingConfig(
            n_episodes=2000,
            learning_rate=3e-4
        )
    )


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Load default config
    config = get_default_config()
    print(f"Environment: {config.env}")
    print(f"Network: {config.network}")
    print(f"Training: {config.training}")
    print(f"Device: {config.training.get_device()}")

    # Load preset
    debug_config = get_fast_debug_config()
    print(f"\nDebug config n_trees: {debug_config.env.n_trees}")
