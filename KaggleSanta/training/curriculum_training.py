# ==================== CURRICULUM TRAINING ====================
"""
Curriculum learning: 10 → 200 ağaç aşamalı training.
Her aşamada model önceki aşamanın checkpoint'inden başlar.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path

from transformer_training import TransformerEnvironment, TransformerTrainer


class CurriculumTrainer:
    """
    Curriculum learning with progressive tree count.
    
    Stages:
    1. 10 trees  - 500 episodes (learn basics)
    2. 30 trees  - 1000 episodes (simple packing)
    3. 50 trees  - 2000 episodes (medium complexity)
    4. 100 trees - 3000 episodes (advanced packing)
    5. 200 trees - 5000 episodes (full challenge)
    """
    
    def __init__(
        self,
        output_dir: str = "curriculum_outputs",
        device: str = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Curriculum stages
        self.stages = [
            {'n_trees': 10, 'episodes': 500, 'lr': 3e-4},
            {'n_trees': 30, 'episodes': 1000, 'lr': 2e-4},
            {'n_trees': 50, 'episodes': 2000, 'lr': 1e-4},
            {'n_trees': 100, 'episodes': 3000, 'lr': 5e-5},
            {'n_trees': 200, 'episodes': 5000, 'lr': 3e-5}
        ]
        
        self.current_model = None
    
    def train(self, start_stage: int = 0, end_stage: int = None):
        """
        Run curriculum training.
        
        Args:
            start_stage: Stage index to start from (0-4)
            end_stage: Stage index to end at (inclusive), None = all
        """
        if end_stage is None:
            end_stage = len(self.stages) - 1
        
        print(f"Curriculum Training: Stages {start_stage} to {end_stage}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        for i in range(start_stage, end_stage + 1):
            stage = self.stages[i]
            print(f"\n[Stage {i+1}/{len(self.stages)}] n_trees={stage['n_trees']}, episodes={stage['episodes']}, lr={stage['lr']}")
            print("-" * 60)
            
            # Create trainer for this stage
            trainer = TransformerTrainer(
                n_trees=stage['n_trees'],
                device=self.device
            )
            
            # Load previous stage checkpoint if exists
            if self.current_model is not None:
                trainer.model.load_state_dict(self.current_model)
                print("  -> Loaded weights from previous stage")
            
            # Update learning rate
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = stage['lr']
            
            # Train
            trainer.train(n_episodes=stage['episodes'])
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f"stage_{i}_trees_{stage['n_trees']}.pt"
            torch.save(trainer.model.state_dict(), checkpoint_path)
            print(f"  -> Saved checkpoint: {checkpoint_path}")
            
            # Keep model for next stage
            self.current_model = trainer.model.state_dict()
        
        print("\n" + "=" * 60)
        print("Curriculum training completed!")
        print(f"Final model: {self.output_dir}/stage_{end_stage}_trees_{self.stages[end_stage]['n_trees']}.pt")


def main():
    parser = argparse.ArgumentParser(description="Curriculum Training for Tree Packing")
    parser.add_argument("--start_stage", type=int, default=0, help="Start stage (0-4)")
    parser.add_argument("--end_stage", type=int, default=None, help="End stage (0-4), default=all")
    parser.add_argument("--output_dir", type=str, default="curriculum_outputs", help="Output directory")
    args = parser.parse_args()
    
    trainer = CurriculumTrainer(output_dir=args.output_dir)
    trainer.train(start_stage=args.start_stage, end_stage=args.end_stage)


if __name__ == "__main__":
    main()
