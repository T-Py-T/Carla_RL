#!/usr/bin/env python3
"""
Highway Model Evaluation Script

Evaluates trained highway RL models and reports performance metrics.
"""

import sys
import os
import glob
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from highway_rl import HighwayDQNAgent, HighwayEnvironment, HighwayTrainer
    import numpy as np
    
    def evaluate_latest_model():
        """Evaluate the most recently trained model."""
        models_dir = Path(__file__).parent.parent.parent / "models" / "highway"
        
        if not models_dir.exists():
            print("ERROR: No models/highway directory found")
            print("Run 'make train-highway' first to train a model")
            return
        
        # Find model files
        model_files = list(models_dir.glob("*_weights.h5"))
        
        if not model_files:
            print("ERROR: No trained models found")
            print("Run 'make train-highway' first to train a model")
            return
        
        # Get latest model
        latest_model = max(model_files, key=os.path.getctime)
        model_name = str(latest_model).replace('_weights.h5', '')
        
        print(f"Evaluating model: {latest_model.name}")
        
        # Create environment and agent
        env = HighwayEnvironment('highway')
        agent = HighwayDQNAgent(env.observation_space.shape, env.action_space.n)
        
        # Load model
        agent.load(model_name)
        
        # Create trainer and evaluate
        trainer = HighwayTrainer(agent, env)
        results = trainer.evaluate(episodes=20)
        
        print("\nEvaluation Results:")
        print("=" * 40)
        print(f"Mean Reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Success Rate:    {results['success_rate']:.2f}")
        print(f"Collision Rate:  {results['collision_rate']:.2f}")
        print(f"Episode Length:  {results['mean_episode_length']:.1f}")
        print(f"Episodes:        {results['episodes']}")
        
        env.close()
        
    if __name__ == "__main__":
        evaluate_latest_model()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("The highway RL system may not be fully set up.")
    print("Try running 'make setup' first.")
except Exception as e:
    print(f"Evaluation error: {e}")
    import traceback
    traceback.print_exc()
