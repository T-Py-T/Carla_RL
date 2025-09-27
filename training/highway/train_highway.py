#!/usr/bin/env python3
"""
Highway-env Training Script

Modern RL training optimized for Apple Silicon using highway-env.
This is the primary training approach for macOS development.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from highway_rl import HighwayDQNAgent, HighwayEnvironment, HighwayTrainer, WandBLogger
from highway_rl.logger import MultiLogger
from highway_rl.trainer import MultiScenarioTrainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Highway RL Agent")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--scenario", type=str, default="highway", 
                       choices=["highway", "merge", "intersection", "parking", "racetrack", "curriculum"],
                       help="Training scenario")
    parser.add_argument("--eval-frequency", type=int, default=50, help="Evaluation frequency")
    parser.add_argument("--save-frequency", type=int, default=100, help="Model save frequency")
    
    # Agent parameters
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--memory-size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--double-dqn", action="store_true", help="Use Double DQN")
    parser.add_argument("--dueling-dqn", action="store_true", help="Use Dueling DQN")
    
    # Logging parameters
    parser.add_argument("--wandb-project", type=str, default="highway-rl-training", 
                       help="W&B project name")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    
    # Model parameters
    parser.add_argument("--load-model", type=str, help="Path to load existing model")
    parser.add_argument("--save-dir", type=str, default="models/highway", help="Model save directory")
    
    # Performance parameters
    parser.add_argument("--mixed-precision", action="store_true", default=True, 
                       help="Use mixed precision training")
    parser.add_argument("--target-reward", type=float, help="Target reward for early stopping")
    
    return parser.parse_args()


def create_agent(args, state_size, action_size):
    """Create DQN agent with specified parameters."""
    return HighwayDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        double_dqn=args.double_dqn,
        dueling_dqn=args.dueling_dqn,
        use_mixed_precision=args.mixed_precision
    )


def create_logger(args):
    """Create logger with specified parameters."""
    if args.no_wandb and args.no_tensorboard:
        return None
    
    config = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "learning_rate": args.learning_rate,
        "epsilon_decay": args.epsilon_decay,
        "memory_size": args.memory_size,
        "batch_size": args.batch_size,
        "double_dqn": args.double_dqn,
        "dueling_dqn": args.dueling_dqn,
        "mixed_precision": args.mixed_precision
    }
    
    return MultiLogger(
        project_name=args.wandb_project,
        experiment_name=args.experiment_name,
        config=config,
        use_wandb=not args.no_wandb,
        use_tensorboard=not args.no_tensorboard
    )


def train_single_scenario(args):
    """Train on a single scenario."""
    print(f"Initializing training for scenario: {args.scenario}")
    
    # Create environment
    env = HighwayEnvironment(
        scenario=args.scenario,
        config_overrides=HighwayEnvironment.get_optimized_config(args.scenario)
    )
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Create agent
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    
    agent = create_agent(args, state_size, action_size)
    
    # Load existing model if specified
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        agent.load(args.load_model)
    
    print(f"Agent created:")
    print(f"  Architecture: {'Dueling' if args.dueling_dqn else 'Standard'} DQN")
    print(f"  Double DQN: {args.double_dqn}")
    print(f"  Mixed Precision: {args.mixed_precision}")
    print(f"  Parameters: {agent.q_network.count_params():,}")
    
    # Create logger
    logger = create_logger(args)
    
    # Create trainer
    trainer = HighwayTrainer(
        agent=agent,
        environment=env,
        logger=logger,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency
    )
    
    # Train
    print(f"\nStarting training...")
    results = trainer.train(
        episodes=args.episodes,
        eval_frequency=args.eval_frequency,
        target_reward=args.target_reward
    )
    
    # Final evaluation
    print(f"\nFinal evaluation:")
    final_eval = trainer.evaluate(episodes=50)
    
    print(f"Final Results:")
    print(f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  Success Rate: {final_eval['success_rate']:.2f}")
    print(f"  Collision Rate: {final_eval['collision_rate']:.2f}")
    print(f"  Training Time: {results['total_training_time']:.2f}s")
    print(f"  Episodes/Hour: {results['episodes_per_hour']:.1f}")
    
    # Cleanup
    if logger:
        logger.finish()
    env.close()
    
    return results


def train_curriculum(args):
    """Train using curriculum learning across multiple scenarios."""
    print("Initializing curriculum training...")
    
    scenarios = ["highway", "merge", "intersection", "parking", "racetrack"]
    
    # Create environment to get dimensions
    temp_env = HighwayEnvironment("highway")
    state_size = temp_env.observation_space.shape
    action_size = temp_env.action_space.n
    temp_env.close()
    
    # Create agent
    agent = create_agent(args, state_size, action_size)
    
    # Load existing model if specified
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        agent.load(args.load_model)
    
    print(f"Agent created for curriculum training:")
    print(f"  Scenarios: {scenarios}")
    print(f"  Architecture: {'Dueling' if args.dueling_dqn else 'Standard'} DQN")
    print(f"  Parameters: {agent.q_network.count_params():,}")
    
    # Create logger
    logger = create_logger(args)
    
    # Create multi-scenario trainer
    trainer = MultiScenarioTrainer(
        agent=agent,
        scenarios=scenarios,
        logger=logger,
        save_dir=args.save_dir
    )
    
    # Train with curriculum
    print(f"\nStarting curriculum training...")
    results = trainer.train_curriculum(
        episodes_per_scenario=args.episodes // len(scenarios),
        evaluation_episodes=20
    )
    
    # Print final results
    print(f"\nCurriculum Training Complete!")
    print(f"Final Performance Across All Scenarios:")
    for scenario, eval_results in results['final_evaluation'].items():
        print(f"  {scenario:12s}: {eval_results['mean_reward']:6.2f} ± {eval_results['std_reward']:5.2f} "
              f"(Success: {eval_results['success_rate']:5.2f})")
    
    # Cleanup
    if logger:
        logger.finish()
    
    return results


def main():
    """Main training function."""
    args = parse_arguments()
    
    print("Highway RL Training")
    print("=" * 50)
    print(f"Platform: Apple Silicon Optimized")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}")
    print(f"Mixed Precision: {args.mixed_precision}")
    print("=" * 50)
    
    try:
        if args.scenario == "curriculum":
            results = train_curriculum(args)
        else:
            results = train_single_scenario(args)
        
        print("\nTraining completed successfully!")
        return results
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return None
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
