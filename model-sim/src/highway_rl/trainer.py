"""
Highway RL Trainer

Modern training pipeline for highway-env with comprehensive evaluation
and Apple Silicon optimizations.
"""

import numpy as np
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .environment import HighwayEnvironment, MultiScenarioEnvironment
from .agent import HighwayDQNAgent
from .logger import WandBLogger, MultiLogger


class HighwayTrainer:
    """Advanced trainer for highway driving RL agents."""
    
    def __init__(
        self,
        agent: HighwayDQNAgent,
        environment: HighwayEnvironment,
        logger: Optional[MultiLogger] = None,
        save_dir: str = "models/highway",
        evaluation_episodes: int = 10,
        save_frequency: int = 100
    ):
        """
        Initialize Highway Trainer.
        
        Args:
            agent: HighwayDQNAgent instance
            environment: HighwayEnvironment instance
            logger: Logger for tracking training progress
            save_dir: Directory to save models
            evaluation_episodes: Number of episodes for evaluation
            save_frequency: Frequency of model saving (episodes)
        """
        self.agent = agent
        self.environment = environment
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_episodes = evaluation_episodes
        self.save_frequency = save_frequency
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_start_time = None
        self.best_mean_reward = float('-inf')
        
    def train(
        self,
        episodes: int,
        max_steps_per_episode: int = 1000,
        eval_frequency: int = 50,
        early_stopping_patience: int = 200,
        target_reward: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            eval_frequency: Frequency of evaluation (episodes)
            early_stopping_patience: Episodes to wait before early stopping
            target_reward: Target reward for early stopping
            
        Returns:
            Training summary statistics
        """
        print(f"Starting training for {episodes} episodes...")
        print(f"Environment: {self.environment.scenario}")
        print(f"Agent architecture: {'Dueling' if self.agent.dueling_dqn else 'Standard'} DQN")
        print(f"Double DQN: {self.agent.double_dqn}")
        
        self.training_start_time = time.time()
        patience_counter = 0
        
        for episode in range(episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state, _ = self.environment.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose action
                action = self.agent.act(state, training=True)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                training_metrics = self.agent.replay()
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Episode completed
            episode_time = time.time() - episode_start_time
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Get episode metrics
            episode_metrics = self.environment.get_episode_summary()
            episode_metrics.update({
                'episode_time': episode_time,
                'steps_per_second': steps / episode_time if episode_time > 0 else 0
            })
            
            # Log episode
            if self.logger:
                self.logger.log_episode(
                    episode_metrics=episode_metrics,
                    agent_metrics=training_metrics,
                    model_metrics=self.agent.get_training_metrics()
                )
            
            # Print progress
            if episode % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards)
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Mean(10): {mean_reward:7.2f} | "
                      f"Steps: {steps:3d} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Collisions: {episode_metrics['collisions']}")
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                eval_results = self.evaluate()
                
                if self.logger:
                    self.logger.log_evaluation(eval_results)
                
                # Check for improvement
                current_mean_reward = eval_results['mean_reward']
                if current_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = current_mean_reward
                    patience_counter = 0
                    
                    # Save best model
                    self.save_model(f"best_model_episode_{episode}")
                    print(f"New best model! Mean reward: {current_mean_reward:.2f}")
                else:
                    patience_counter += 1
                
                print(f"Evaluation | Mean: {current_mean_reward:.2f} | "
                      f"Std: {eval_results['std_reward']:.2f} | "
                      f"Success Rate: {eval_results['success_rate']:.2f}")
            
            # Save model periodically
            if episode % self.save_frequency == 0 and episode > 0:
                self.save_model(f"checkpoint_episode_{episode}")
            
            # Early stopping
            if target_reward and len(self.episode_rewards) >= 100:
                recent_mean = np.mean(self.episode_rewards[-100:])
                if recent_mean >= target_reward:
                    print(f"Target reward {target_reward} reached! Stopping training.")
                    break
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping: No improvement for {early_stopping_patience} evaluations.")
                break
        
        # Training completed
        training_time = time.time() - self.training_start_time
        
        # Final evaluation
        final_eval = self.evaluate()
        
        # Save final model
        self.save_model("final_model")
        
        # Training summary
        summary = {
            'episodes_trained': episode + 1,
            'total_training_time': training_time,
            'best_mean_reward': self.best_mean_reward,
            'final_evaluation': final_eval,
            'episodes_per_hour': (episode + 1) / (training_time / 3600),
            'mean_episode_length': np.mean(self.episode_lengths),
            'total_steps': sum(self.episode_lengths)
        }
        
        print("\nTraining Summary:")
        print(f"Episodes: {summary['episodes_trained']}")
        print(f"Training Time: {summary['total_training_time']:.2f}s")
        print(f"Best Mean Reward: {summary['best_mean_reward']:.2f}")
        print(f"Episodes/Hour: {summary['episodes_per_hour']:.1f}")
        
        return summary
    
    def evaluate(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the agent's performance.
        
        Args:
            episodes: Number of evaluation episodes (uses default if None)
            
        Returns:
            Evaluation metrics
        """
        if episodes is None:
            episodes = self.evaluation_episodes
        
        print(f"Evaluating for {episodes} episodes...")
        
        rewards = []
        episode_lengths = []
        collisions = []
        success_count = 0
        
        for episode in range(episodes):
            state, _ = self.environment.reset()
            total_reward = 0
            steps = 0
            episode_collisions = 0
            
            for _ in range(1000):  # Max steps for evaluation
                # Use greedy policy (no exploration)
                action = self.agent.act(state, training=False)
                
                next_state, reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Track collisions
                if 'crashed' in info and info['crashed']:
                    episode_collisions += 1
                
                if done:
                    # Check success criteria (scenario-dependent)
                    if self._is_successful_episode(total_reward, steps, episode_collisions):
                        success_count += 1
                    break
            
            rewards.append(total_reward)
            episode_lengths.append(steps)
            collisions.append(episode_collisions)
        
        # Calculate metrics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        success_rate = success_count / episodes
        collision_rate = np.mean([c > 0 for c in collisions])
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'mean_episode_length': np.mean(episode_lengths),
            'episodes': episodes,
            'individual_rewards': rewards
        }
    
    def _is_successful_episode(self, reward: float, steps: int, collisions: int) -> bool:
        """
        Determine if an episode was successful based on scenario-specific criteria.
        
        Args:
            reward: Total episode reward
            steps: Episode length
            collisions: Number of collisions
            
        Returns:
            Whether the episode was successful
        """
        # Basic success criteria - can be customized per scenario
        if collisions > 0:
            return False
        
        # Scenario-specific success criteria
        scenario = self.environment.scenario
        
        if scenario == 'highway':
            return reward > 20 and steps > 100
        elif scenario == 'merge':
            return reward > 15 and steps > 50
        elif scenario == 'intersection':
            return reward > 10 and steps > 30
        elif scenario == 'parking':
            return reward > 0.5  # Parking has different reward scale
        elif scenario == 'racetrack':
            return reward > 25
        else:
            return reward > 10  # Default criteria
    
    def save_model(self, name: str):
        """Save model with given name."""
        filepath = self.save_dir / name
        self.agent.save(str(filepath))
        
        # Save training metadata
        metadata = {
            'episode_rewards': self.episode_rewards[-100:],  # Last 100 episodes
            'episode_lengths': self.episode_lengths[-100:],
            'best_mean_reward': self.best_mean_reward,
            'scenario': self.environment.scenario,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        import json
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, name: str):
        """Load model with given name."""
        filepath = self.save_dir / name
        self.agent.load(str(filepath))


class MultiScenarioTrainer:
    """Trainer for multiple highway scenarios with curriculum learning."""
    
    def __init__(
        self,
        agent: HighwayDQNAgent,
        scenarios: List[str],
        logger: Optional[MultiLogger] = None,
        save_dir: str = "models/highway_multi",
        curriculum_threshold: float = 15.0
    ):
        """
        Initialize multi-scenario trainer.
        
        Args:
            agent: HighwayDQNAgent instance
            scenarios: List of scenarios to train on
            logger: Logger instance
            save_dir: Directory to save models
            curriculum_threshold: Reward threshold to advance to next scenario
        """
        self.agent = agent
        self.scenarios = scenarios
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.curriculum_threshold = curriculum_threshold
        
        # Create multi-scenario environment
        self.environment = MultiScenarioEnvironment(scenarios)
        
        # Create individual trainers for each scenario
        self.trainers = {}
        for scenario in scenarios:
            env = HighwayEnvironment(scenario)
            self.trainers[scenario] = HighwayTrainer(
                agent=agent,
                environment=env,
                logger=logger,
                save_dir=str(self.save_dir / scenario)
            )
    
    def train_curriculum(
        self,
        episodes_per_scenario: int = 200,
        evaluation_episodes: int = 20
    ) -> Dict[str, Any]:
        """
        Train using curriculum learning across scenarios.
        
        Args:
            episodes_per_scenario: Episodes to train on each scenario
            evaluation_episodes: Episodes for evaluation
            
        Returns:
            Training summary
        """
        print("Starting curriculum training across scenarios...")
        
        results = {}
        
        for i, scenario in enumerate(self.scenarios):
            print(f"\n{'='*50}")
            print(f"Training on scenario: {scenario} ({i+1}/{len(self.scenarios)})")
            print(f"{'='*50}")
            
            trainer = self.trainers[scenario]
            
            # Train on current scenario
            scenario_results = trainer.train(
                episodes=episodes_per_scenario,
                eval_frequency=50
            )
            
            # Evaluate performance
            eval_results = trainer.evaluate(episodes=evaluation_episodes)
            
            results[scenario] = {
                'training': scenario_results,
                'evaluation': eval_results
            }
            
            # Check if ready for next scenario
            if eval_results['mean_reward'] < self.curriculum_threshold:
                print(f"Warning: Performance below threshold ({self.curriculum_threshold}) "
                      f"for {scenario}. Mean reward: {eval_results['mean_reward']:.2f}")
            
            # Save scenario-specific model
            trainer.save_model(f"curriculum_{scenario}_complete")
        
        # Final evaluation across all scenarios
        print(f"\n{'='*50}")
        print("Final evaluation across all scenarios")
        print(f"{'='*50}")
        
        final_results = {}
        for scenario in self.scenarios:
            self.environment.switch_scenario(scenario)
            trainer = HighwayTrainer(
                agent=self.agent,
                environment=self.environment.environments[scenario],
                logger=self.logger
            )
            eval_results = trainer.evaluate(episodes=evaluation_episodes)
            final_results[scenario] = eval_results
            
            print(f"{scenario:12s} | Mean: {eval_results['mean_reward']:6.2f} | "
                  f"Success: {eval_results['success_rate']:5.2f} | "
                  f"Collisions: {eval_results['collision_rate']:5.2f}")
        
        # Save final multi-scenario model
        final_filepath = self.save_dir / "final_curriculum_model"
        self.agent.save(str(final_filepath))
        
        return {
            'scenario_results': results,
            'final_evaluation': final_results,
            'scenarios_trained': self.scenarios
        }
