"""
Highway Environment Manager

Manages highway-env simulation environments optimized for Apple Silicon.
Provides rich environment feedback and multiple driving scenarios.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, Any, List


class HighwayEnvironment:
    """Highway environment wrapper with enhanced features for RL training."""
    
    SCENARIOS = {
        'highway': 'highway-fast-v0',
        'merge': 'merge-v0', 
        'intersection': 'intersection-v0',
        'parking': 'parking-v0',
        'racetrack': 'racetrack-v0'
    }
    
    def __init__(
        self,
        scenario: str = 'highway',
        render_mode: Optional[str] = None,
        config_overrides: Optional[Dict] = None
    ):
        """
        Initialize highway environment.
        
        Args:
            scenario: Driving scenario ('highway', 'merge', 'intersection', 'parking', 'racetrack')
            render_mode: Rendering mode ('human', 'rgb_array', None)
            config_overrides: Environment configuration overrides
        """
        self.scenario = scenario
        self.env_id = self.SCENARIOS.get(scenario, 'highway-fast-v0')
        
        # Create environment
        self.env = gym.make(self.env_id, render_mode=render_mode)
        
        # Apply configuration overrides
        if config_overrides:
            if hasattr(self.env, 'configure'):
                self.env.configure(config_overrides)
            else:
                # For newer gymnasium versions, configuration is handled differently
                self.env.unwrapped.configure(config_overrides)
        
        # Environment info
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Metrics tracking
        self.episode_metrics = {
            'rewards': [],
            'collisions': 0,
            'speed_violations': 0,
            'lane_changes': 0,
            'distance_traveled': 0.0
        }
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment and metrics."""
        obs, info = self.env.reset()
        
        # Reset metrics
        self.episode_metrics = {
            'rewards': [],
            'collisions': 0,
            'speed_violations': 0,
            'lane_changes': 0,
            'distance_traveled': 0.0
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and track metrics.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track metrics
        self.episode_metrics['rewards'].append(reward)
        
        # Extract additional metrics from info if available
        if 'crashed' in info and info['crashed']:
            self.episode_metrics['collisions'] += 1
            
        if 'speed' in info:
            # Track speed violations (assuming speed limit of 30)
            if info['speed'] > 30:
                self.episode_metrics['speed_violations'] += 1
                
        # Estimate distance traveled (rough approximation)
        if 'speed' in info:
            self.episode_metrics['distance_traveled'] += info['speed'] * 0.1  # dt = 0.1s
            
        return obs, reward, terminated, truncated, info
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get comprehensive episode metrics."""
        rewards = self.episode_metrics['rewards']
        
        return {
            'total_reward': sum(rewards),
            'mean_reward': np.mean(rewards) if rewards else 0,
            'episode_length': len(rewards),
            'collisions': self.episode_metrics['collisions'],
            'speed_violations': self.episode_metrics['speed_violations'],
            'lane_changes': self.episode_metrics['lane_changes'],
            'distance_traveled': self.episode_metrics['distance_traveled'],
            'reward_std': np.std(rewards) if len(rewards) > 1 else 0,
            'scenario': self.scenario
        }
    
    def render(self):
        """Render environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    @classmethod
    def get_optimized_config(cls, scenario: str) -> Dict:
        """Get optimized configuration for each scenario."""
        base_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        }
        
        # Scenario-specific optimizations
        if scenario == 'highway':
            base_config.update({
                "lanes_count": 4,
                "vehicles_count": 50,
                "duration": 40,
                "initial_spacing": 2,
                "collision_reward": -1,
                "reward_speed_range": [20, 30],
                "normalize_reward": True
            })
        elif scenario == 'merge':
            base_config.update({
                "collision_reward": -1,
                "normalize_reward": True,
                "merging_lane_length": 80
            })
        elif scenario == 'intersection':
            base_config.update({
                "collision_reward": -5,
                "normalize_reward": True,
                "incoming_vehicle_destination": None
            })
        elif scenario == 'parking':
            base_config.update({
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False
                },
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "normalize_reward": True
            })
        elif scenario == 'racetrack':
            base_config.update({
                "collision_reward": -1,
                "normalize_reward": True,
                "lap_reward": 1
            })
            
        return base_config


class MultiScenarioEnvironment:
    """Manager for training across multiple highway scenarios."""
    
    def __init__(self, scenarios: List[str], render_mode: Optional[str] = None):
        """
        Initialize multi-scenario environment.
        
        Args:
            scenarios: List of scenarios to cycle through
            render_mode: Rendering mode
        """
        self.scenarios = scenarios
        self.current_scenario_idx = 0
        self.render_mode = render_mode
        
        # Create environments for each scenario
        self.environments = {}
        for scenario in scenarios:
            config = HighwayEnvironment.get_optimized_config(scenario)
            self.environments[scenario] = HighwayEnvironment(
                scenario=scenario,
                render_mode=render_mode,
                config_overrides=config
            )
        
        self.current_env = self.environments[scenarios[0]]
        
    def switch_scenario(self, scenario: Optional[str] = None):
        """Switch to a different scenario."""
        if scenario is None:
            # Cycle to next scenario
            self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenarios)
            scenario = self.scenarios[self.current_scenario_idx]
        
        if scenario in self.environments:
            self.current_env = self.environments[scenario]
            return True
        return False
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset current environment."""
        return self.current_env.reset()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step current environment."""
        return self.current_env.step(action)
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get episode summary from current environment."""
        return self.current_env.get_episode_summary()
    
    def render(self):
        """Render current environment."""
        return self.current_env.render()
    
    def close(self):
        """Close all environments."""
        for env in self.environments.values():
            env.close()
    
    @property
    def observation_space(self):
        """Get observation space of current environment."""
        return self.current_env.observation_space
    
    @property
    def action_space(self):
        """Get action space of current environment."""
        return self.current_env.action_space
    
    @property
    def current_scenario(self) -> str:
        """Get current scenario name."""
        return self.scenarios[self.current_scenario_idx]
