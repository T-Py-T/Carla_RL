"""
Advanced Logging and Monitoring

Weights & Biases integration for comprehensive training evaluation
optimized for Apple Silicon performance monitoring.
"""

import wandb
import numpy as np
import tensorflow as tf
import time
import psutil
import platform
from typing import Dict, Any, Optional
from pathlib import Path


class WandBLogger:
    """Weights & Biases logger with Apple Silicon optimizations."""
    
    def __init__(
        self,
        project_name: str = "highway-rl-training",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_system_metrics: bool = True
    ):
        """
        Initialize W&B logger.
        
        Args:
            project_name: W&B project name
            experiment_name: Experiment name (auto-generated if None)
            config: Configuration dictionary to log
            log_system_metrics: Whether to log system performance metrics
        """
        self.project_name = project_name
        self.log_system_metrics = log_system_metrics
        self.start_time = time.time()
        
        # Initialize W&B
        if experiment_name is None:
            experiment_name = f"highway-training-{int(time.time())}"
            
        # Enhanced config with system info
        full_config = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        if config:
            full_config.update(config)
            
        # Detect Apple Silicon
        if platform.processor() == 'arm' or 'arm64' in platform.platform().lower():
            full_config["apple_silicon"] = True
            full_config["metal_acceleration"] = self._check_metal_acceleration()
        else:
            full_config["apple_silicon"] = False
            
        wandb.init(
            project=self.project_name,
            name=experiment_name,
            config=full_config
        )
        
        self.episode_count = 0
        self.step_count = 0
        
    def _check_metal_acceleration(self) -> bool:
        """Check if Metal acceleration is available."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except:
            return False
    
    def log_episode(
        self,
        episode_metrics: Dict[str, Any],
        agent_metrics: Optional[Dict[str, Any]] = None,
        model_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log episode metrics.
        
        Args:
            episode_metrics: Episode performance metrics
            agent_metrics: Agent-specific metrics (epsilon, learning rate, etc.)
            model_metrics: Model performance metrics (loss, Q-values, etc.)
        """
        self.episode_count += 1
        
        # Base episode metrics
        log_data = {
            "episode": self.episode_count,
            "episode/total_reward": episode_metrics.get("total_reward", 0),
            "episode/mean_reward": episode_metrics.get("mean_reward", 0),
            "episode/length": episode_metrics.get("episode_length", 0),
            "episode/collisions": episode_metrics.get("collisions", 0),
            "episode/speed_violations": episode_metrics.get("speed_violations", 0),
            "episode/distance_traveled": episode_metrics.get("distance_traveled", 0),
            "episode/scenario": episode_metrics.get("scenario", "unknown"),
            "time/episode_time": time.time() - self.start_time
        }
        
        # Agent metrics
        if agent_metrics:
            for key, value in agent_metrics.items():
                log_data[f"agent/{key}"] = value
                
        # Model metrics
        if model_metrics:
            for key, value in model_metrics.items():
                log_data[f"model/{key}"] = value
        
        # System metrics
        if self.log_system_metrics:
            system_metrics = self._get_system_metrics()
            log_data.update(system_metrics)
            
        wandb.log(log_data)
    
    def log_training_step(
        self,
        step_metrics: Dict[str, Any],
        model_weights: Optional[Dict] = None
    ):
        """
        Log training step metrics.
        
        Args:
            step_metrics: Training step metrics
            model_weights: Model weights for histogram logging
        """
        self.step_count += 1
        
        log_data = {
            "step": self.step_count,
            "training/loss": step_metrics.get("loss", 0),
            "training/q_mean": step_metrics.get("q_mean", 0),
            "training/q_std": step_metrics.get("q_std", 0),
            "training/learning_rate": step_metrics.get("learning_rate", 0),
            "training/epsilon": step_metrics.get("epsilon", 0),
            "training/replay_buffer_size": step_metrics.get("buffer_size", 0)
        }
        
        # Log model weight histograms periodically
        if model_weights and self.step_count % 100 == 0:
            for name, weights in model_weights.items():
                wandb.log({f"weights/{name}": wandb.Histogram(weights.numpy())})
        
        wandb.log(log_data)
    
    def log_evaluation(
        self,
        eval_metrics: Dict[str, Any],
        scenario_breakdown: Optional[Dict[str, Dict]] = None
    ):
        """
        Log evaluation results.
        
        Args:
            eval_metrics: Overall evaluation metrics
            scenario_breakdown: Per-scenario performance breakdown
        """
        log_data = {
            "eval/mean_reward": eval_metrics.get("mean_reward", 0),
            "eval/std_reward": eval_metrics.get("std_reward", 0),
            "eval/success_rate": eval_metrics.get("success_rate", 0),
            "eval/collision_rate": eval_metrics.get("collision_rate", 0),
            "eval/episodes_evaluated": eval_metrics.get("episodes", 0)
        }
        
        # Per-scenario breakdown
        if scenario_breakdown:
            for scenario, metrics in scenario_breakdown.items():
                for metric, value in metrics.items():
                    log_data[f"eval/{scenario}/{metric}"] = value
        
        wandb.log(log_data)
    
    def log_model_comparison(
        self,
        model_name: str,
        comparison_metrics: Dict[str, float]
    ):
        """Log model comparison metrics."""
        log_data = {}
        for metric, value in comparison_metrics.items():
            log_data[f"comparison/{model_name}/{metric}"] = value
            
        wandb.log(log_data)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "system/cpu_percent": cpu_percent,
                "system/memory_percent": memory.percent,
                "system/memory_used_gb": memory.used / (1024**3),
                "system/memory_available_gb": memory.available / (1024**3)
            }
            
            # GPU metrics (if available)
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # For Apple Silicon, we can't get detailed GPU metrics easily
                    # but we can log that GPU is being used
                    metrics["system/gpu_available"] = len(gpus)
            except:
                pass
                
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not collect system metrics: {e}")
            return {}
    
    def log_hyperparameter_sweep(
        self,
        sweep_config: Dict[str, Any],
        best_metrics: Dict[str, float]
    ):
        """Log hyperparameter sweep results."""
        wandb.log({
            "sweep/best_reward": best_metrics.get("reward", 0),
            "sweep/best_success_rate": best_metrics.get("success_rate", 0),
            "sweep/configuration": sweep_config
        })
    
    def save_model_artifact(
        self,
        model_path: str,
        model_name: str = "highway_dqn_model",
        metadata: Optional[Dict] = None
    ):
        """Save model as W&B artifact."""
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=metadata or {}
        )
        
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish W&B run."""
        total_time = time.time() - self.start_time
        wandb.log({
            "summary/total_training_time": total_time,
            "summary/total_episodes": self.episode_count,
            "summary/total_steps": self.step_count,
            "summary/episodes_per_hour": self.episode_count / (total_time / 3600)
        })
        
        wandb.finish()


class TensorBoardLogger:
    """Enhanced TensorBoard logger for local monitoring."""
    
    def __init__(self, log_dir: str = "logs/tensorboard"):
        """Initialize TensorBoard logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = tf.summary.create_file_writer(str(self.log_dir))
        self.step = 0
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log scalar value."""
        if step is None:
            step = self.step
            
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)
            
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """Log histogram."""
        if step is None:
            step = self.step
            
        with self.writer.as_default():
            tf.summary.histogram(name, values, step=step)
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        """Log image."""
        if step is None:
            step = self.step
            
        with self.writer.as_default():
            tf.summary.image(name, image, step=step)
    
    def flush(self):
        """Flush writer."""
        self.writer.flush()
    
    def close(self):
        """Close writer."""
        self.writer.close()


class MultiLogger:
    """Combined logger using both W&B and TensorBoard."""
    
    def __init__(
        self,
        project_name: str = "highway-rl-training",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True
    ):
        """Initialize multi-logger."""
        self.loggers = []
        
        if use_wandb:
            self.wandb_logger = WandBLogger(project_name, experiment_name, config)
            self.loggers.append(self.wandb_logger)
            
        if use_tensorboard:
            self.tb_logger = TensorBoardLogger()
            self.loggers.append(self.tb_logger)
    
    def log_episode(self, episode_metrics: Dict, agent_metrics: Dict = None, model_metrics: Dict = None):
        """Log to all loggers."""
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.log_episode(episode_metrics, agent_metrics, model_metrics)
            
        if hasattr(self, 'tb_logger'):
            # Log key metrics to TensorBoard
            self.tb_logger.log_scalar("episode/total_reward", episode_metrics.get("total_reward", 0))
            self.tb_logger.log_scalar("episode/length", episode_metrics.get("episode_length", 0))
            if agent_metrics:
                for key, value in agent_metrics.items():
                    self.tb_logger.log_scalar(f"agent/{key}", value)
    
    def finish(self):
        """Finish all loggers."""
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.finish()
        if hasattr(self, 'tb_logger'):
            self.tb_logger.close()
