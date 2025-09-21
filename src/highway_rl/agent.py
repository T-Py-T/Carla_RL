"""
Highway DQN Agent

Modern DQN implementation optimized for highway-env and Apple Silicon.
Features improved architecture and training stability.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from collections import deque
from typing import Tuple, Optional, Dict, Any
import os


class HighwayDQNAgent:
    """DQN Agent optimized for highway driving scenarios."""
    
    def __init__(
        self,
        state_size: Tuple[int, ...],
        action_size: int,
        learning_rate: float = 0.001,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        double_dqn: bool = True,
        dueling_dqn: bool = True,
        use_mixed_precision: bool = True
    ):
        """
        Initialize Highway DQN Agent.
        
        Args:
            state_size: Shape of state observations
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            memory_size: Size of replay buffer
            batch_size: Training batch size
            target_update_freq: Frequency of target network updates
            double_dqn: Whether to use Double DQN
            dueling_dqn: Whether to use Dueling DQN architecture
            use_mixed_precision: Whether to use mixed precision training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        
        # Mixed precision for Apple Silicon optimization
        if use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Training metrics
        self.training_step = 0
        self.loss_history = []
        
    def _build_network(self) -> keras.Model:
        """Build the neural network architecture."""
        if len(self.state_size) == 2:
            # 2D state (e.g., kinematic features)
            return self._build_dense_network()
        elif len(self.state_size) == 3:
            # 3D state (e.g., image observations)
            return self._build_cnn_network()
        else:
            raise ValueError(f"Unsupported state shape: {self.state_size}")
    
    def _build_dense_network(self) -> keras.Model:
        """Build dense network for kinematic observations."""
        inputs = keras.layers.Input(shape=self.state_size)
        
        # Flatten if needed
        if len(self.state_size) > 1:
            x = keras.layers.Flatten()(inputs)
        else:
            x = inputs
            
        # Dense layers with batch normalization
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        if self.dueling_dqn:
            # Dueling DQN architecture
            value_stream = keras.layers.Dense(64, activation='relu')(x)
            value = keras.layers.Dense(1, name='value')(value_stream)
            
            advantage_stream = keras.layers.Dense(64, activation='relu')(x)
            advantage = keras.layers.Dense(self.action_size, name='advantage')(advantage_stream)
            
            # Combine value and advantage
            mean_advantage = keras.layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
            )(advantage)
            
            q_values = keras.layers.Add()([
                value,
                keras.layers.Subtract()([advantage, mean_advantage])
            ])
        else:
            # Standard DQN
            q_values = keras.layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=q_values)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _build_cnn_network(self) -> keras.Model:
        """Build CNN network for image observations."""
        inputs = keras.layers.Input(shape=self.state_size)
        
        # Convolutional layers
        x = keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Flatten()(x)
        
        # Dense layers
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        if self.dueling_dqn:
            # Dueling architecture for CNN
            value_stream = keras.layers.Dense(256, activation='relu')(x)
            value = keras.layers.Dense(1, name='value')(value_stream)
            
            advantage_stream = keras.layers.Dense(256, activation='relu')(x)
            advantage = keras.layers.Dense(self.action_size, name='advantage')(advantage_stream)
            
            mean_advantage = keras.layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
            )(advantage)
            
            q_values = keras.layers.Add()([
                value,
                keras.layers.Subtract()([advantage, mean_advantage])
            ])
        else:
            q_values = keras.layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=q_values)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Ensure state has batch dimension
        if len(state.shape) == len(self.state_size):
            state = np.expand_dims(state, axis=0)
            
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> Dict[str, float]:
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        if self.double_dqn:
            # Double DQN: use main network to select actions, target network to evaluate
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            target_q_values = rewards + (1 - dones) * 0.95 * next_q_values[np.arange(self.batch_size), next_actions]
        else:
            # Standard DQN
            next_q_values = self.target_network.predict(next_states, verbose=0)
            target_q_values = rewards + (1 - dones) * 0.95 * np.amax(next_q_values, axis=1)
        
        # Update Q-values for taken actions
        target_q_full = current_q_values.copy()
        target_q_full[np.arange(self.batch_size), actions] = target_q_values
        
        # Train network
        history = self.q_network.fit(
            states, target_q_full,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0
        )
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update training metrics
        self.training_step += 1
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return {
            'loss': loss,
            'epsilon': self.epsilon,
            'q_mean': np.mean(current_q_values),
            'q_std': np.std(current_q_values),
            'learning_rate': float(self.q_network.optimizer.learning_rate),
            'buffer_size': len(self.memory)
        }
    
    def update_target_network(self):
        """Update target network with main network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save(self, filepath: str):
        """Save model weights and configuration."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.q_network.save_weights(f"{filepath}_weights.h5")
        
        # Save configuration
        config = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'double_dqn': self.double_dqn,
            'dueling_dqn': self.dueling_dqn,
            'training_step': self.training_step
        }
        
        import json
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, filepath: str):
        """Load model weights and configuration."""
        import json
        
        # Load configuration
        with open(f"{filepath}_config.json", 'r') as f:
            config = json.load(f)
        
        # Update agent configuration
        self.epsilon = config.get('epsilon', self.epsilon)
        self.training_step = config.get('training_step', 0)
        
        # Load weights
        self.q_network.load_weights(f"{filepath}_weights.h5")
        self.update_target_network()
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        import io
        stream = io.StringIO()
        self.q_network.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics."""
        if not self.loss_history:
            return {}
            
        recent_losses = self.loss_history[-100:]  # Last 100 training steps
        
        return {
            'total_training_steps': self.training_step,
            'current_epsilon': self.epsilon,
            'recent_mean_loss': np.mean(recent_losses),
            'recent_loss_std': np.std(recent_losses),
            'memory_utilization': len(self.memory) / self.memory_size,
            'model_parameters': self.q_network.count_params()
        }