"""
Highway-env Reinforcement Learning Package

Modern RL training optimized for Apple Silicon using highway-env simulator.
This is the primary training platform for macOS development.
"""

from .agent import HighwayDQNAgent
from .environment import HighwayEnvironment
from .trainer import HighwayTrainer
from .logger import WandBLogger
from .playback_hud import PlaybackFrame, PlaybackHUD

__all__ = [
    "HighwayDQNAgent",
    "HighwayEnvironment",
    "HighwayTrainer",
    "WandBLogger",
    "PlaybackFrame",
    "PlaybackHUD",
]

__version__ = '1.0.0'
