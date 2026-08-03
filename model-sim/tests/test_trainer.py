"""Focused behavioral tests for the refactored Highway trainer loop."""

from pathlib import Path
import sys

import numpy as np

# The project's current editable package metadata does not expose ``src`` on
# sys.path, so tests follow the existing cross-platform test's direct-run setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from highway_rl.trainer import HighwayTrainer


class FakeAgent:
    """Minimal agent implementing the public trainer collaboration surface."""

    dueling_dqn = False
    double_dqn = True
    epsilon = 0.1

    def __init__(self):
        self.experiences = []
        self.saved_paths = []

    def act(self, state, training=True):
        return 0

    def remember(self, state, action, reward, next_state, done):
        self.experiences.append((state, action, reward, next_state, done))

    def replay(self):
        return {"loss": 0.25}

    def get_training_metrics(self):
        return {"epsilon": self.epsilon}

    def save(self, path):
        self.saved_paths.append(Path(path))


class FakeEnvironment:
    """Two-step parking scenario suitable for training and evaluation."""

    scenario = "parking"

    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.array([0.0]), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps == 2
        return np.array([float(self.steps)]), 1.0, terminated, False, {"crashed": False}

    def get_episode_summary(self):
        return {"collisions": 0}


def test_train_runs_refactored_episode_and_persists_final_metadata(tmp_path):
    agent = FakeAgent()
    trainer = HighwayTrainer(
        agent=agent,
        environment=FakeEnvironment(),
        save_dir=str(tmp_path),
        evaluation_episodes=1,
        save_frequency=100,
    )

    summary = trainer.train(episodes=1, max_steps_per_episode=5)

    assert summary["episodes_trained"] == 1
    assert summary["total_steps"] == 2
    assert summary["mean_episode_length"] == 2
    assert summary["final_evaluation"]["success_rate"] == 1.0
    assert len(agent.experiences) == 2
    assert agent.saved_paths == [tmp_path / "final_model"]
    assert (tmp_path / "final_model_metadata.json").exists()


def test_train_handles_zero_requested_episodes(tmp_path):
    trainer = HighwayTrainer(
        agent=FakeAgent(),
        environment=FakeEnvironment(),
        save_dir=str(tmp_path),
        evaluation_episodes=1,
    )

    summary = trainer.train(episodes=0)

    assert summary["episodes_trained"] == 0
    assert summary["total_steps"] == 0
    assert summary["mean_episode_length"] == 0
