"""Tests for playback HUD helpers and Q-value diagnostics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from highway_rl.agent import HighwayDQNAgent
from highway_rl.playback_hud import PlaybackFrame


def test_get_q_values_returns_action_vector():
    agent = HighwayDQNAgent(
        state_size=(15, 6),
        action_size=5,
        use_mixed_precision=False,
    )
    state = np.zeros((15, 6), dtype=np.float32)
    q_values = agent.get_q_values(state)

    assert q_values.shape == (5,)
    assert np.all(np.isfinite(q_values))


def test_act_with_details_masks_illegal_actions():
    agent = HighwayDQNAgent(
        state_size=(15, 6),
        action_size=5,
        use_mixed_precision=False,
    )
    agent.epsilon = 0.0
    state = np.zeros((15, 6), dtype=np.float32)

    decision = agent.act_with_details(
        state,
        training=False,
        available_actions=[1, 3],
    )

    assert decision["action"] in {1, 3}
    assert decision["probabilities"].shape == (5,)
    assert np.isclose(np.sum(decision["probabilities"][1:4:2]), 1.0)


def test_playback_frame_inspector_json():
    frame = PlaybackFrame(
        step=4,
        episode=0,
        action=3,
        action_label="FASTER",
        q_values=np.array([0.1, 0.2, 0.3, 0.9, 0.4]),
        probabilities=np.array([0.05, 0.1, 0.1, 0.7, 0.05]),
        available_actions=[1, 3, 4],
        observation=np.ones((15, 6)),
    )
    payload = frame.to_inspector_dict()

    assert payload["action"]["label"] == "FASTER"
    assert payload["q_values"]["3"] == 0.9
    assert payload["available_actions"] == [1, 3, 4]
    assert payload["observation"]["shape"] == [15, 6]
