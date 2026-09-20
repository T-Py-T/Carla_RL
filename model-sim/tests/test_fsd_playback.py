"""Tests for FSD HUD and offline town renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carla_rl.fsd_hud import FSDFrame, FSDHUD
from carla_rl.local_policy import LocalDrivingPolicy
from carla_rl.offline_town import OfflineTownRenderer
from carla_rl.path_projection import candidate_path_points, draw_candidate_paths, project_vehicle_points


def test_local_policy_returns_probabilities():
    policy = LocalDrivingPolicy()
    decision = policy.decide(speed_kmh=30.0, lane_offset=0.1)
    assert decision.action in range(4)
    assert np.isclose(decision.probabilities.sum(), 1.0, atol=1e-5)
    assert len(decision.q_values) == 4


def test_offline_renderer_produces_camera_frame():
    renderer = OfflineTownRenderer(width=320, height=180)
    renderer.reset()
    renderer.step(0)
    frame = renderer.render()
    assert frame.shape == (180, 320, 3)
    assert frame.mean() > 0


def test_path_projection_draws_on_frame():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    points = candidate_path_points(0)
    projected = project_vehicle_points(points, width=320, height=180)
    assert len(projected) >= 2
    overlaid = draw_candidate_paths(frame, 0, [0.4, 0.2, 0.2, 0.2])
    assert overlaid.sum() > 0


def test_fsd_hud_render_includes_panels():
    renderer = OfflineTownRenderer(width=640, height=360)
    hud = FSDHUD()
    policy = LocalDrivingPolicy()
    renderer.reset()
    decision = policy.decide(speed_kmh=25.0)
    renderer.step(decision.action)
    hud.update(
        FSDFrame(
            action=decision.action,
            action_label=decision.action_label,
            q_values=decision.q_values,
            probabilities=decision.probabilities,
            speed_kmh=renderer.state.speed_kmh,
        )
    )
    composed = hud.render(renderer.render())
    assert composed.shape == (360, 640, 3)
    # JevPilot glass bottom HUD should brighten the centered dock strip.
    assert composed[-24, 320, 0] > 180
