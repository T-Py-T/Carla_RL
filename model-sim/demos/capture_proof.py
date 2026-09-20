#!/usr/bin/env python3
"""Capture before/after proof artifacts for the 3D FSD playback demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_rl.fsd_hud import FSDFrame, FSDHUD
from carla_rl.local_policy import LocalDrivingPolicy
from carla_rl.offline_town import OfflineTownRenderer
from carla_rl import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture FSD playback proof artifacts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/cursor/artifacts/carla-fsd-p0"),
        help="Directory for hosted walkthrough artifacts",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def render_before_frame(width: int, height: int) -> np.ndarray:
    """Reference-style plain ego frame without overlays (rejected PR #110 look)."""
    renderer = OfflineTownRenderer(width=width, height=height, town=settings.DEFAULT_TOWN)
    renderer.reset()
    for _ in range(20):
        renderer.step(0)
    return renderer.render()


def render_after_frame(width: int, height: int, frames: int) -> np.ndarray:
    renderer = OfflineTownRenderer(width=width, height=height, town=settings.DEFAULT_TOWN)
    policy = LocalDrivingPolicy()
    hud = FSDHUD(title="FSD Town Playback")
    renderer.reset()
    policy.reset()
    last = None
    total_reward = 0.0
    for step in range(frames):
        decision = policy.decide(
            speed_kmh=renderer.state.speed_kmh,
            lane_offset=renderer.state.lane_offset,
        )
        renderer.step(decision.action)
        reward = renderer.state.speed_kmh / 60.0
        total_reward += reward
        hud.update(
            FSDFrame(
                step=step,
                episode=1,
                reward=reward,
                total_reward=total_reward,
                action=decision.action,
                action_label=decision.action_label,
                q_values=decision.q_values,
                probabilities=decision.probabilities,
                speed_kmh=renderer.state.speed_kmh,
                town=settings.DEFAULT_TOWN,
                mode="offline",
            )
        )
        last = hud.render(renderer.render())
    assert last is not None
    return last


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio

        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        imageio.mimsave(path, rgb_frames, fps=fps)
        return
    except ImportError:
        pass

    # Pillow fallback
    from PIL import Image

    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def rollout_frames(width: int, height: int, count: int) -> list[np.ndarray]:
    renderer = OfflineTownRenderer(width=width, height=height, town=settings.DEFAULT_TOWN)
    policy = LocalDrivingPolicy()
    hud = FSDHUD(title="FSD Town Playback")
    renderer.reset()
    policy.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0
    for step in range(count):
        decision = policy.decide(
            speed_kmh=renderer.state.speed_kmh,
            lane_offset=renderer.state.lane_offset,
        )
        renderer.step(decision.action)
        reward = renderer.state.speed_kmh / 60.0
        total_reward += reward
        hud.update(
            FSDFrame(
                step=step,
                episode=1,
                reward=reward,
                total_reward=total_reward,
                action=decision.action,
                action_label=decision.action_label,
                q_values=decision.q_values,
                probabilities=decision.probabilities,
                speed_kmh=renderer.state.speed_kmh,
                town=settings.DEFAULT_TOWN,
                mode="offline",
            )
        )
        frames.append(hud.render(renderer.render()))
    return frames


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    before = render_before_frame(args.width, args.height)
    after = render_after_frame(args.width, args.height, frames=min(args.frames, 90))
    rollout = rollout_frames(args.width, args.height, args.frames)

    before_path = args.output_dir / "before_plain_ego.png"
    after_path = args.output_dir / "after_fsd_overlay.png"
    gif_path = args.output_dir / "fsd_town_playback_demo.gif"
    mp4_path = args.output_dir / "fsd_town_playback_demo.mp4"

    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)
    write_gif(rollout, gif_path, args.fps)

    demo_cmd = [
        sys.executable,
        str(ROOT / "demos" / "fsd_playback.py"),
        "--mode",
        "offline",
        "--headless",
        "--steps",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--save-video",
        str(mp4_path),
    ]
    subprocess.run(demo_cmd, check=True)

    print("Artifacts written:")
    for path in (before_path, after_path, gif_path, mp4_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
