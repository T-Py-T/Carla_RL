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
    parser.add_argument(
        "--renderer",
        choices=["opencv", "threejs", "auto"],
        default="auto",
        help="Proof renderer: threejs (JevPilot WebGL) or opencv fallback",
    )
    return parser.parse_args()


def frame_from_rollout(renderer: OfflineTownRenderer, hud: FSDHUD, decision, step: int, total_reward: float):
    town_state = renderer.state
    reward = town_state.speed_kmh / 60.0
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
            speed_kmh=town_state.speed_kmh,
            town=settings.DEFAULT_TOWN,
            mode="offline",
            maneuver=settings.MANEUVER_LABELS.get(decision.action, decision.action_label),
            distance_m=town_state.distance_m,
        )
    )
    return hud.render(renderer.render()), reward


def render_before_frame(width: int, height: int) -> np.ndarray:
    renderer = OfflineTownRenderer(width=width, height=height, town=settings.DEFAULT_TOWN)
    renderer.reset()
    for _ in range(20):
        renderer.step(0)
    return renderer.render()


def rollout_frames(width: int, height: int, count: int) -> list[np.ndarray]:
    renderer = OfflineTownRenderer(width=width, height=height, town=settings.DEFAULT_TOWN)
    policy = LocalDrivingPolicy()
    hud = FSDHUD()
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
        frame, reward = frame_from_rollout(renderer, hud, decision, step, total_reward)
        total_reward += reward
        frames.append(frame)
    return frames


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
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


def try_threejs_capture(args: argparse.Namespace, mp4_path: Path) -> list[np.ndarray] | None:
    script = ROOT / "demos" / "web" / "scripts" / "capture.mjs"
    if not script.exists():
        return None
    cmd = [
        "node",
        str(script),
        "--output-dir",
        str(args.output_dir),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--video",
        str(mp4_path),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=ROOT / "demos" / "web")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    frames_dir = args.output_dir / "threejs_frames"
    if not frames_dir.exists():
        return None
    frames = []
    for path in sorted(frames_dir.glob("frame_*.png")):
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(image)
    return frames or None


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = args.output_dir / "fsd_town_playback_demo.mp4"
    gif_path = args.output_dir / "fsd_town_playback_demo.gif"
    before_path = args.output_dir / "before_plain_ego.png"
    after_path = args.output_dir / "after_fsd_overlay.png"

    rollout: list[np.ndarray] | None = None
    if args.renderer in ("threejs", "auto"):
        rollout = try_threejs_capture(args, mp4_path)

    if rollout is None:
        if args.renderer == "threejs":
            print("Three.js capture unavailable; falling back to OpenCV renderer.")
        rollout = rollout_frames(args.width, args.height, args.frames)
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

    before = render_before_frame(args.width, args.height)
    after = rollout[min(len(rollout) - 1, 89)]

    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)
    write_gif(rollout, gif_path, args.fps)

    print("Artifacts written:")
    for path in (before_path, after_path, gif_path, mp4_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
