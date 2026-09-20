#!/usr/bin/env python3
"""Capture JevPilot-style Three.js FSD proof artifacts (required for Taylor review)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Three.js town FSD playback proof (JevPilot aesthetic)"
    )
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


def run_threejs_capture(args: argparse.Namespace) -> list[np.ndarray]:
    script = ROOT / "demos" / "web" / "scripts" / "capture.mjs"
    web_dir = ROOT / "demos" / "web"
    if not script.exists():
        raise FileNotFoundError(f"Missing Three.js capture script: {script}")

    subprocess.run(["npm", "install", "--silent"], cwd=web_dir, check=True)

    mp4_path = args.output_dir / "fsd_town_playback_demo.mp4"
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
    subprocess.run(cmd, check=True, cwd=web_dir)

    frames_dir = args.output_dir / "threejs_frames"
    frames: list[np.ndarray] = []
    for frame_path in sorted(frames_dir.glob("frame_*.png")):
        image = cv2.imread(str(frame_path))
        if image is not None:
            frames.append(image)
    if not frames:
        raise RuntimeError("Three.js capture produced no frames")
    return frames


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rollout = run_threejs_capture(args)
    gif_path = args.output_dir / "fsd_town_playback_demo.gif"
    write_gif(rollout, gif_path, args.fps)

    print("Artifacts written (Three.js town — NOT highway-env):")
    for name in (
        "before_plain_ego.png",
        "after_fsd_overlay.png",
        "fsd_town_playback_demo.gif",
        "fsd_town_playback_demo.mp4",
    ):
        print(f"  {args.output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
