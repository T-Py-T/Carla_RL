#!/usr/bin/env python3
"""3D town ego-camera FSD playback demo (CARLA or offline fallback)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carla_rl.env import CarlaPlaybackEnv
from carla_rl.fsd_hud import FSDFrame, FSDHUD
from carla_rl.local_policy import LocalDrivingPolicy
from carla_rl.offline_town import OfflineTownRenderer
from carla_rl import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D town ego-camera playback with projected paths and decision UI"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "carla", "offline"],
        default="auto",
        help="auto tries CARLA then falls back to offline renderer",
    )
    parser.add_argument("--host", default=settings.CARLA_HOST)
    parser.add_argument("--port", type=int, default=settings.CARLA_PORT)
    parser.add_argument("--town", default=settings.DEFAULT_TOWN)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=settings.IMG_WIDTH)
    parser.add_argument("--height", type=int, default=settings.IMG_HEIGHT)
    parser.add_argument("--headless", action="store_true", help="Do not open a GUI window")
    parser.add_argument("--save-frame", type=str, help="Optional PNG path for the last frame")
    parser.add_argument("--save-video", type=str, help="Optional MP4 path for the rollout")
    return parser.parse_args()


def try_connect_carla(args: argparse.Namespace) -> CarlaPlaybackEnv | None:
    if args.mode == "offline":
        return None
    if not CarlaPlaybackEnv.is_available():
        if args.mode == "carla":
            raise RuntimeError("carla Python API not installed")
        return None
    env = CarlaPlaybackEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        img_width=args.width,
        img_height=args.height,
    )
    try:
        env.connect()
        env.reset()
        return env
    except Exception as exc:  # noqa: BLE001 - demo fallback path
        if args.mode == "carla":
            raise RuntimeError(f"Could not connect to CARLA at {args.host}:{args.port}") from exc
        print(f"CARLA unavailable ({exc}); using offline town renderer.")
        env.close()
        return None


def run_offline(args: argparse.Namespace) -> int:
    renderer = OfflineTownRenderer(width=args.width, height=args.height, town=args.town)
    policy = LocalDrivingPolicy()
    hud = FSDHUD(title="FSD Town Playback (offline)")
    writer = _open_video_writer(args.save_video, args.width, args.height, args.fps)
    last_frame = None
    total_reward = 0.0

    for episode in range(args.episodes):
        renderer.reset()
        policy.reset()
        total_reward = 0.0
        for step in range(args.steps):
            decision = policy.decide(
                speed_kmh=renderer.state.speed_kmh,
                lane_offset=renderer.state.lane_offset,
                obstacle_ahead=step % 97 == 40,
            )
            town_state = renderer.step(decision.action)
            camera = renderer.render()
            reward = town_state.speed_kmh / 60.0
            total_reward += reward

            hud.update(
                FSDFrame(
                    step=step,
                    episode=episode + 1,
                    reward=reward,
                    total_reward=total_reward,
                    action=decision.action,
                    action_label=decision.action_label,
                    q_values=decision.q_values,
                    probabilities=decision.probabilities,
                    speed_kmh=town_state.speed_kmh,
                    town=args.town,
                    mode="offline",
                    info={"distance_m": round(town_state.distance_m, 2)},
                )
            )
            last_frame = hud.render(camera)
            if writer is not None:
                writer.write(last_frame)
            if not args.headless:
                cv2.imshow("FSD Town Playback", last_frame)
                if cv2.waitKey(int(max(1000 / args.fps, 1))) & 0xFF == ord("q"):
                    if writer is not None:
                        writer.release()
                    cv2.destroyAllWindows()
                    return 0
            else:
                time.sleep(1.0 / args.fps)

    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()
    if args.save_frame and last_frame is not None:
        cv2.imwrite(args.save_frame, last_frame)
    return 0


def run_carla(args: argparse.Namespace, env: CarlaPlaybackEnv) -> int:
    policy = LocalDrivingPolicy()
    hud = FSDHUD(title="FSD Town Playback (CARLA)")
    writer = _open_video_writer(args.save_video, args.width, args.height, args.fps)
    last_frame = None
    total_reward = 0.0

    try:
        for episode in range(args.episodes):
            env.reset()
            policy.reset()
            total_reward = 0.0
            for step in range(args.steps):
                obs = env.get_observation()
                speed = 0.0
                lane_offset = 0.0
                if obs is not None and hasattr(env, "vehicle") and env.vehicle is not None:
                    velocity = env.vehicle.get_velocity()
                    speed = 3.6 * float(np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
                    lane_offset = env.vehicle.get_transform().location.y % 3.5 - 1.75

                decision = policy.decide(speed_kmh=speed, lane_offset=lane_offset)
                _, reward, done, info = env.step(decision.action)
                total_reward += reward
                hud.update(
                    FSDFrame(
                        step=step,
                        episode=episode + 1,
                        reward=reward,
                        total_reward=total_reward,
                        action=decision.action,
                        action_label=decision.action_label,
                        q_values=decision.q_values,
                        probabilities=decision.probabilities,
                        speed_kmh=info.get("speed_kmh", speed),
                        crashed=info.get("crashed", False),
                        town=args.town,
                        mode="carla",
                        info=info,
                    )
                )
                last_frame = hud.render(env.get_observation())
                if writer is not None:
                    writer.write(last_frame)
                if not args.headless:
                    cv2.imshow("FSD Town Playback", last_frame)
                    if cv2.waitKey(int(max(1000 / args.fps, 1))) & 0xFF == ord("q"):
                        return 0
                else:
                    time.sleep(1.0 / args.fps)
                if done:
                    break
    finally:
        if writer is not None:
            writer.release()
        env.close()
        if not args.headless:
            cv2.destroyAllWindows()
    if args.save_frame and last_frame is not None:
        cv2.imwrite(args.save_frame, last_frame)
    return 0


def _open_video_writer(path: str | None, width: int, height: int, fps: float):
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def main() -> int:
    args = parse_args()
    env = try_connect_carla(args)
    if env is None:
        return run_offline(args)
    return run_carla(args, env)


if __name__ == "__main__":
    raise SystemExit(main())
