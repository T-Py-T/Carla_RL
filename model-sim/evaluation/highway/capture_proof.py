#!/usr/bin/env python3
"""
Capture before/after playback proof artifacts for PR review.

BEFORE: baseline highway-env rollout (no JevPilot HUD overlays).
AFTER:  JevPilot-style HUD playback with candidate paths and inspector.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from highway_rl import HighwayDQNAgent, HighwayEnvironment
from highway_rl.playback_hud import PlaybackFrame, PlaybackHUD


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture before/after playback proof")
    parser.add_argument(
        "--model",
        type=str,
        default="models/highway/final_model",
        help="Checkpoint path without .weights.h5 suffix",
    )
    parser.add_argument("--scenario", type=str, default="highway")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/cursor/artifacts/playback-proof",
        help="Directory for frames and GIFs",
    )
    parser.add_argument("--gif-duration-ms", type=int, default=350)
    return parser.parse_args()


def load_agent(
    model_path: Path, state_size: tuple[int, ...], action_size: int
) -> HighwayDQNAgent:
    agent = HighwayDQNAgent(
        state_size=state_size,
        action_size=action_size,
        use_mixed_precision=False,
    )
    agent.load(str(model_path))
    agent.epsilon = 0.0
    return agent


def capture_sim_frame(env_unwrapped: Any) -> np.ndarray:
    viewer = env_unwrapped.viewer
    if viewer is None:
        raise RuntimeError("Viewer not initialized")
    viewer.enabled = True
    viewer.display()
    frame = pygame.surfarray.array3d(viewer.sim_surface)
    return np.moveaxis(frame, 0, 1)


def capture_hud_frame(env_unwrapped: Any) -> np.ndarray:
    viewer = env_unwrapped.viewer
    if viewer is None:
        raise RuntimeError("Viewer not initialized")
    viewer.enabled = True
    viewer.display()

    sim_surface = viewer.sim_surface
    width, sim_height = sim_surface.get_size()
    if viewer.agent_surface is not None:
        agent_surface = viewer.agent_surface
        total_height = sim_height + agent_surface.get_height()
        composite = pygame.Surface((width, total_height))
        composite.blit(sim_surface, (0, 0))
        composite.blit(agent_surface, (0, sim_height))
        source = composite
    else:
        source = sim_surface

    frame = pygame.surfarray.array3d(source)
    return np.moveaxis(frame, 0, 1)


def save_png(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def save_gif(frames: List[np.ndarray], path: Path, duration_ms: int) -> None:
    from PIL import Image

    if not frames:
        raise ValueError("No frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def rollout_baseline(
    env: HighwayEnvironment,
    agent: HighwayDQNAgent,
    seed: int,
    max_steps: int,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    state, _ = env.reset(seed=seed)
    done = False
    step = 0

    while not done and step < max_steps:
        action = agent.act(state, training=False)
        state, _, terminated, truncated, _ = env.step(action)
        step += 1
        env.render()
        frames.append(capture_sim_frame(env.get_unwrapped()))
        done = terminated or truncated

    return frames


def rollout_with_hud(
    env: HighwayEnvironment,
    agent: HighwayDQNAgent,
    seed: int,
    max_steps: int,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    action_labels = env.get_action_labels()
    hud: Optional[PlaybackHUD] = None

    state, _ = env.reset(seed=seed)
    done = False
    step = 0
    total_reward = 0.0

    while not done and step < max_steps:
        available_actions = env.get_available_actions()
        decision = agent.act_with_details(
            state,
            training=False,
            available_actions=available_actions,
        )
        action = decision["action"]
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        frame = PlaybackFrame(
            step=step,
            reward=reward,
            total_reward=total_reward,
            action=action,
            action_label=action_labels[action],
            q_values=decision["q_values"],
            probabilities=decision["probabilities"],
            available_actions=available_actions,
            explored=decision["explored"],
            epsilon=decision["epsilon"],
            speed=float(info.get("speed", 0.0)),
            crashed=bool(info.get("crashed", False)),
            scenario=env.scenario,
            observation=state,
            info=dict(info),
        )

        if hud is None:
            env.render()
            hud = PlaybackHUD(env.get_unwrapped(), action_labels)
            hud.attach()
        else:
            hud.update(frame)
            env.get_unwrapped().viewer.set_agent_action_sequence([action])

        env.render()
        frames.append(capture_hud_frame(env.get_unwrapped()))
        state = state
        done = terminated or truncated

    return frames


def main() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    before_dir = output_dir / "before_frames"
    after_dir = output_dir / "after_frames"

    model_path = Path(args.model)
    if not model_path.with_suffix(".weights.h5").exists() and not (
        Path(str(model_path) + ".weights.h5").exists()
    ):
        weights = Path(f"{model_path}.weights.h5")
        if not weights.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    baseline_config = HighwayEnvironment.get_optimized_config(args.scenario)
    baseline_config["offscreen_rendering"] = True

    hud_config = HighwayEnvironment.get_playback_config(args.scenario)
    hud_config["offscreen_rendering"] = True

    baseline_env = HighwayEnvironment(
        scenario=args.scenario,
        render_mode="rgb_array",
        config_overrides=baseline_config,
    )
    hud_env = HighwayEnvironment(
        scenario=args.scenario,
        render_mode="rgb_array",
        config_overrides=hud_config,
    )

    agent_baseline = load_agent(
        model_path,
        baseline_env.observation_space.shape,
        baseline_env.action_space.n,
    )
    agent_hud = load_agent(
        model_path,
        hud_env.observation_space.shape,
        hud_env.action_space.n,
    )

    print("Capturing BEFORE (baseline highway-env playback)...")
    before_frames = rollout_baseline(
        baseline_env, agent_baseline, args.seed, args.max_steps
    )
    baseline_env.close()

    print("Capturing AFTER (JevPilot-style HUD playback)...")
    after_frames = rollout_with_hud(hud_env, agent_hud, args.seed, args.max_steps)
    hud_env.close()

    for index, frame in enumerate(before_frames, start=1):
        save_png(frame, before_dir / f"frame_{index:02d}.png")
    for index, frame in enumerate(after_frames, start=1):
        save_png(frame, after_dir / f"frame_{index:02d}.png")

    before_gif = output_dir / "playback_before.gif"
    after_gif = output_dir / "playback_after.gif"
    save_gif(before_frames, before_gif, args.gif_duration_ms)
    save_gif(after_frames, after_gif, args.gif_duration_ms)

    print(f"BEFORE frames: {len(before_frames)} -> {before_dir}")
    print(f"AFTER frames:  {len(after_frames)} -> {after_dir}")
    print(f"BEFORE GIF: {before_gif}")
    print(f"AFTER GIF:  {after_gif}")
    print(
        "Stats:",
        f"before shape={before_frames[0].shape}",
        f"after shape={after_frames[0].shape}",
        f"before mean={before_frames[-1].mean():.1f}",
        f"after mean={after_frames[-1].mean():.1f}",
    )


if __name__ == "__main__":
    main()
