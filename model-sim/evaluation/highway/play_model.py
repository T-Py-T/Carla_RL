#!/usr/bin/env python3
"""
Playback a trained Highway DQN checkpoint with a JevPilot-style HUD.

Visualizes local observations, DiscreteMetaAction labels, and per-action
Q-values / softmax probabilities while rolling out the policy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pygame

# Add src to path for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from highway_rl import HighwayDQNAgent, HighwayEnvironment
from highway_rl.playback_hud import PlaybackFrame, PlaybackHUD


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playback a Highway DQN checkpoint with JevPilot-style HUD"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Checkpoint path without the .weights.h5 suffix",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="highway",
        choices=["highway", "merge", "intersection", "parking", "racetrack"],
        help="Driving scenario to visualize",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to play")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Target playback FPS when rendering to a window",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Environment seed for reproducible rollouts",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for HUD probability bars",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Gymnasium render mode",
    )
    parser.add_argument(
        "--save-frame",
        type=str,
        help="Optional PNG output path for the last rendered frame",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use pygame dummy video driver for offscreen capture",
    )
    return parser.parse_args()


def resolve_model_path(explicit_model: Optional[str]) -> Path:
    if explicit_model:
        return Path(explicit_model)

    models_dir = Path("models/highway")
    if not models_dir.exists():
        raise FileNotFoundError(
            "No --model provided and models/highway does not exist. "
            "Train a checkpoint first or pass --model."
        )

    candidates = sorted(models_dir.glob("*_config.json"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoints found in models/highway. "
            "Run training first or pass --model."
        )

    latest = candidates[-1]
    return latest.with_name(latest.name.replace("_config.json", ""))


def load_agent(model_path: Path, state_size: tuple[int, ...], action_size: int) -> HighwayDQNAgent:
    agent = HighwayDQNAgent(
        state_size=state_size,
        action_size=action_size,
        use_mixed_precision=False,
    )
    agent.load(str(model_path))
    agent.epsilon = 0.0
    return agent


def capture_hud_frame(env_unwrapped: Any) -> np.ndarray:
    """Composite the simulation view and HUD panel into one RGB frame."""
    viewer = env_unwrapped.viewer
    if viewer is None:
        raise RuntimeError("Viewer is not initialized. Call env.render() first.")

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
    else:
        composite = sim_surface

    frame = pygame.surfarray.array3d(composite)
    return np.moveaxis(frame, 0, 1)


def save_rgb_frame(frame: np.ndarray, output_path: Path) -> None:
    try:
        import imageio.v3 as iio
    except ImportError:
        from PIL import Image

        Image.fromarray(frame).save(output_path)
        return

    iio.imwrite(output_path, frame)


def run_playback(args: argparse.Namespace) -> dict[str, Any]:
    if args.headless:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    model_path = resolve_model_path(args.model)
    config = HighwayEnvironment.get_playback_config(args.scenario)
    if args.headless or args.save_frame:
        config["offscreen_rendering"] = True
    env = HighwayEnvironment(
        scenario=args.scenario,
        render_mode=args.render_mode,
        config_overrides=config,
    )
    agent = load_agent(model_path, env.observation_space.shape, env.action_space.n)
    action_labels = env.get_action_labels()

    hud: Optional[PlaybackHUD] = None
    episode_summaries: list[dict[str, Any]] = []
    last_frame: Optional[np.ndarray] = None

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < args.max_steps:
            available_actions = env.get_available_actions()
            decision = agent.act_with_details(
                state,
                training=False,
                available_actions=available_actions,
                temperature=args.temperature,
            )
            action = decision["action"]
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            frame = PlaybackFrame(
                step=step,
                episode=episode,
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
                scenario=args.scenario,
                observation=state,
                info=dict(info),
            )

            if hud is None:
                env.render()
                hud = PlaybackHUD(env.get_unwrapped(), action_labels)
                hud.attach()
            else:
                hud.update(frame)
                env.env.unwrapped.viewer.set_agent_action_sequence([action])

            env.render()
            if args.save_frame or args.render_mode == "rgb_array":
                last_frame = capture_hud_frame(env.get_unwrapped())

            if args.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        truncated = True
                        break
                time.sleep(max(0.0, 1.0 / args.fps))

            state = next_state
            done = terminated or truncated

        episode_summaries.append(
            {
                "episode": episode + 1,
                "steps": step,
                "total_reward": total_reward,
                "collisions": env.episode_metrics["collisions"],
            }
        )

    if args.save_frame and last_frame is not None:
        output_path = Path(args.save_frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb_frame(last_frame, output_path)

    env.close()
    return {
        "model": str(model_path),
        "scenario": args.scenario,
        "episodes": episode_summaries,
    }


def main() -> None:
    args = parse_arguments()
    summary = run_playback(args)
    print("Playback complete")
    print(f"Model: {summary['model']}")
    print(f"Scenario: {summary['scenario']}")
    for episode in summary["episodes"]:
        print(
            f"  Episode {episode['episode']}: "
            f"{episode['steps']} steps, "
            f"reward={episode['total_reward']:.2f}, "
            f"collisions={episode['collisions']}"
        )


if __name__ == "__main__":
    main()
