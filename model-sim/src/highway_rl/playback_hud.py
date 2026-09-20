"""
JevPilot-style playback HUD for highway-env DQN rollouts.

Renders candidate action trajectories, a navigation strip, and a Q-value /
probability inspector using highway-env's pygame viewer hooks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pygame

from highway_env.vehicle.graphics import VehicleGraphics


ACTION_COLORS = {
    "LANE_LEFT": (80, 180, 255),
    "IDLE": (180, 180, 180),
    "LANE_RIGHT": (80, 180, 255),
    "FASTER": (90, 220, 120),
    "SLOWER": (255, 170, 70),
}

PANEL_BG = (18, 22, 30)
PANEL_BORDER = (70, 90, 120)
TEXT_PRIMARY = (235, 240, 248)
TEXT_MUTED = (150, 165, 185)
SELECTED = (0, 255, 220)
ILLEGAL = (90, 95, 105)


@dataclass
class PlaybackFrame:
    """Signals captured for one rollout step."""

    step: int = 0
    episode: int = 0
    reward: float = 0.0
    total_reward: float = 0.0
    action: int = 1
    action_label: str = "IDLE"
    q_values: np.ndarray = field(default_factory=lambda: np.zeros(5))
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(5))
    available_actions: List[int] = field(default_factory=list)
    explored: bool = False
    epsilon: float = 0.0
    speed: float = 0.0
    crashed: bool = False
    scenario: str = "highway"
    observation: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_inspector_dict(self) -> Dict[str, Any]:
        """Serialize the current frame for the JSON inspector panel."""
        obs_summary: Dict[str, Any] = {}
        if self.observation is not None:
            obs = np.asarray(self.observation)
            obs_summary = {
                "shape": list(obs.shape),
                "mean": float(np.mean(obs)),
                "std": float(np.std(obs)),
                "nonzero": int(np.count_nonzero(obs)),
            }

        return {
            "episode": self.episode,
            "step": self.step,
            "scenario": self.scenario,
            "action": {
                "index": self.action,
                "label": self.action_label,
                "explored": self.explored,
                "epsilon": round(self.epsilon, 4),
            },
            "reward": {
                "step": round(self.reward, 4),
                "total": round(self.total_reward, 4),
            },
            "vehicle": {
                "speed_mps": round(self.speed, 2),
                "crashed": self.crashed,
            },
            "q_values": {
                str(index): round(float(value), 4)
                for index, value in enumerate(self.q_values)
            },
            "probabilities": {
                str(index): round(float(value), 4)
                for index, value in enumerate(self.probabilities)
            },
            "available_actions": self.available_actions,
            "observation": obs_summary,
            "info": self.info,
        }


class PlaybackHUD:
    """Stateful HUD renderer for highway-env playback."""

    def __init__(
        self,
        env_unwrapped: Any,
        action_labels: Dict[int, str],
        title: str = "JevPilot Playback HUD",
    ) -> None:
        self.env = env_unwrapped
        self.action_labels = action_labels
        self.title = title
        self.frame = PlaybackFrame()
        self._fonts_initialized = False
        self._font_title: Optional[pygame.font.Font] = None
        self._font_body: Optional[pygame.font.Font] = None
        self._font_mono: Optional[pygame.font.Font] = None

    def update(self, frame: PlaybackFrame) -> None:
        """Store the latest rollout signals for rendering."""
        self.frame = frame

    def attach(self) -> None:
        """Register the HUD with highway-env's viewer."""
        viewer = self.env.viewer
        if viewer is None:
            raise RuntimeError("Viewer is not initialized. Call env.render() once first.")
        viewer.set_agent_display(self._display_callback)

    def make_display_callback(self) -> Callable:
        """Return a callback suitable for EnvViewer.set_agent_display."""
        return self._display_callback

    def _ensure_fonts(self) -> None:
        if self._fonts_initialized:
            return
        pygame.font.init()
        self._font_title = pygame.font.SysFont("dejavusansmono", 16, bold=True)
        self._font_body = pygame.font.SysFont("dejavusans", 14)
        self._font_mono = pygame.font.SysFont("dejavusansmono", 12)
        self._fonts_initialized = True

    def _display_callback(self, agent_surface: pygame.Surface, sim_surface: Any) -> None:
        self._ensure_fonts()
        self._draw_candidate_paths(sim_surface)
        self._draw_nav_strip(sim_surface)
        self._draw_inspector(agent_surface)

    def _draw_candidate_paths(self, sim_surface: Any) -> None:
        vehicle = self.env.vehicle
        action_type = self.env.action_type
        if vehicle is None or action_type is None:
            return

        policy_dt = 1 / self.env.config["policy_frequency"]
        sim_dt = 1 / self.env.config["simulation_frequency"]
        trajectory_dt = policy_dt / 3

        legal = set(self.frame.available_actions or list(self.action_labels))
        q_values = np.asarray(self.frame.q_values)
        legal_q = [q_values[index] for index in legal]
        q_min = min(legal_q) if legal_q else float(np.min(q_values))
        q_max = max(legal_q) if legal_q else float(np.max(q_values))
        q_span = max(q_max - q_min, 1e-6)

        for action_index, label in self.action_labels.items():
            is_selected = action_index == self.frame.action
            is_legal = action_index in legal
            base_color = ACTION_COLORS.get(label, (200, 200, 200))

            if not is_legal:
                color = ILLEGAL
                width = 1
            else:
                normalized = (q_values[action_index] - q_min) / q_span
                color = self._blend(base_color, (255, 255, 255), 0.15 + 0.55 * normalized)
                width = 4 if is_selected else 2

            meta_action = action_type.actions[action_index]
            trajectory = vehicle.predict_trajectory(
                [meta_action],
                policy_dt,
                trajectory_dt,
                sim_dt,
            )
            if is_selected:
                VehicleGraphics.display_trajectory(
                    trajectory, sim_surface, offscreen=self.env.config["offscreen_rendering"]
                )
            self._draw_trajectory_line(sim_surface, trajectory, color, width)

    def _draw_trajectory_line(
        self,
        sim_surface: Any,
        trajectory: List[Any],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        if len(trajectory) < 2:
            return
        points = [
            sim_surface.pos2pix(state.position[0], state.position[1])
            for state in trajectory
        ]
        pygame.draw.lines(sim_surface, color, False, points, width)

    def _draw_nav_strip(self, sim_surface: Any) -> None:
        width, height = sim_surface.get_size()
        strip_height = 34
        strip = pygame.Surface((width, strip_height), pygame.SRCALPHA)
        strip.fill((10, 14, 22, 210))
        sim_surface.blit(strip, (0, height - strip_height))

        status = "CRASH" if self.frame.crashed else "OK"
        status_color = (255, 90, 90) if self.frame.crashed else (110, 220, 140)
        nav_text = (
            f"Ep {self.frame.episode + 1}  "
            f"Step {self.frame.step}  "
            f"Speed {self.frame.speed:.1f} m/s  "
            f"R {self.frame.total_reward:.2f}  "
            f"Action {self.frame.action_label}"
        )
        if self.frame.explored:
            nav_text += f"  explore ε={self.frame.epsilon:.2f}"

        label = self._font_body.render(nav_text, True, TEXT_PRIMARY)
        sim_surface.blit(label, (10, height - strip_height + 8))

        status_label = self._font_body.render(status, True, status_color)
        sim_surface.blit(status_label, (width - status_label.get_width() - 12, height - strip_height + 8))

    def _draw_inspector(self, agent_surface: pygame.Surface) -> None:
        width, height = agent_surface.get_size()
        agent_surface.fill(PANEL_BG)
        pygame.draw.rect(agent_surface, PANEL_BORDER, agent_surface.get_rect(), 1)

        y = 8
        title = self._font_title.render(self.title, True, SELECTED)
        agent_surface.blit(title, (12, y))
        y += 24

        scenario = self._font_body.render(
            f"Scenario: {self.frame.scenario}", True, TEXT_MUTED
        )
        agent_surface.blit(scenario, (12, y))
        y += 22

        bar_left = 12
        bar_width = max(120, width // 3)
        bar_height = 14
        for action_index, label in self.action_labels.items():
            q_value = float(self.frame.q_values[action_index])
            probability = float(self.frame.probabilities[action_index])
            is_selected = action_index == self.frame.action
            is_legal = action_index in (self.frame.available_actions or [])
            name_color = SELECTED if is_selected else TEXT_PRIMARY
            if not is_legal:
                name_color = TEXT_MUTED

            name = self._font_body.render(label, True, name_color)
            agent_surface.blit(name, (bar_left, y))

            track_x = bar_left + 110
            pygame.draw.rect(
                agent_surface,
                (35, 42, 56),
                (track_x, y + 2, bar_width, bar_height),
                border_radius=3,
            )
            fill_width = int(bar_width * max(0.0, min(1.0, probability)))
            fill_color = ACTION_COLORS.get(label, (180, 180, 180))
            if is_selected:
                fill_color = SELECTED
            if fill_width > 0:
                pygame.draw.rect(
                    agent_surface,
                    fill_color,
                    (track_x, y + 2, fill_width, bar_height),
                    border_radius=3,
                )

            stats = self._font_mono.render(
                f"Q {q_value:+.3f}  P {probability:.2f}",
                True,
                TEXT_MUTED,
            )
            agent_surface.blit(stats, (track_x + bar_width + 10, y + 1))
            y += 22

        y += 6
        json_header = self._font_body.render("Inspector (JSON)", True, TEXT_PRIMARY)
        agent_surface.blit(json_header, (12, y))
        y += 18

        inspector = json.dumps(self.frame.to_inspector_dict(), indent=2)
        json_x = 12
        json_width = width - 24
        for line in inspector.splitlines():
            if y > height - 14:
                break
            wrapped = self._wrap_text(line, json_width)
            for wrapped_line in wrapped:
                if y > height - 14:
                    break
                rendered = self._font_mono.render(wrapped_line, True, TEXT_MUTED)
                agent_surface.blit(rendered, (json_x, y))
                y += 14

    @staticmethod
    def _blend(
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
        ratio: float,
    ) -> tuple[int, int, int]:
        ratio = max(0.0, min(1.0, ratio))
        return tuple(
            int(color_a[index] * (1 - ratio) + color_b[index] * ratio)
            for index in range(3)
        )

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        if not text:
            return [""]
        words = text.split(" ")
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self._font_mono.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines
