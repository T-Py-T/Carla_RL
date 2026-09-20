"""FSD / JevPilot-style decision HUD overlaid on ego camera frames."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

import cv2
import numpy as np

from . import settings
from .jevpilot_theme import (
    ACCENT_RED,
    TEXT_FAINT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    blend_glass,
    put_text,
)


@dataclass
class FSDFrame:
    step: int = 0
    episode: int = 0
    reward: float = 0.0
    total_reward: float = 0.0
    action: int = 0
    action_label: str = "FORWARD"
    q_values: np.ndarray = field(default_factory=lambda: np.zeros(4))
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(4))
    speed_kmh: float = 0.0
    speed_limit_kmh: int = 50
    crashed: bool = False
    town: str = "Town03"
    mode: str = "offline"
    maneuver: str = "Continue straight"
    distance_m: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

    def to_inspector_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "town": self.town,
            "mode": self.mode,
            "maneuver": self.maneuver,
            "action": {
                "index": self.action,
                "label": self.action_label,
            },
            "reward": {
                "step": round(self.reward, 4),
                "total": round(self.total_reward, 4),
            },
            "vehicle": {
                "speed_kmh": round(self.speed_kmh, 2),
                "speed_limit_kmh": self.speed_limit_kmh,
                "crashed": self.crashed,
            },
            "q_values": {
                str(i): round(float(v), 4) for i, v in enumerate(self.q_values)
            },
            "probabilities": {
                str(i): round(float(v), 4) for i, v in enumerate(self.probabilities)
            },
            "info": self.info,
        }


class FSDHUD:
    """JevPilot-inspired HUD: `.navigation-hud`, `.bottom-hud`, JSON inspector."""

    def __init__(self, title: str = "Carla RL · FSD Playback") -> None:
        self.title = title
        self.frame = FSDFrame()

    def update(self, frame: FSDFrame) -> None:
        self.frame = frame

    def render(self, camera_bgr: np.ndarray) -> np.ndarray:
        from .path_projection import draw_candidate_paths, draw_lane_centerline

        composed = draw_lane_centerline(camera_bgr.copy())
        composed = draw_candidate_paths(
            composed,
            selected_action=self.frame.action,
            probabilities=self.frame.probabilities,
        )
        composed = self._draw_topbar(composed)
        composed = self._draw_navigation_hud(composed)
        composed = self._draw_bottom_hud(composed)
        composed = self._draw_json_inspector(composed)
        return composed

    def _draw_topbar(self, image: np.ndarray) -> np.ndarray:
        blend_glass(image, (22, 18, 290, 66))
        put_text(image, self.title, (34, 44), scale=0.52, color=TEXT_PRIMARY, thickness=1)
        put_text(
            image,
            f"{self.frame.town} · local policy",
            (34, 58),
            scale=0.34,
            color=TEXT_MUTED,
        )
        return image

    def _draw_navigation_hud(self, image: np.ndarray) -> np.ndarray:
        w = image.shape[1]
        x0 = w - 27 - 330
        y0 = 18
        x1 = w - 27
        y1 = y0 + 68
        blend_glass(image, (x0, y0, x1, y1))

        # Turn icon block
        cv2.rectangle(image, (x0 + 12, y0 + 16), (x0 + 42, y0 + 46), (245, 245, 245), -1)
        arrow = "^" if self.frame.action == 0 else ("<" if self.frame.action == 1 else ">")
        put_text(image, arrow, (x0 + 22, y0 + 40), scale=0.55, color=TEXT_PRIMARY)

        put_text(
            image,
            self.frame.maneuver[:28],
            (x0 + 52, y0 + 36),
            scale=0.48,
            color=TEXT_PRIMARY,
        )
        put_text(
            image,
            f"{self.frame.distance_m:.0f} m ahead",
            (x0 + 52, y0 + 52),
            scale=0.34,
            color=TEXT_FAINT,
        )

        cv2.line(image, (x1 - 92, y0 + 18), (x1 - 92, y0 + 50), (247, 247, 247), 1)
        put_text(
            image,
            f"{max(0, int(420 - self.frame.distance_m))} m left",
            (x1 - 84, y0 + 40),
            scale=0.34,
            color=TEXT_MUTED,
        )
        return image

    def _draw_bottom_hud(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        dock_w = min(760, w - 32)
        x0 = (w - dock_w) // 2
        y0 = h - 17 - 76
        x1 = x0 + dock_w
        y1 = h - 17
        blend_glass(image, (x0, y0, x1, y1))

        # Speed cluster
        speed_text = f"{int(round(self.frame.speed_kmh))}"
        put_text(image, speed_text, (x0 + 18, y0 + 52), scale=1.35, color=TEXT_PRIMARY, thickness=2)
        put_text(image, "km/h", (x0 + 24, y0 + 66), scale=0.32, color=TEXT_MUTED)

        # Speed limit badge
        limit_x = x0 + 88
        cv2.rectangle(image, (limit_x, y0 + 22), (limit_x + 34, y0 + 52), (255, 255, 255), -1)
        cv2.rectangle(image, (limit_x, y0 + 22), (limit_x + 34, y0 + 52), TEXT_PRIMARY, 1)
        put_text(image, "LIMIT", (limit_x + 4, y0 + 32), scale=0.28, color=TEXT_FAINT)
        put_text(
            image,
            str(self.frame.speed_limit_kmh),
            (limit_x + 8, y0 + 48),
            scale=0.55,
            color=TEXT_PRIMARY,
        )

        cv2.line(image, (x0 + 136, y0 + 20), (x0 + 136, y0 + 58), (247, 247, 247), 1)

        # Pilot / decision status
        state = "Local autopilot" if self.frame.mode != "offline" else "Local playback"
        put_text(image, state, (x0 + 150, y0 + 36), scale=0.42, color=TEXT_PRIMARY)
        put_text(
            image,
            f"{self.frame.action_label} · p={self.frame.probabilities[self.frame.action]:.0%}",
            (x0 + 150, y0 + 54),
            scale=0.36,
            color=TEXT_MUTED,
        )

        cv2.line(image, (x1 - 250, y0 + 20), (x1 - 250, y0 + 58), (247, 247, 247), 1)

        # Candidate toggles (visual only — mirrors JevPilot dock tools)
        labels = ["FORWARD", "LEFT", "RIGHT", "BRAKE"]
        bx = x1 - 236
        for idx, label in enumerate(labels):
            selected = idx == self.frame.action
            color = ACCENT_RED if selected else TEXT_MUTED
            cv2.rectangle(
                image,
                (bx, y0 + 26),
                (bx + 52, y0 + 50),
                (255, 255, 255) if selected else (245, 245, 245),
                -1,
            )
            cv2.rectangle(image, (bx, y0 + 26), (bx + 52, y0 + 50), color if selected else (230, 230, 230), 1)
            put_text(image, label[:3], (bx + 8, y0 + 44), scale=0.32, color=color if selected else TEXT_MUTED)
            bx += 58

        return image

    def _draw_json_inspector(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        panel_w = 320
        panel_h = 190
        x0 = w - panel_w - 24
        y0 = 96
        blend_glass(image, (x0, y0, x0 + panel_w, y0 + panel_h))

        put_text(image, "{} Under the hood", (x0 + 12, y0 + 22), scale=0.42, color=TEXT_PRIMARY)
        put_text(image, "LIVE · local signals", (x0 + 170, y0 + 22), scale=0.30, color=TEXT_FAINT)

        json_text = json.dumps(self.frame.to_inspector_dict(), indent=2)
        lines = json_text.splitlines()[:10]
        ty = y0 + 42
        for line in lines:
            color = TEXT_MUTED
            stripped = line.strip()
            if stripped.startswith('"') and stripped.endswith('":'):
                color = (180, 120, 60)  # json-key-ish
            elif any(token in stripped for token in ("true", "false", "null")):
                color = (160, 110, 90)
            elif stripped and stripped[0].isdigit():
                color = (130, 90, 50)
            put_text(image, line[:38], (x0 + 10, ty), scale=0.31, color=color)
            ty += 14
        return image
