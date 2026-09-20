"""FSD / JevPilot-style decision HUD overlaid on ego camera frames."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import settings


PANEL_BG = (18, 22, 30)
PANEL_BORDER = (70, 90, 120)
TEXT_PRIMARY = (235, 240, 248)
TEXT_MUTED = (150, 165, 185)
SELECTED = (0, 255, 220)


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
    crashed: bool = False
    town: str = "Town03"
    mode: str = "offline"
    info: Dict[str, Any] = field(default_factory=dict)

    def to_inspector_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "town": self.town,
            "mode": self.mode,
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
    """Render path overlays plus decision readout on a BGR ego frame."""

    def __init__(self, title: str = "FSD Town Playback") -> None:
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
        composed = self._draw_nav_strip(composed)
        composed = self._draw_inspector(composed)
        composed = self._draw_title(composed)
        return composed

    def _draw_title(self, image: np.ndarray) -> np.ndarray:
        cv2.rectangle(image, (12, 12), (360, 44), PANEL_BG, -1)
        cv2.rectangle(image, (12, 12), (360, 44), PANEL_BORDER, 1)
        cv2.putText(
            image,
            self.title,
            (22, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )
        return image

    def _draw_nav_strip(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        top = h - 58
        cv2.rectangle(image, (0, top), (w, h), PANEL_BG, -1)
        cv2.line(image, (0, top), (w, top), PANEL_BORDER, 1)
        text = (
            f"ep {self.frame.episode}  step {self.frame.step}  "
            f"{self.frame.speed_kmh:5.1f} km/h  action {self.frame.action_label}  "
            f"mode {self.frame.mode}  town {self.frame.town}"
        )
        cv2.putText(
            image,
            text,
            (16, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )
        return image

    def _draw_inspector(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        panel_w = 300
        panel_h = min(h - 70, 360)
        x0 = w - panel_w - 12
        y0 = 56
        cv2.rectangle(image, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
        cv2.rectangle(image, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BORDER, 1)
        cv2.putText(
            image,
            "Policy signals",
            (x0 + 12, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        bar_y = y0 + 42
        for action, prob in enumerate(self.frame.probabilities):
            label = settings.ACTION_LABELS.get(action, str(action))
            q_val = float(self.frame.q_values[action]) if action < len(self.frame.q_values) else 0.0
            color = SELECTED if action == self.frame.action else settings.ACTION_COLORS_BGR.get(
                action, (160, 160, 160)
            )
            cv2.putText(
                image,
                f"{label[:8]:8s} q={q_val:+.2f}",
                (x0 + 12, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                TEXT_MUTED if action != self.frame.action else TEXT_PRIMARY,
                1,
                cv2.LINE_AA,
            )
            bar_x = x0 + 130
            bar_width = int(140 * float(prob))
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + 140, bar_y + 12), (40, 48, 60), -1)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 12), color, -1)
            bar_y += 28

        json_text = json.dumps(self.frame.to_inspector_dict(), indent=2)
        lines = json_text.splitlines()[:8]
        ty = bar_y + 10
        for line in lines:
            cv2.putText(
                image,
                line[:42],
                (x0 + 10, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                TEXT_MUTED,
                1,
                cv2.LINE_AA,
            )
            ty += 14
        return image
