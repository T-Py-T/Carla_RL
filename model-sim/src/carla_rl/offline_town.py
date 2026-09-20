"""Synthetic 3D town ego-camera renderer for proof capture without CARLA."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import settings
from .jevpilot_theme import BUILDING_FACADE, ROAD_SHOULDER, ROAD_SURFACE, SKY_HORIZON, SKY_TOP


@dataclass
class TownState:
    distance_m: float = 0.0
    heading: float = 0.0
    lane_offset: float = 0.0
    speed_kmh: float = 24.0


class OfflineTownRenderer:
    """CPU perspective town view styled after JevPilot Three.js scenes."""

    def __init__(
        self,
        width: int = settings.IMG_WIDTH,
        height: int = settings.IMG_HEIGHT,
        town: str = settings.DEFAULT_TOWN,
    ) -> None:
        self.width = width
        self.height = height
        self.town = town
        self.state = TownState()
        self._phase = 0.0

    def reset(self) -> TownState:
        self.state = TownState()
        self._phase = 0.0
        return self.state

    def step(self, action: int) -> TownState:
        throttle = settings.THROTTLE if action != 3 else 0.0
        steer = {
            0: 0.0,
            1: -settings.STEER,
            2: settings.STEER,
            3: 0.0,
        }.get(action, 0.0)
        brake = settings.BRAKE if action == 3 else 0.0

        accel = throttle * 2.2 - brake * 4.0
        self.state.speed_kmh = float(np.clip(self.state.speed_kmh + accel, 0.0, 65.0))
        speed_mps = self.state.speed_kmh / 3.6
        self.state.heading += steer * 0.035
        self.state.lane_offset += np.sin(self.state.heading) * 0.02
        self.state.distance_m += speed_mps * 0.12
        self._phase += 0.08 + speed_mps * 0.01
        return self.state

    def render(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_sky(frame)
        self._draw_shoulder(frame)
        self._draw_buildings(frame)
        self._draw_road(frame)
        self._draw_lane_markings(frame)
        self._draw_horizon_glow(frame)
        self._draw_hood(frame)
        return frame

    def _draw_sky(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        for y in range(horizon):
            t = y / max(horizon, 1)
            color = tuple(
                int(SKY_TOP[i] + (SKY_HORIZON[i] - SKY_TOP[i]) * t) for i in range(3)
            )
            frame[y, :] = color

    def _draw_shoulder(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        bottom_w = int(self.width * 1.05)
        top_w = int(self.width * 0.34)
        left_bottom = (self.width - bottom_w) // 2
        right_bottom = left_bottom + bottom_w
        left_top = (self.width - top_w) // 2
        right_top = left_top + top_w
        pts = np.array(
            [
                [left_bottom, self.height],
                [right_bottom, self.height],
                [right_top, horizon - 6],
                [left_top, horizon - 6],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], ROAD_SHOULDER)

    def _draw_buildings(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        rng = np.random.default_rng(int(self.state.distance_m) // 4)
        palette = [
            BUILDING_FACADE,
            (155, 163, 171),
            (139, 152, 160),
            (125, 138, 148),
        ]
        for index in range(16):
            depth = 0.12 + (index / 16.0) * 0.88
            base_x = int((index / 16.0) * self.width + np.sin(self._phase + index) * 14)
            side = -1 if index % 2 == 0 else 1
            width = int(70 + 150 * depth)
            height = int(horizon - 50 - depth * (110 + rng.integers(0, 70)))
            x = base_x if side < 0 else self.width - base_x - width
            x = int(np.clip(x, -width // 2, self.width - width // 2))
            color = palette[index % len(palette)]
            cv2.rectangle(
                frame,
                (x, height),
                (x + width, horizon + int(24 * depth)),
                color,
                -1,
            )
            for row in range(5):
                for col in range(max(2, width // 28)):
                    wx = x + 10 + col * (width // max(2, width // 28))
                    wy = height + 14 + row * 18
                    if wy < horizon - 10:
                        cv2.rectangle(frame, (wx, wy), (wx + 12, wy + 10), (220, 228, 235), 1)

    def _draw_road(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        bottom_w = int(self.width * 0.88)
        top_w = int(self.width * 0.16)
        left_bottom = (self.width - bottom_w) // 2
        right_bottom = left_bottom + bottom_w
        left_top = (self.width - top_w) // 2
        right_top = left_top + top_w
        pts = np.array(
            [
                [left_bottom, self.height],
                [right_bottom, self.height],
                [right_top, horizon],
                [left_top, horizon],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], ROAD_SURFACE)

    def _draw_lane_markings(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        scroll = (self.state.distance_m * 40) % 60
        for lane in (-0.22, 0.0, 0.22):
            for depth in np.linspace(0.05, 1.0, 18):
                y = int(horizon + (self.height - horizon) * depth)
                x_center = self.width / 2 + lane * self.width * (0.15 + depth * 0.75)
                x_center += np.sin(self._phase * 0.5 + depth * 3) * 3
                dash = int((depth * 60 + scroll) % 30)
                if dash < 18:
                    w = int(8 + depth * 18)
                    cv2.line(
                        frame,
                        (int(x_center - w / 2), y),
                        (int(x_center + w / 2), y),
                        (235, 238, 242),
                        2,
                        cv2.LINE_AA,
                    )

    def _draw_horizon_glow(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        overlay = frame.copy()
        cv2.line(overlay, (0, horizon), (self.width, horizon), SKY_HORIZON, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    def _draw_hood(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        pts = np.array([[0, h], [w, h], [w, int(h * 0.84)], [0, int(h * 0.87)]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (20, 22, 24))
        cv2.line(frame, (0, int(h * 0.87)), (w, int(h * 0.84)), (70, 74, 78), 2, cv2.LINE_AA)
