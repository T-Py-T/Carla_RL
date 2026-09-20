"""Synthetic 3D town ego-camera renderer for proof capture without CARLA."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import settings


@dataclass
class TownState:
    distance_m: float = 0.0
    heading: float = 0.0
    lane_offset: float = 0.0
    speed_kmh: float = 24.0


class OfflineTownRenderer:
    """CPU-only perspective town view for FSD-style playback demos."""

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
        self._draw_buildings(frame)
        self._draw_road(frame)
        self._draw_lane_markings(frame)
        self._draw_horizon_glow(frame)
        self._draw_hood(frame)
        return frame

    def _draw_sky(self, frame: np.ndarray) -> None:
        for y in range(self.height):
            t = y / max(self.height * 0.55, 1)
            color = (
                min(int(28 + 40 * t), 255),
                min(int(48 + 70 * t), 255),
                min(int(88 + 120 * t), 255),
            )
            frame[y, :] = color

    def _draw_buildings(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        rng = np.random.default_rng(int(self.state.distance_m) // 4)
        for index in range(14):
            depth = 0.15 + (index / 14.0) * 0.85
            base_x = int((index / 14.0) * self.width + np.sin(self._phase + index) * 18)
            side = -1 if index % 2 == 0 else 1
            width = int(80 + 160 * depth)
            height = int(horizon - 40 - depth * (120 + rng.integers(0, 80)))
            x = base_x if side < 0 else self.width - base_x - width
            x = int(np.clip(x, -width // 2, self.width - width // 2))
            shade = int(35 + 55 * depth)
            cv2.rectangle(
                frame,
                (x, height),
                (x + width, horizon + int(30 * depth)),
                (shade, shade + 8, shade + 18),
                -1,
            )
            # Window grid
            for row in range(4):
                for col in range(3):
                    wx = x + 12 + col * (width // 4)
                    wy = height + 18 + row * 22
                    if wy < horizon - 8:
                        cv2.rectangle(frame, (wx, wy), (wx + 14, wy + 12), (190, 210, 230), 1)

    def _draw_road(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        bottom_w = int(self.width * 0.95)
        top_w = int(self.width * 0.18)
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
        cv2.fillPoly(frame, [pts], (48, 48, 52))

    def _draw_lane_markings(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        scroll = (self.state.distance_m * 40) % 60
        for lane in (-0.22, 0.0, 0.22):
            for depth in np.linspace(0.05, 1.0, 18):
                y = int(horizon + (self.height - horizon) * depth)
                x_center = self.width / 2 + lane * self.width * (0.15 + depth * 0.75)
                x_center += np.sin(self._phase * 0.5 + depth * 3) * 4
                dash = int((depth * 60 + scroll) % 30)
                if dash < 18:
                    w = int(8 + depth * 18)
                    cv2.line(
                        frame,
                        (int(x_center - w / 2), y),
                        (int(x_center + w / 2), y),
                        (220, 220, 220),
                        2,
                        cv2.LINE_AA,
                    )

    def _draw_horizon_glow(self, frame: np.ndarray) -> None:
        horizon = int(self.height * 0.52)
        overlay = frame.copy()
        cv2.line(overlay, (0, horizon), (self.width, horizon), (120, 170, 220), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    def _draw_hood(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        pts = np.array([[0, h], [w, h], [w, int(h * 0.82)], [0, int(h * 0.86)]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (12, 12, 14))
        cv2.line(frame, (0, int(h * 0.86)), (w, int(h * 0.82)), (70, 70, 75), 2, cv2.LINE_AA)
