"""JevPilot visual tokens extracted from standardagents/jevpilot (look-only)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def hex_bgr(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


# JevPilot road-vectors.js candidate palette
PATH_FORWARD = hex_bgr("#48a5ff")
PATH_SELECTED = hex_bgr("#007aff")
PATH_CYAN = hex_bgr("#38bcd6")
PATH_AMBER = hex_bgr("#e6a34b")
PATH_COLLISION = hex_bgr("#e86940")
PATH_REVERSE = hex_bgr("#9a6bff")

# Scene.js environment
SKY_TOP = hex_bgr("#b7c9db")
SKY_HORIZON = hex_bgr("#d8e4ef")
ROAD_SURFACE = hex_bgr("#70817c")
ROAD_SHOULDER = hex_bgr("#d8d6c9")
BUILDING_FACADE = hex_bgr("#8d9cab")

# style.css typography / chrome
TEXT_PRIMARY = hex_bgr("#171a20")
TEXT_MUTED = hex_bgr("#6b7077")
TEXT_FAINT = hex_bgr("#a3a6ab")
ACCENT_RED = hex_bgr("#e82127")
PANEL_LINE = hex_bgr("#e3e5e8")
GLASS_FILL = (237, 237, 237)
GLASS_BORDER = (228, 230, 232)

ACTION_PATH_COLORS = {
    0: PATH_FORWARD,
    1: PATH_AMBER,
    2: PATH_AMBER,
    3: PATH_COLLISION,
}


def blend_glass(image: np.ndarray, rect: Tuple[int, int, int, int], alpha: float = 0.88) -> None:
    """Draw a JevPilot `.glass` panel (white, blurred feel via high-alpha fill)."""
    x0, y0, x1, y1 = rect
    roi = image[y0:y1, x0:x1]
    overlay = np.full_like(roi, GLASS_FILL)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    cv2.rectangle(image, (x0, y0), (x1, y1), GLASS_BORDER, 1, cv2.LINE_AA)


def put_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.45,
    color: Tuple[int, int, int] = TEXT_PRIMARY,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
