"""CARLA connection and demo defaults for FSD playback."""

from __future__ import annotations

import os

from .jevpilot_theme import (
    PATH_AMBER,
    PATH_COLLISION,
    PATH_CYAN,
    PATH_FORWARD,
    PATH_SELECTED,
)

# Connection
CARLA_HOST = os.environ.get("CARLA_HOST", "localhost")
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
CARLA_TIMEOUT_SEC = float(os.environ.get("CARLA_TIMEOUT_SEC", "10.0"))

# Ego RGB camera (matches historical CarlaEnv defaults)
IMG_WIDTH = int(os.environ.get("CARLA_IMG_WIDTH", "960"))
IMG_HEIGHT = int(os.environ.get("CARLA_IMG_HEIGHT", "540"))
CAMERA_FOV = float(os.environ.get("CARLA_CAMERA_FOV", "90"))
CAMERA_OFFSET_X = float(os.environ.get("CARLA_CAMERA_X", "1.6"))
CAMERA_OFFSET_Z = float(os.environ.get("CARLA_CAMERA_Z", "1.4"))

# Town / traffic
DEFAULT_TOWN = os.environ.get("CARLA_TOWN", "Town03")
SYNC_MODE = os.environ.get("CARLA_SYNC_MODE", "1") == "1"
FIXED_DELTA_SECONDS = float(os.environ.get("CARLA_FIXED_DELTA", "0.05"))
NPC_COUNT = int(os.environ.get("CARLA_NPC_COUNT", "25"))

# Discrete driving actions used by the local policy + HUD
ACTION_LABELS = {
    0: "FORWARD",
    1: "LEFT",
    2: "RIGHT",
    3: "BRAKE",
}

MANEUVER_LABELS = {
    0: "Continue straight",
    1: "Bear left",
    2: "Bear right",
    3: "Slow for hazard",
}

# JevPilot candidate path colors (BGR)
ACTION_COLORS_BGR = {
    0: PATH_FORWARD,
    1: PATH_AMBER,
    2: PATH_AMBER,
    3: PATH_COLLISION,
}

SELECTED_PATH_COLOR_BGR = PATH_SELECTED
SELECTED_PATH_GLOW_BGR = PATH_CYAN

THROTTLE = 0.45
STEER = 0.35
BRAKE = 0.8
