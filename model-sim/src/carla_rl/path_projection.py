"""Project candidate driving paths onto an ego RGB camera frame."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from . import settings
from .jevpilot_theme import PATH_FORWARD


Point2D = Tuple[int, int]


def camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_vehicle_points(
    points_vehicle: Sequence[Tuple[float, float, float]],
    width: int,
    height: int,
    fov_deg: float = settings.CAMERA_FOV,
) -> List[Point2D]:
    """Project 3D points in the ego frame (x forward, y right, z up) to pixels."""
    k = camera_intrinsics(width, height, fov_deg)
    projected: List[Point2D] = []
    for x, y, z in points_vehicle:
        if x <= 0.5:
            continue
        cam = np.array([y, -z, x], dtype=np.float64)
        uvw = k @ cam
        if uvw[2] <= 0:
            continue
        u = int(round(uvw[0] / uvw[2]))
        v = int(round(uvw[1] / uvw[2]))
        if 0 <= u < width and 0 <= v < height:
            projected.append((u, v))
    return projected


def candidate_path_points(
    action: int,
    length_m: float = 28.0,
    step_m: float = 1.5,
) -> List[Tuple[float, float, float]]:
    """Generate a smooth arc in vehicle coordinates for one discrete action."""
    steer = {
        0: 0.0,
        1: -0.55,
        2: 0.55,
        3: 0.0,
    }.get(action, 0.0)

    points: List[Tuple[float, float, float]] = []
    heading = 0.0
    x = 2.0
    y = 0.0
    for _ in np.arange(0.0, length_m, step_m):
        heading += steer * 0.08
        x += step_m * np.cos(heading)
        y += step_m * np.sin(heading)
        points.append((x, y, -1.55))
    return points


def _draw_path_polyline(
    canvas: np.ndarray,
    points: List[Point2D],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    for start, end in zip(points, points[1:]):
        cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)


def draw_candidate_paths(
    frame_bgr: np.ndarray,
    selected_action: int,
    probabilities: Iterable[float],
    alpha_selected: float = 0.92,
    alpha_other: float = 0.62,
) -> np.ndarray:
    """Overlay JevPilot-style candidate paths (blue selected ribbon + amber/cyan alts)."""
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    probs = list(probabilities)

    # Draw non-selected candidates first.
    for action in range(len(probs)):
        if action == selected_action:
            continue
        points = project_vehicle_points(
            candidate_path_points(action), width=width, height=height
        )
        if len(points) < 2:
            continue
        color = settings.ACTION_COLORS_BGR.get(action, PATH_FORWARD)
        overlay = output.copy()
        _draw_path_polyline(overlay, points, color, thickness=2)
        cv2.addWeighted(overlay, alpha_other, output, 1.0 - alpha_other, 0, output)

    # Selected path: glow ribbon then bright blue centerline (JevPilot road-vectors.js).
    selected_points = project_vehicle_points(
        candidate_path_points(selected_action), width=width, height=height
    )
    if len(selected_points) >= 2:
        glow = output.copy()
        _draw_path_polyline(glow, selected_points, settings.SELECTED_PATH_GLOW_BGR, thickness=8)
        cv2.addWeighted(glow, 0.18, output, 0.82, 0, output)
        ribbon = output.copy()
        _draw_path_polyline(ribbon, selected_points, settings.SELECTED_PATH_COLOR_BGR, thickness=5)
        cv2.addWeighted(ribbon, alpha_selected, output, 1.0 - alpha_selected, 0, output)

    return output


def draw_lane_centerline(frame_bgr: np.ndarray) -> np.ndarray:
    """Subtle road anchor — JevPilot keeps the route visible under ribbons."""
    height, width = frame_bgr.shape[:2]
    center = project_vehicle_points(
        [(6, 0, -1.55), (12, 0, -1.55), (18, 0, -1.55), (24, 0, -1.55)],
        width=width,
        height=height,
    )
    overlay = frame_bgr.copy()
    for start, end in zip(center, center[1:]):
        cv2.line(overlay, start, end, (201, 196, 193), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, frame_bgr, 0.78, 0, frame_bgr)
    return frame_bgr
