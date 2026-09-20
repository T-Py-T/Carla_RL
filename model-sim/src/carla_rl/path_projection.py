"""Project candidate driving paths onto an ego RGB camera frame."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from . import settings


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


def draw_candidate_paths(
    frame_bgr: np.ndarray,
    selected_action: int,
    probabilities: Iterable[float],
    alpha_selected: float = 0.95,
    alpha_other: float = 0.55,
) -> np.ndarray:
    """Overlay colored candidate paths on a BGR camera frame."""
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    probs = list(probabilities)

    for action in range(len(probs)):
        points = project_vehicle_points(
            candidate_path_points(action), width=width, height=height
        )
        if len(points) < 2:
            continue
        color = settings.ACTION_COLORS_BGR.get(action, (200, 200, 200))
        thickness = 4 if action == selected_action else 2
        alpha = alpha_selected if action == selected_action else alpha_other
        overlay = output.copy()
        for start, end in zip(points, points[1:]):
            cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)

    return output


def draw_lane_centerline(frame_bgr: np.ndarray) -> np.ndarray:
    """Draw a subtle centerline to anchor the projected path in offline mode."""
    height, width = frame_bgr.shape[:2]
    center = project_vehicle_points(
        [(6, 0, -1.55), (12, 0, -1.55), (18, 0, -1.55), (24, 0, -1.55)],
        width=width,
        height=height,
    )
    overlay = frame_bgr.copy()
    for start, end in zip(center, center[1:]):
        cv2.line(overlay, start, end, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0, frame_bgr)
    return frame_bgr
