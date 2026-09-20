"""Local rule-based driving policy with synthetic Q-values for FSD HUD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import settings


@dataclass
class PolicyDecision:
    action: int
    action_label: str
    q_values: np.ndarray
    probabilities: np.ndarray
    explored: bool = False
    epsilon: float = 0.0


class LocalDrivingPolicy:
    """Lightweight local policy; no external Jev/API dependencies."""

    def __init__(self, action_size: int = 4, temperature: float = 1.0) -> None:
        self.action_size = action_size
        self.temperature = max(temperature, 1e-6)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def decide(
        self,
        speed_kmh: float,
        lane_offset: float = 0.0,
        obstacle_ahead: bool = False,
    ) -> PolicyDecision:
        self._step += 1
        q = np.zeros(self.action_size, dtype=np.float32)
        q[0] = 1.2 + min(speed_kmh / 40.0, 1.0)
        q[1] = 0.35 + max(lane_offset, 0.0) * 2.0
        q[2] = 0.35 + max(-lane_offset, 0.0) * 2.0
        q[3] = 1.4 if obstacle_ahead or speed_kmh > 55 else -0.2

        # Gentle weave so candidate paths stay visible in demos.
        weave = np.sin(self._step / 18.0) * 0.25
        q[1] += max(weave, 0.0)
        q[2] += max(-weave, 0.0)

        scaled = q / self.temperature
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= np.sum(probs)
        action = int(np.argmax(q))

        return PolicyDecision(
            action=action,
            action_label=settings.ACTION_LABELS[action],
            q_values=q,
            probabilities=probs,
        )
