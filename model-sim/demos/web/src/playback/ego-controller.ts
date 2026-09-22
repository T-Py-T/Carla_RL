import * as THREE from "three";
import type { ControlObservations, SensorFrame } from "./sensor-adapter";
import { BUMPER_CLEAR_M, EGO_HALF_LENGTH, NPC_HALF_LENGTH } from "./world";

const LABELS = ["FORWARD", "LEFT", "RIGHT", "BRAKE"];

export interface ControlDecision {
  action: number;
  accel: number;
  steer: number;
  q: number[];
  probs: number[];
  forwardGapM: number;
  source: "sensor-feedback";
}

export interface ActuationResult {
  speedKmh: number;
  heading: number;
  positionZ: number;
  delta: number;
  action: number;
  forwardGapM: number;
}

/**
 * Sense → decide → actuate ego longitudinal control from fused sensor observations.
 * Rule-based feedback controller — NOT a loaded RL checkpoint.
 */
export class EgoController {
  /** Fuse LiDAR, proximity, and track detections into control observations. */
  static observe(frame: SensorFrame, ego: THREE.Object3D): ControlObservations {
    return frame.controlObs;
  }

  /**
   * Longitudinal (+ minimal lane-center lateral) decision from sensor observations.
   */
  static decide(obs: ControlObservations, speedKmh: number): ControlDecision {
    const gap = obs.forwardGapM;
    let action = 0;
    let accel = 1.2;

    if (gap < 28) accel = Math.min(accel, -1.8);
    if (gap < 20) {
      accel = Math.min(accel, -2.8);
      action = 3;
    }
    if (gap < 14) {
      accel = -5.0;
      action = 3;
    }
    if (gap < 8) {
      accel = -8;
      action = 3;
    }
    if (obs.innerThreat > 0.65 && gap < 32) {
      accel = Math.min(accel, -2.0);
      action = 3;
    }

    const steer = action === 3 ? 0 : obs.lateralBias * 0.018;

    const q = [
      gap > 24 ? 1.4 : 0.2,
      0.35 + Math.max(obs.lateralBias, 0) * 0.5,
      0.35 + Math.max(-obs.lateralBias, 0) * 0.5,
      action === 3 ? 2.1 : gap < 30 ? 0.8 : -0.3,
    ];
    const exp = q.map((v) => Math.exp(v / 1.02));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map((v) => v / sum);

    return { action, accel, steer, q, probs, forwardGapM: gap, source: "sensor-feedback" };
  }

  /**
   * Integrate speed, heading, and forward pose with optional bumper clamp from track lead.
   */
  static actuate(
    ego: THREE.Object3D,
    decision: ControlDecision,
    obs: ControlObservations,
    speedKmh: number,
    heading: number,
    captureMode: boolean,
  ): ActuationResult {
    let nextSpeed = speedKmh;
    let action = decision.action;

    if (obs.forwardGapM < 20) {
      nextSpeed = Math.min(
        nextSpeed,
        Math.max(14, (obs.forwardGapM - BUMPER_CLEAR_M) * 2.4),
      );
    }
    if (obs.forwardGapM < 14) {
      nextSpeed = Math.min(
        nextSpeed,
        Math.max(6, (obs.forwardGapM - BUMPER_CLEAR_M) * 2.0),
      );
    }
    if (obs.forwardGapM < 8) {
      nextSpeed = Math.min(
        nextSpeed,
        Math.max(0, (obs.forwardGapM - BUMPER_CLEAR_M) * 3.2),
      );
    }

    nextSpeed = Math.max(0, Math.min(65, nextSpeed + decision.accel * 0.085));

    let nextHeading = heading;
    if (captureMode || action === 3) {
      nextHeading = 0;
    } else {
      nextHeading += decision.steer;
    }

    let delta = nextSpeed / 3.6 / 10;
    let positionZ = ego.position.z;

    if (obs.leadWorldZ !== null) {
      const stopZ = obs.leadWorldZ + NPC_HALF_LENGTH + EGO_HALF_LENGTH + BUMPER_CLEAR_M;
      const nextZ = positionZ - delta;
      if (nextZ < stopZ) {
        delta = Math.max(0, positionZ - stopZ);
        positionZ = stopZ;
        nextSpeed = 0;
        action = 3;
      } else {
        positionZ = nextZ;
      }
    } else {
      positionZ -= delta;
    }

    return {
      speedKmh: nextSpeed,
      heading: nextHeading,
      positionZ,
      delta,
      action,
      forwardGapM: obs.leadWorldZ !== null
        ? positionZ - obs.leadWorldZ - EGO_HALF_LENGTH - NPC_HALF_LENGTH
        : obs.forwardGapM,
    };
  }

  static label(action: number): string {
    return LABELS[action] ?? "FORWARD";
  }
}
