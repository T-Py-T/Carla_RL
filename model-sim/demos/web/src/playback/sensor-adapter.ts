import * as THREE from "three";
import { EGO_HALF_LENGTH, NPC_HALF_LENGTH } from "./world";

export interface LidarPoint {
  x: number;
  y: number;
  z: number;
  r: number;
  g: number;
  b: number;
}

export interface TrackDetection {
  trackId: number;
  kind: "vehicle" | "pedestrian";
  distance: number;
  color: number;
  object: THREE.Object3D;
}

export interface ProximityZone {
  radius: number;
  /** 0 = clear, 1 = imminent threat */
  threat: number;
}

/** Fused observations for the ego feedback controller (sense stage). */
export interface ControlObservations {
  /** Bumper-to-forward-obstacle gap estimated from tracks + LiDAR (m). */
  forwardGapM: number;
  /** Inner proximity ring threat 0..1. */
  innerThreat: number;
  /** Minimum forward LiDAR return in ego lane wedge (m). */
  forwardLidarMinM: number;
  /** Nearest in-lane vehicle track bumper gap (m). */
  trackBumperGapM: number;
  /** World Z of in-lane lead from track fusion (bumper clamp). */
  leadWorldZ: number | null;
  /** Small lateral bias from asymmetric proximity (−1..1, lane-centering). */
  lateralBias: number;
}

export interface SensorFrame {
  lidarPoints: LidarPoint[];
  tracks: TrackDetection[];
  proximityZones: ProximityZone[];
  closestThreatM: number;
  scannerPhase: number;
  pointCount: number;
  source: "synthetic-adapter";
  controlObs: ControlObservations;
}

/** Distinct per-track colors (vehicles + pedestrians). */
const TRACK_PALETTE = [
  0x00c2ff, 0xff6b6b, 0xffcc33, 0x9b59b6, 0x2ecc71,
  0xe67e22, 0x1abc9c, 0xf39c12, 0xe74c3c, 0x3498db,
  0xff85c0, 0x7bed9f,
];

/** Must match PerceptionViz display rings — three bands, no stacked z-fight. */
export const PROXIMITY_RADII = [12, 24, 38];
const LIDAR_BEAMS = 72;
const LIDAR_MAX_RANGE = 48;
const LIDAR_FOV = 1.15; // radians (~66° half-FOV each side)
const _bbox = new THREE.Box3();
const _center = new THREE.Vector3();
const _size = new THREE.Vector3();
const _local = new THREE.Vector3();
const _world = new THREE.Vector3();

/**
 * Synthetic sensor adapter — derives structured LiDAR, proximity zones, and
 * persistent track detections from traffic actor poses (playback path).
 * No CARLA sensor required; deterministic angular sweep (not random snow).
 */
export class SensorAdapter {
  private _trackIds = new Map<string, number>();
  private _nextTrackId = 1;
  private _phase = 0;

  private _trackId(uuid: string): number {
    if (!this._trackIds.has(uuid)) {
      this._trackIds.set(uuid, this._nextTrackId++);
    }
    return this._trackIds.get(uuid)!;
  }

  trackColor(trackId: number): number {
    return TRACK_PALETTE[(trackId - 1) % TRACK_PALETTE.length];
  }

  /** Ego-local frame: +X right, +Y up, -Z forward. */
  private _toEgoLocal(
    worldPos: THREE.Vector3,
    egoPos: THREE.Vector3,
    egoYaw: number,
    out: THREE.Vector3,
  ): THREE.Vector3 {
    out.copy(worldPos).sub(egoPos);
    out.applyAxisAngle(new THREE.Vector3(0, 1, 0), -egoYaw);
    return out;
  }

  private _groundReturns(
    egoYaw: number,
    out: LidarPoint[],
    startIdx: number,
  ): number {
    let idx = startIdx;
    const beams = LIDAR_BEAMS;
    for (let b = 0; b < beams; b++) {
      const angle = -LIDAR_FOV / 2 + (b / (beams - 1)) * LIDAR_FOV;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      // Forward wedge: sample range bins along each beam
      const bins = 6;
      for (let r = 0; r < bins; r++) {
        const range = 4 + (r / (bins - 1)) * (LIDAR_MAX_RANGE - 4);
        const lx = sinA * range;
        const lz = -cosA * range;
        const ly = 0.35 + (range / LIDAR_MAX_RANGE) * 0.45;
        const intensity = 0.35 + (1 - range / LIDAR_MAX_RANGE) * 0.55;
        out[idx++] = {
          x: lx,
          y: ly,
          z: lz,
          r: 0.12 * intensity,
          g: 0.72 * intensity,
          b: 0.95 * intensity,
        };
      }
    }
    return idx;
  }

  private _actorReturns(
    actor: THREE.Object3D,
    egoPos: THREE.Vector3,
    egoYaw: number,
    out: LidarPoint[],
    startIdx: number,
    maxIdx: number,
  ): number {
    let idx = startIdx;
    _bbox.setFromObject(actor);
    if (_bbox.isEmpty()) return idx;
    _bbox.getCenter(_center);
    _bbox.getSize(_size);
    if (
      !Number.isFinite(_center.x) ||
      !Number.isFinite(_center.y) ||
      !Number.isFinite(_size.x)
    ) {
      return idx;
    }

    const dist = _center.distanceTo(egoPos);
    if (dist > LIDAR_MAX_RANGE + 5) return idx;

    const local = this._toEgoLocal(_center, egoPos, egoYaw, _local);
    if (local.z > -1.5) return idx; // behind or beside ego

    const samples = actor.userData.kind === "vehicle" ? 24 : 10;
    const halfW = Math.max(_size.x, 0.8) / 2;
    const halfH = Math.max(_size.y, 1.2) / 2;
    const halfD = Math.max(_size.z, 1.6) / 2;

    for (let s = 0; s < samples && idx < maxIdx; s++) {
      const u = (s % 4) / 3;
      const v = Math.floor(s / 4) / (Math.ceil(samples / 4) - 1 || 1);
      _world.set(
        _center.x + (u - 0.5) * halfW * 2,
        _center.y - halfH + v * halfH * 2,
        _center.z + ((s * 7) % 5 - 2) * halfD * 0.4,
      );
      const lp = this._toEgoLocal(_world, egoPos, egoYaw, _local);
      const intensity = 0.5 + (1 - dist / LIDAR_MAX_RANGE) * 0.4;
      out[idx++] = {
        x: lp.x,
        y: lp.y,
        z: lp.z,
        r: 0.18 * intensity,
        g: 0.88 * intensity,
        b: 0.98 * intensity,
      };
    }
    return idx;
  }

  private _laneOverlap(ax: number, bx: number): boolean {
    return Math.abs(ax - bx) < 1.35;
  }

  /** Actor-surface returns only — ground bins sit at ~4 m and must not drive braking. */
  private _forwardLidarMin(lidarPoints: LidarPoint[]): number {
    let minRange = Infinity;
    for (const p of lidarPoints) {
      if (p.y < -50) continue;
      if (p.z > -1.5) continue;
      if (Math.abs(p.x) > 2.4) continue;
      // Ground returns: g≈0.72·i, b≈0.95·i. Actor hits: g≈0.88·i, b≈0.98·i.
      if (p.g < p.b * 0.9) continue;
      const range = Math.hypot(p.x, p.z);
      if (range < minRange) minRange = range;
    }
    return minRange === Infinity ? LIDAR_MAX_RANGE : minRange;
  }

  private _trackObservations(
    ego: THREE.Object3D,
    tracks: TrackDetection[],
  ): { bumperGapM: number; leadWorldZ: number | null; lateralBias: number } {
    const egoPos = ego.position;
    let minBumper = Infinity;
    let leadWorldZ: number | null = null;
    let leftThreat = 0;
    let rightThreat = 0;

    for (const track of tracks) {
      if (track.kind !== "vehicle") continue;
      const obj = track.object;
      const centerGap = egoPos.z - obj.position.z;
      if (centerGap > 120 || centerGap < -8) continue;
      if (!this._laneOverlap(obj.position.x, egoPos.x)) {
        const side = obj.position.x - egoPos.x;
        const threat = Math.max(0, 1 - track.distance / PROXIMITY_RADII[0]);
        if (side < 0) leftThreat = Math.max(leftThreat, threat);
        else rightThreat = Math.max(rightThreat, threat);
        continue;
      }
      const bumper = centerGap - EGO_HALF_LENGTH - NPC_HALF_LENGTH;
      if (bumper < minBumper) {
        minBumper = bumper;
        leadWorldZ = obj.position.z;
      }
    }

    const lateralBias = Math.max(-1, Math.min(1, (leftThreat - rightThreat) * 0.35));

    return {
      bumperGapM: minBumper === Infinity ? Infinity : minBumper,
      leadWorldZ,
      lateralBias,
    };
  }

  private _fuseControlObs(
    ego: THREE.Object3D,
    frame: Omit<SensorFrame, "controlObs">,
  ): ControlObservations {
    const forwardLidarMinM = this._forwardLidarMin(frame.lidarPoints);
    const trackObs = this._trackObservations(ego, frame.tracks);
    const innerThreat = frame.proximityZones[0]?.threat ?? 0;

    const lidarBumperEst = forwardLidarMinM - NPC_HALF_LENGTH - 0.6;
    let forwardGapM = trackObs.bumperGapM;
    // Tracks are authoritative for in-lane lead; LiDAR fills gaps when no track lock.
    if (!Number.isFinite(forwardGapM)) {
      forwardGapM = Number.isFinite(lidarBumperEst)
        ? lidarBumperEst
        : Math.min(forwardLidarMinM, frame.closestThreatM) - EGO_HALF_LENGTH;
    } else if (Number.isFinite(lidarBumperEst) && lidarBumperEst < forwardGapM - 6) {
      // Tighten only when LiDAR sees a nearer surface than the fused track (≥6 m margin).
      forwardGapM = lidarBumperEst;
    }
    if (!Number.isFinite(forwardGapM) || forwardGapM > 200) {
      forwardGapM = 200;
    }

    return {
      forwardGapM,
      innerThreat,
      forwardLidarMinM,
      trackBumperGapM: trackObs.bumperGapM,
      leadWorldZ: trackObs.leadWorldZ,
      lateralBias: trackObs.lateralBias,
    };
  }

  private _proximityZones(
    ego: THREE.Object3D,
    actors: THREE.Object3D[],
  ): { zones: ProximityZone[]; closest: number } {
    const egoPos = ego.position;
    let closest = Infinity;

    for (const actor of actors) {
      if (actor === ego) continue;
      const d = actor.position.distanceTo(egoPos);
      if (d < closest) closest = d;
    }

    const zones: ProximityZone[] = PROXIMITY_RADII.map((radius) => {
      let threat = 0;
      if (closest <= radius) {
        threat = Math.max(0, 1 - (closest - 4) / (radius - 4));
      }
      return { radius, threat };
    });

    return { zones, closest: closest === Infinity ? LIDAR_MAX_RANGE : closest };
  }

  synthesize(ego: THREE.Object3D, actors: THREE.Object3D[]): SensorFrame {
    this._phase += 0.05;
    const egoPos = ego.position;
    const egoYaw = ego.rotation.y;
    const maxPoints = 720;
    const lidarPoints: LidarPoint[] = new Array(maxPoints);

    let idx = this._groundReturns(egoYaw, lidarPoints, 0);

    const tracked = actors.filter((a) => a !== ego);
    for (const actor of tracked) {
      if (idx >= maxPoints) break;
      idx = this._actorReturns(actor, egoPos, egoYaw, lidarPoints, idx, maxPoints);
    }

    // Pad unused slots off-screen
    while (idx < maxPoints) {
      lidarPoints[idx++] = { x: 0, y: -100, z: 0, r: 0, g: 0, b: 0 };
    }

    const tracks: TrackDetection[] = [];
    for (const actor of tracked) {
      const dist = actor.position.distanceTo(egoPos);
      if (dist > 50) continue;
      const trackId = this._trackId(actor.uuid);
      tracks.push({
        trackId,
        kind: (actor.userData.kind as "vehicle" | "pedestrian") ?? "vehicle",
        distance: dist,
        color: this.trackColor(trackId),
        object: actor,
      });
    }

    const { zones, closest } = this._proximityZones(ego, tracked);

    const partial = {
      lidarPoints,
      tracks,
      proximityZones: zones,
      closestThreatM: closest,
      scannerPhase: this._phase,
      pointCount: maxPoints,
      source: "synthetic-adapter" as const,
    };

    return {
      ...partial,
      controlObs: this._fuseControlObs(ego, partial),
    };
  }

  /** Expose last frame stats for JSON HUD. */
  frameStats(frame: SensorFrame) {
    const obs = frame.controlObs;
    return {
      tracks: frame.tracks.map((t) => ({
        id: t.trackId,
        kind: t.kind,
        distance_m: +t.distance.toFixed(1),
        color: `#${t.color.toString(16).padStart(6, "0")}`,
      })),
      lidar_points: frame.pointCount,
      closest_threat_m: +frame.closestThreatM.toFixed(1),
      proximity_zones: frame.proximityZones.map((z) => ({
        radius_m: z.radius,
        threat: +z.threat.toFixed(2),
      })),
      control: {
        forward_gap_m: +obs.forwardGapM.toFixed(2),
        forward_lidar_min_m: +obs.forwardLidarMinM.toFixed(2),
        track_bumper_gap_m: Number.isFinite(obs.trackBumperGapM)
          ? +obs.trackBumperGapM.toFixed(2)
          : null,
        inner_threat: +obs.innerThreat.toFixed(2),
        lateral_bias: +obs.lateralBias.toFixed(3),
      },
      scanner: frame.source,
    };
  }
}
