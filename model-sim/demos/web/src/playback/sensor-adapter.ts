import * as THREE from "three";

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

export interface SensorFrame {
  lidarPoints: LidarPoint[];
  tracks: TrackDetection[];
  proximityZones: ProximityZone[];
  closestThreatM: number;
  scannerPhase: number;
  pointCount: number;
  source: "synthetic-adapter";
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

    return {
      lidarPoints,
      tracks,
      proximityZones: zones,
      closestThreatM: closest,
      scannerPhase: this._phase,
      pointCount: maxPoints,
      source: "synthetic-adapter",
    };
  }

  /** Expose last frame stats for JSON HUD. */
  frameStats(frame: SensorFrame) {
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
      scanner: frame.source,
    };
  }
}
