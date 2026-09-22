import * as THREE from "three";
import type { SensorFrame } from "./sensor-adapter";

const PROXIMITY_COLORS = { clear: 0x2ecc71, warn: 0xf1c40f, threat: 0xe74c3c };
/** Display radii (m) — three bands to avoid stacked-ring z-fight. */
const PROXIMITY_DISPLAY = [12, 24, 38];

export class PerceptionViz {
  root: THREE.Group;
  boxGroup: THREE.Group;
  points: THREE.Points;
  proximityRings: THREE.Mesh[];
  scanWedge: THREE.Mesh;
  scanRing: THREE.Mesh;
  threatArc: THREE.Mesh;
  private boxes = new Map<string, THREE.BoxHelper>();
  private _pointCount = 720;
  private _threatZ = 12;

  constructor(scene: THREE.Scene) {
    this.root = new THREE.Group();
    this.boxGroup = new THREE.Group();
    scene.add(this.root);
    scene.add(this.boxGroup);

    const positions = new Float32Array(this._pointCount * 3);
    const colors = new Float32Array(this._pointCount * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 0.038,
        vertexColors: true,
        transparent: true,
        opacity: 0.52,
        depthWrite: false,
        depthTest: true,
        sizeAttenuation: true,
      }),
    );
    this.points.renderOrder = 3;
    this.root.add(this.points);

    this.scanRing = new THREE.Mesh(
      new THREE.RingGeometry(1.4, 1.65, 48),
      new THREE.MeshBasicMaterial({
        color: 0x38bcd6,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: false,
      }),
    );
    this.scanRing.rotation.x = -Math.PI / 2;
    this.scanRing.position.y = 0.16;
    this.root.add(this.scanRing);

    this.proximityRings = PROXIMITY_DISPLAY.map((radius, i) => {
      const band = 0.35;
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(radius - band, radius, 64),
        new THREE.MeshBasicMaterial({
          color: 0x2ecc71,
          transparent: true,
          opacity: 0.1,
          side: THREE.DoubleSide,
          depthWrite: false,
          depthTest: false,
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.14 + i * 0.008;
      this.root.add(ring);
      return ring;
    });

    this.scanWedge = new THREE.Mesh(
      new THREE.CircleGeometry(14, 32, -0.5, 1.0),
      new THREE.MeshBasicMaterial({
        color: 0x007aff,
        transparent: true,
        opacity: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: false,
      }),
    );
    this.scanWedge.rotation.x = -Math.PI / 2;
    this.scanWedge.rotation.z = Math.PI / 2;
    this.scanWedge.position.set(0, 0.13, -1.2);
    this.root.add(this.scanWedge);

    this.threatArc = new THREE.Mesh(
      new THREE.RingGeometry(0.8, 1.6, 24, 1, -0.3, 0.6),
      new THREE.MeshBasicMaterial({
        color: 0xe74c3c,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: false,
      }),
    );
    this.threatArc.rotation.x = -Math.PI / 2;
    this.threatArc.position.set(0, 0.15, -8);
    this.root.add(this.threatArc);
  }

  setVisible(visible: boolean) {
    this.root.visible = visible;
    this.boxGroup.visible = visible;
    for (const helper of this.boxes.values()) {
      helper.visible = visible;
    }
  }

  private _applyLidarPoints(frame: SensorFrame) {
    const pos = this.points.geometry.attributes.position as THREE.BufferAttribute;
    const col = this.points.geometry.attributes.color as THREE.BufferAttribute;
    const pts = frame.lidarPoints;
    for (let i = 0; i < this._pointCount; i++) {
      const p = pts[i];
      pos.setXYZ(i, p.x, Math.max(p.y, 0.18), p.z);
      col.setXYZ(i, p.r, p.g, p.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
  }

  private _updateProximityRings(frame: SensorFrame) {
    const zones = frame.proximityZones;
    this.proximityRings.forEach((ring, i) => {
      const targetRadius = PROXIMITY_DISPLAY[i];
      const zone = zones.find((z) => Math.abs(z.radius - targetRadius) < 8) ?? zones[i];
      if (!zone) return;
      const mat = ring.material as THREE.MeshBasicMaterial;
      const t = zone.threat;
      const color = new THREE.Color();
      if (t < 0.35) {
        color.setHex(PROXIMITY_COLORS.clear);
      } else if (t < 0.7) {
        color.lerpColors(
          new THREE.Color(PROXIMITY_COLORS.clear),
          new THREE.Color(PROXIMITY_COLORS.warn),
          (t - 0.35) / 0.35,
        );
      } else {
        color.lerpColors(
          new THREE.Color(PROXIMITY_COLORS.warn),
          new THREE.Color(PROXIMITY_COLORS.threat),
          (t - 0.7) / 0.3,
        );
      }
      mat.color.copy(color);
      mat.opacity = 0.06 + t * 0.18;
    });

    const close = Number.isFinite(frame.closestThreatM) ? frame.closestThreatM : 50;
    const targetZ = -Math.min(Math.max(close, 6), 36);
    this._threatZ += (targetZ - this._threatZ) * 0.12;
    this.threatArc.position.z = this._threatZ;
    const threatMat = this.threatArc.material as THREE.MeshBasicMaterial;
    threatMat.opacity = close < 18 ? 0.28 : 0.1;
  }

  update(ego: THREE.Object3D, frame: SensorFrame) {
    this.root.position.copy(ego.position);
    this.root.rotation.y = ego.rotation.y;

    this.scanRing.rotation.z = frame.scannerPhase;
    this.scanWedge.rotation.z = Math.PI / 2 + frame.scannerPhase * 0.25;

    this._applyLidarPoints(frame);
    this._updateProximityRings(frame);

    const seen = new Set<string>();
    for (const track of frame.tracks) {
      const pos = track.object.position;
      if (!Number.isFinite(pos.x) || !Number.isFinite(pos.y) || !Number.isFinite(pos.z)) continue;

      seen.add(track.object.uuid);
      const color = track.color;
      if (!this.boxes.has(track.object.uuid)) {
        const helper = new THREE.BoxHelper(track.object, color);
        const mat = helper.material as THREE.LineBasicMaterial;
        mat.transparent = true;
        mat.opacity = 0.88;
        mat.depthTest = false;
        mat.depthWrite = false;
        mat.linewidth = 2;
        helper.renderOrder = 8;
        this.boxGroup.add(helper);
        this.boxes.set(track.object.uuid, helper);
      } else {
        const helper = this.boxes.get(track.object.uuid)!;
        helper.setFromObject(track.object);
        (helper.material as THREE.LineBasicMaterial).color.setHex(color);
      }
    }

    for (const [uuid, helper] of this.boxes) {
      if (!seen.has(uuid)) {
        this.boxGroup.remove(helper);
        helper.geometry.dispose();
        this.boxes.delete(uuid);
      }
    }
  }

  dispose() {
    for (const helper of this.boxes.values()) {
      this.boxGroup.remove(helper);
      helper.geometry.dispose();
    }
    this.boxes.clear();
  }
}
