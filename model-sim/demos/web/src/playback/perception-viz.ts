import * as THREE from "three";
import type { SensorFrame } from "./sensor-adapter";

const PROXIMITY_COLORS = [
  { clear: 0x2ecc71, warn: 0xf1c40f, threat: 0xe74c3c },
];

export class PerceptionViz {
  root: THREE.Group;
  points: THREE.Points;
  proximityRings: THREE.Mesh[];
  scanWedge: THREE.Mesh;
  scanRing: THREE.Mesh;
  threatArc: THREE.Mesh;
  private boxes = new Map<string, THREE.BoxHelper>();
  private _pointCount = 720;

  constructor(scene: THREE.Scene) {
    this.root = new THREE.Group();
    scene.add(this.root);

    const positions = new Float32Array(this._pointCount * 3);
    const colors = new Float32Array(this._pointCount * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 0.055,
        vertexColors: true,
        transparent: true,
        opacity: 0.62,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    );
    this.points.renderOrder = 2;
    this.root.add(this.points);

    // Rotating scan ring at ego
    this.scanRing = new THREE.Mesh(
      new THREE.RingGeometry(1.6, 1.9, 64),
      new THREE.MeshBasicMaterial({
        color: 0x38bcd6,
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scanRing.rotation.x = -Math.PI / 2;
    this.scanRing.position.y = 0.11;
    this.root.add(this.scanRing);

    // Proximity range rings (10/20/30/40/50 m)
    this.proximityRings = [10, 20, 30, 40, 50].map((radius) => {
      const inner = radius - 0.15;
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(inner, radius, 64),
        new THREE.MeshBasicMaterial({
          color: 0x2ecc71,
          transparent: true,
          opacity: 0.12,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.09;
      this.root.add(ring);
      return ring;
    });

    // Forward lidar wedge (ground-plane coverage indicator)
    this.scanWedge = new THREE.Mesh(
      new THREE.CircleGeometry(16, 32, -0.55, 1.1),
      new THREE.MeshBasicMaterial({
        color: 0x007aff,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scanWedge.rotation.x = -Math.PI / 2;
    this.scanWedge.rotation.z = Math.PI / 2;
    this.scanWedge.position.set(0, 0.1, -1.5);
    this.root.add(this.scanWedge);

    // Closest-threat forward arc
    this.threatArc = new THREE.Mesh(
      new THREE.CircleGeometry(1, 24, -0.35, 0.7),
      new THREE.MeshBasicMaterial({
        color: 0xe74c3c,
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.threatArc.rotation.x = -Math.PI / 2;
    this.threatArc.position.set(0, 0.12, -2);
    this.root.add(this.threatArc);
  }

  setVisible(visible: boolean) {
    this.root.visible = visible;
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
      pos.setXYZ(i, p.x, p.y, p.z);
      col.setXYZ(i, p.r, p.g, p.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
  }

  private _updateProximityRings(frame: SensorFrame) {
    const palette = PROXIMITY_COLORS[0];
    frame.proximityZones.forEach((zone, i) => {
      const ring = this.proximityRings[i];
      if (!ring) return;
      const mat = ring.material as THREE.MeshBasicMaterial;
      const t = zone.threat;
      const color = new THREE.Color();
      if (t < 0.35) {
        color.setHex(palette.clear);
      } else if (t < 0.7) {
        color.lerpColors(new THREE.Color(palette.clear), new THREE.Color(palette.warn), (t - 0.35) / 0.35);
      } else {
        color.lerpColors(new THREE.Color(palette.warn), new THREE.Color(palette.threat), (t - 0.7) / 0.3);
      }
      mat.color.copy(color);
      mat.opacity = 0.14 + t * 0.32;
    });

    const threatMat = this.threatArc.material as THREE.MeshBasicMaterial;
    const close = Number.isFinite(frame.closestThreatM) ? frame.closestThreatM : 50;
    const scale = Math.min(1, Math.max(0.15, close / 30));
    this.threatArc.scale.set(scale * 8, scale * 8, 1);
    threatMat.opacity = close < 20 ? 0.35 : 0.12;
    this.threatArc.position.z = -Math.min(close, 40);
  }

  update(ego: THREE.Object3D, frame: SensorFrame) {
    this.root.position.copy(ego.position);
    this.root.rotation.y = ego.rotation.y;

    this.scanRing.rotation.z = frame.scannerPhase;
    this.scanWedge.rotation.z = Math.PI / 2 + frame.scannerPhase * 0.4;

    this._applyLidarPoints(frame);
    this._updateProximityRings(frame);

    const seen = new Set<string>();
    for (const track of frame.tracks) {
      seen.add(track.object.uuid);
      const color = track.color;
      if (!this.boxes.has(track.object.uuid)) {
        const helper = new THREE.BoxHelper(track.object, color);
        helper.material.transparent = true;
        helper.material.opacity = 0.92;
        helper.renderOrder = 7;
        this.root.parent!.add(helper);
        this.boxes.set(track.object.uuid, helper);
      } else {
        const helper = this.boxes.get(track.object.uuid)!;
        helper.setFromObject(track.object);
        (helper.material as THREE.LineBasicMaterial).color.setHex(color);
      }
    }

    for (const [uuid, helper] of this.boxes) {
      if (!seen.has(uuid)) {
        this.root.parent!.remove(helper);
        helper.geometry.dispose();
        this.boxes.delete(uuid);
      }
    }
  }

  dispose() {
    for (const helper of this.boxes.values()) {
      this.root.parent?.remove(helper);
      helper.geometry.dispose();
    }
    this.boxes.clear();
  }
}
