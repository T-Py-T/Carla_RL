// @ts-nocheck — legacy stylized viz; perception mesh work HARD blocked
import * as THREE from "three";

const BOX_COLORS = {
  vehicle: 0x00c2ff,
  pedestrian: 0xffcc33,
};

export class PerceptionViz {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    scene.add(this.root);
    this.boxes = new Map();
    this._pointCount = 720;

    const positions = new Float32Array(this._pointCount * 3);
    const colors = new Float32Array(this._pointCount * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 0.028,
        vertexColors: true,
        transparent: true,
        opacity: 0.32,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    );
    this.points.renderOrder = 2;
    this.root.add(this.points);

    this.scanRing = new THREE.Mesh(
      new THREE.RingGeometry(1.8, 2.05, 64),
      new THREE.MeshBasicMaterial({
        color: 0x38bcd6,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scanRing.rotation.x = -Math.PI / 2;
    this.scanRing.position.y = 0.11;
    this.root.add(this.scanRing);

    // Forward lidar wedge (not a full circle — avoids "snow in the sky")
    this.scanWedge = new THREE.Mesh(
      new THREE.CircleGeometry(16, 32, -0.55, 1.1),
      new THREE.MeshBasicMaterial({
        color: 0x007aff,
        transparent: true,
        opacity: 0.07,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scanWedge.rotation.x = -Math.PI / 2;
    this.scanWedge.rotation.z = Math.PI / 2;
    this.scanWedge.position.set(0, 0.1, -1.5);
    this.root.add(this.scanWedge);
    this._phase = 0;
  }

  setVisible(visible) {
    this.root.visible = visible;
    for (const helper of this.boxes.values()) {
      helper.visible = visible;
    }
  }

  _refreshLidarPoints(actors) {
    const pos = this.points.geometry.attributes.position;
    const col = this.points.geometry.attributes.color;
    let index = 0;

    // Ground returns in forward wedge (road scan)
    while (index < this._pointCount * 0.75) {
      const ahead = 3 + Math.random() * 34;
      const lateral = (Math.random() - 0.5) * 8.5;
      const y = 0.06 + Math.random() * 0.25;
      pos.setXYZ(index, lateral, y, -ahead);
      const intensity = 0.45 + (1 - ahead / 38) * 0.4;
      col.setXYZ(index, 0.15 * intensity, 0.75 * intensity, 0.95 * intensity);
      index += 1;
    }

    // Returns on tracked actors (detection-aligned points)
    for (const actor of actors) {
      if (index >= this._pointCount) break;
      const local = actor.position.clone().sub(this.root.position);
      local.applyAxisAngle(new THREE.Vector3(0, 1, 0), -this.root.rotation.y);
      if (local.z > -2) continue;
      const samples = actor.userData.kind === "vehicle" ? 28 : 12;
      for (let s = 0; s < samples && index < this._pointCount; s++) {
        pos.setXYZ(
          index,
          local.x + (Math.random() - 0.5) * 1.6,
          0.4 + Math.random() * 1.2,
          local.z + (Math.random() - 0.5) * 2.0,
        );
        col.setXYZ(index, 0.2, 0.85, 0.95);
        index += 1;
      }
    }

    while (index < this._pointCount) {
      pos.setXYZ(index, 0, -100, 0);
      col.setXYZ(index, 0, 0, 0);
      index += 1;
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
  }

  update(ego, actors) {
    this.root.position.copy(ego.position);
    this.root.rotation.y = ego.rotation.y;
    this._phase += 0.05;
    this.scanRing.rotation.z = this._phase;
    this.scanWedge.rotation.z = Math.PI / 2 + this._phase * 0.4;

    const tracked = actors.filter((a) => a !== ego);
    this._refreshLidarPoints(tracked);

    const seen = new Set();
    for (const actor of tracked) {
      const dist = actor.position.distanceTo(ego.position);
      if (dist > 50) continue;
      seen.add(actor.uuid);
      const color = BOX_COLORS[actor.userData.kind] ?? 0xffffff;
      if (!this.boxes.has(actor.uuid)) {
        const helper = new THREE.BoxHelper(actor, color);
        helper.material.transparent = true;
        helper.material.opacity = 0.95;
        helper.renderOrder = 7;
        this.scene.add(helper);
        this.boxes.set(actor.uuid, helper);
      } else {
        this.boxes.get(actor.uuid).setFromObject(actor);
      }
    }
    for (const [uuid, helper] of this.boxes) {
      if (!seen.has(uuid)) {
        this.scene.remove(helper);
        helper.geometry.dispose();
        this.boxes.delete(uuid);
      }
    }
  }

  dispose() {
    for (const helper of this.boxes.values()) {
      this.scene.remove(helper);
      helper.geometry.dispose();
    }
    this.boxes.clear();
  }
}
