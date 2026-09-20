import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";

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

    const count = 3200;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 3 + Math.random() * 24;
      const y = 0.05 + Math.random() * 3.8;
      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = Math.sin(angle) * radius;
      const t = radius / 27;
      colors[i * 3] = 0.15 + t * 0.55;
      colors[i * 3 + 1] = 0.82 - t * 0.35;
      colors[i * 3 + 2] = 1.0 - t * 0.25;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 0.06,
        vertexColors: true,
        transparent: true,
        opacity: 0.62,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    );
    this.points.renderOrder = 2;
    this.root.add(this.points);

    this.scanRing = new THREE.Mesh(
      new THREE.RingGeometry(3.6, 3.95, 72),
      new THREE.MeshBasicMaterial({
        color: 0x38bcd6,
        transparent: true,
        opacity: 0.42,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scanRing.rotation.x = -Math.PI / 2;
    this.scanRing.position.y = 0.11;
    this.root.add(this.scanRing);

    this.scanBeam = new THREE.Mesh(
      new THREE.CircleGeometry(14, 48, 0.4, 1.2),
      new THREE.MeshBasicMaterial({
        color: 0x007aff,
        transparent: true,
        opacity: 0.06,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scanBeam.rotation.x = -Math.PI / 2;
    this.scanBeam.position.y = 0.09;
    this.root.add(this.scanBeam);
    this._phase = 0;
  }

  setVisible(visible) {
    this.root.visible = visible;
    for (const helper of this.boxes.values()) {
      helper.visible = visible;
    }
  }

  update(ego, actors) {
    this.root.position.copy(ego.position);
    this.root.rotation.y = ego.rotation.y;
    this._phase += 0.055;
    this.scanRing.rotation.z = this._phase;
    this.scanBeam.rotation.z = this._phase * 0.65;

    const seen = new Set();
    for (const actor of actors) {
      if (actor === ego) continue;
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
