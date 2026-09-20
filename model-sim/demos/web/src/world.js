import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";
import { createNpcVehicle, createPedestrian } from "./ego-vehicle.js";

export function buildTown(scene) {
  scene.background = new THREE.Color("#9eb6cc");
  scene.fog = new THREE.Fog("#9eb6cc", 45, 220);

  scene.add(new THREE.HemisphereLight("#eef4fb", "#607580", 1.15));
  const sun = new THREE.DirectionalLight("#fff8ee", 1.05);
  sun.position.set(30, 55, 20);
  scene.add(sun);
  const fill = new THREE.DirectionalLight("#b8d4f0", 0.35);
  fill.position.set(-20, 18, -10);
  scene.add(fill);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(320, 320),
    new THREE.MeshStandardMaterial({ color: "#d8d6c9", roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const roadMat = new THREE.MeshStandardMaterial({ color: "#5f716c", roughness: 0.92 });
  const mainRoad = new THREE.Mesh(new THREE.PlaneGeometry(13, 280), roadMat);
  mainRoad.rotation.x = -Math.PI / 2;
  mainRoad.position.y = 0.04;
  scene.add(mainRoad);

  const crossRoad = new THREE.Mesh(new THREE.PlaneGeometry(70, 13), roadMat);
  crossRoad.rotation.x = -Math.PI / 2;
  crossRoad.position.set(0, 0.045, -35);
  scene.add(crossRoad);

  const sidewalkMat = new THREE.MeshStandardMaterial({ color: "#c9c5b8", roughness: 0.95 });
  for (const offset of [-8.2, 8.2]) {
    const sw = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 280), sidewalkMat);
    sw.rotation.x = -Math.PI / 2;
    sw.position.set(offset, 0.035, 0);
    scene.add(sw);
  }

  const palette = ["#8d9cab", "#7f93a3", "#6d8494", "#95a8b8", "#566878"];
  for (let block = 0; block < 18; block++) {
    const z = -120 + block * 14;
    for (const side of [-1, 1]) {
      const w = 5 + (block % 4) * 1.2;
      const h = 8 + (block % 5) * 2.5;
      const d = 7 + (block % 3);
      const building = new THREE.Mesh(
        new THREE.BoxGeometry(w, h, d),
        new THREE.MeshStandardMaterial({
          color: palette[(block + (side > 0 ? 1 : 0)) % palette.length],
          roughness: 0.82,
        }),
      );
      building.position.set(side * (11 + (block % 2)), h / 2, z);
      scene.add(building);

      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 3; col++) {
          const win = new THREE.Mesh(
            new THREE.PlaneGeometry(0.9, 1.1),
            new THREE.MeshStandardMaterial({
              color: block % 3 ? "#dbe4ee" : "#c8d2dc",
              emissive: block % 4 ? "#334455" : "#000000",
              emissiveIntensity: block % 4 ? 0.15 : 0,
            }),
          );
          win.position.set(
            building.position.x - side * (d / 2 + 0.02),
            2 + row * 2.2,
            building.position.z - 2 + col * 2,
          );
          win.rotation.y = side > 0 ? -Math.PI / 2 : Math.PI / 2;
          scene.add(win);
        }
      }
    }
  }

  for (let i = 0; i < 24; i++) {
    const z = -100 + i * 9;
    for (const side of [-1, 1]) {
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.12, 0.16, 1.6, 8),
        new THREE.MeshStandardMaterial({ color: "#5a4638" }),
      );
      trunk.position.set(side * 9.5, 0.8, z + (i % 2));
      scene.add(trunk);
      const foliage = new THREE.Mesh(
        new THREE.SphereGeometry(0.75 + (i % 3) * 0.15, 8, 8),
        new THREE.MeshStandardMaterial({ color: "#4f8a62", roughness: 0.9 }),
      );
      foliage.position.set(trunk.position.x, 2.0, trunk.position.z);
      scene.add(foliage);
    }
  }

  for (let i = 0; i < 16; i++) {
    const z = -90 + i * 12;
    for (const side of [-1, 1]) {
      const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.06, 0.08, 4.5, 6),
        new THREE.MeshStandardMaterial({ color: "#444" }),
      );
      pole.position.set(side * 10.5, 2.25, z);
      scene.add(pole);
      const lamp = new THREE.Mesh(
        new THREE.SphereGeometry(0.18, 8, 8),
        new THREE.MeshStandardMaterial({ color: "#fff6dd", emissive: "#ffaa55", emissiveIntensity: 0.4 }),
      );
      lamp.position.set(pole.position.x, 4.5, pole.position.z);
      scene.add(lamp);
    }
  }
}

export class TrafficSystem {
  constructor(scene) {
    this.scene = scene;
    this.actors = [];
    this._spawnInitial();
  }

  _spawnInitial() {
    const specs = [
      { type: "vehicle", x: -2.5, z: -18, speed: 0.09, color: "#4a6678" },
      { type: "vehicle", x: 2.8, z: -32, speed: 0.07, color: "#7a5a48" },
      { type: "vehicle", x: 0.5, z: -52, speed: 0.11, color: "#3d6b58" },
      { type: "vehicle", x: -3.2, z: 12, speed: -0.08, color: "#6a5068" },
      { type: "pedestrian", x: -9, z: -22, speed: 0.025, lane: 1 },
      { type: "pedestrian", x: 9.2, z: -40, speed: -0.02, lane: -1 },
      { type: "pedestrian", x: -9, z: -58, speed: 0.018, lane: 1 },
    ];
    for (const spec of specs) {
      const mesh =
        spec.type === "vehicle"
          ? createNpcVehicle(spec.color)
          : createPedestrian(spec.lane > 0 ? "#548975" : "#c27d55");
      mesh.position.set(spec.x, 0, spec.z);
      if (spec.speed < 0) mesh.rotation.y = Math.PI;
      this.scene.add(mesh);
      this.actors.push({ mesh, speed: spec.speed, kind: spec.type });
    }
  }

  step(ego, tick = 0) {
    for (const actor of this.actors) {
      actor.mesh.position.z -= actor.speed;
      if (actor.kind === "pedestrian") {
        actor.mesh.position.x += Math.sin(tick * 0.09 + actor.mesh.position.z * 0.1) * 0.005;
      }
      if (actor.mesh.position.z < ego.position.z - 80) {
        actor.mesh.position.z = ego.position.z + 40 + Math.random() * 20;
      }
      if (actor.mesh.position.z > ego.position.z + 30 && actor.speed < 0) {
        actor.mesh.position.z = ego.position.z - 60;
      }
    }
    return this.actors.map((a) => a.mesh);
  }
}
