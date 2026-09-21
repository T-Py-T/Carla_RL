// @ts-nocheck — crude buildings (intentionally not JevPilot scenery)
import * as THREE from "three";
import { buildTownRoadNetwork } from "../vendor/jevpilot/jevpilot-road";
import { placeStreetLamps } from "../vendor/jevpilot/scenery-assets";
import { createNpcVehicleFallback, createPedestrian, createProceduralNpcVehicle } from "./ego";

const MIN_VEHICLE_GAP = 9.5;
const MIN_SPAWN_AHEAD = 14;
/** Four-lane one-way main road (−Z travel) — 3 m lanes across 12 m pavement. */
export const ONE_WAY_LANE_X = [-4.5, -1.5, 1.5, 4.5];
export const EGO_LANE_X = ONE_WAY_LANE_X[1];
const LANE_X = ONE_WAY_LANE_X;
/** Main × cross junction — keep building boxes out of this band for open sightlines. */
const INTERSECTION_Z = -35;
const JUNCTION_CLEARANCE_Z = 26;

function blocksJunction(z: number, halfDepth = 0) {
  return Math.abs(z - INTERSECTION_Z) < JUNCTION_CLEARANCE_Z + halfDepth;
}

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
    new THREE.MeshStandardMaterial({ color: "#b2c5a0", roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  scene.add(ground);

  const roadGroup = new THREE.Group();
  scene.add(roadGroup);
  scene.userData.trafficLights = buildTownRoadNetwork(roadGroup, {
    mainRoad: { length: 280, cx: 0, cz: 0 },
    crossRoad: { length: 70, cx: 0, cz: -35 },
    intersection: { cx: 0, cz: -35, offset: 0 },
  });

  // Poly Haven CC0 street_lamp_01 — JevPilot scenery-assets.js placement along main + cross roads.
  void placeStreetLamps(scene, [
    { ax: 0, az: -140, bx: 0, bz: 140, length: 280 },
    { ax: -35, az: -35, bx: 35, bz: -35, length: 70 },
  ]);

  const palette = ["#8d9cab", "#7f93a3", "#6d8494", "#95a8b8", "#566878"];
  for (let block = 0; block < 18; block++) {
    const z = -120 + block * 14;
    for (const side of [-1, 1]) {
      const w = 5 + (block % 4) * 1.2;
      const h = 8 + (block % 5) * 2.5;
      const d = 7 + (block % 3);
      if (blocksJunction(z, d / 2)) continue;
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
    if (blocksJunction(z + (i % 2))) continue;
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

}

export class TrafficSystem {
  constructor(scene) {
    this.scene = scene;
    this.actors = [];
  }

  spawnInitial(proceduralTraffic = false) {
    if (this.actors.length) return;
    this._spawnInitial(proceduralTraffic);
  }

  _spawnInitial(proceduralTraffic = false) {
    // Research CC shortlist — Kenney Car Kit + Khronos + OGA UAZ (NOT Model Y).
    // After `npm run fetch:sketchfab-traffic`, swap in SKETCHFAB_TRAFFIC_MODELS ids for mixed fleet.
    // All NPC vehicles ahead in parallel one-way lanes — slow lead in ego lane for braking demo.
    const specs = [
      { type: "vehicle", x: EGO_LANE_X, z: -8.5, speed: 0.022, modelId: "kenney-sedan" },
      { type: "vehicle", x: ONE_WAY_LANE_X[0], z: -18, speed: 0.075, modelId: "kenney-hatchback-sports" },
      { type: "vehicle", x: ONE_WAY_LANE_X[2], z: -26, speed: 0.068, modelId: "kenney-van" },
      { type: "vehicle", x: ONE_WAY_LANE_X[3], z: -34, speed: 0.072, modelId: "kenney-firetruck" },
      { type: "vehicle", x: ONE_WAY_LANE_X[0], z: -48, speed: 0.08, modelId: "kenney-truck" },
      { type: "vehicle", x: ONE_WAY_LANE_X[2], z: -58, speed: 0.07, modelId: "kenney-delivery" },
      { type: "vehicle", x: ONE_WAY_LANE_X[3], z: -70, speed: 0.065, modelId: "khronos-milk-truck" },
      { type: "vehicle", x: ONE_WAY_LANE_X[0], z: -82, speed: 0.06, modelId: "oga-uaz-truck" },
      { type: "pedestrian", x: -9, z: -22, speed: 0.025, lane: 1 },
      { type: "pedestrian", x: 9.2, z: -40, speed: 0.02, lane: -1 },
      { type: "pedestrian", x: -9, z: -58, speed: 0.018, lane: 1 },
    ];
    for (const spec of specs) {
      const mesh =
        spec.type === "vehicle"
          ? proceduralTraffic
            ? createProceduralNpcVehicle(spec.modelId)
            : createNpcVehicleFallback(spec.modelId)
          : createPedestrian(spec.lane > 0 ? "#548975" : "#c27d55");
      mesh.position.set(spec.x, 0, spec.z);
      if (spec.type === "vehicle") mesh.rotation.y = 0;
      this.scene.add(mesh);
      this.actors.push({ mesh, speed: spec.speed, kind: spec.type });
    }
  }

  _laneOverlap(ax, bx) {
    return Math.abs(ax - bx) < 1.35;
  }

  _gapToEgo(mesh, ego) {
    return ego.position.z - mesh.position.z;
  }

  _canPlaceVehicle(x, z, ego, ignoreMesh = null) {
    if (z > ego.position.z - MIN_SPAWN_AHEAD) return false;
    for (const actor of this.actors) {
      if (actor.kind !== "vehicle" || actor.mesh === ignoreMesh) continue;
      if (this._laneOverlap(x, actor.mesh.position.x) && Math.abs(z - actor.mesh.position.z) < MIN_VEHICLE_GAP) {
        return false;
      }
    }
    return Math.hypot(x - ego.position.x, z - ego.position.z) >= MIN_VEHICLE_GAP;
  }

  _respawnVehicleAhead(actor, ego) {
    for (let attempt = 0; attempt < 16; attempt += 1) {
      const x = LANE_X[Math.floor(Math.random() * LANE_X.length)];
      const z = ego.position.z - 28 - Math.random() * 48;
      if (this._canPlaceVehicle(x, z, ego, actor.mesh)) {
        actor.mesh.position.set(x, 0, z);
        return;
      }
    }
  }

  _leadGap(ego) {
    let minGap = Infinity;
    for (const actor of this.actors) {
      if (actor.kind !== "vehicle") continue;
      const gap = this._gapToEgo(actor.mesh, ego);
      if (gap < 4 || gap > 120) continue;
      if (!this._laneOverlap(actor.mesh.position.x, ego.position.x)) continue;
      minGap = Math.min(minGap, gap);
    }
    return minGap;
  }

  step(ego, tick = 0) {
    for (const actor of this.actors) {
      if (actor.kind === "vehicle") {
        const nextZ = actor.mesh.position.z - actor.speed;
        const gapAfter = ego.position.z - nextZ;
        if (
          this._laneOverlap(actor.mesh.position.x, ego.position.x) &&
          gapAfter < MIN_VEHICLE_GAP &&
          gapAfter > -2
        ) {
          continue;
        }
        actor.mesh.position.z = nextZ;
      } else {
        actor.mesh.position.z -= actor.speed;
        actor.mesh.position.x += Math.sin(tick * 0.09 + actor.mesh.position.z * 0.1) * 0.005;
      }

      if (actor.kind === "vehicle" && actor.mesh.position.z < ego.position.z - 90) {
        this._respawnVehicleAhead(actor, ego);
      }
      if (actor.kind === "pedestrian" && actor.mesh.position.z < ego.position.z - 70) {
        actor.mesh.position.z = ego.position.z - 30 - Math.random() * 25;
      }
    }

    return {
      actors: this.actors.map((a) => a.mesh),
      leadGap: this._leadGap(ego),
    };
  }
}
