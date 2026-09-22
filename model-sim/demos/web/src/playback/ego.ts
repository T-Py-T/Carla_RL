import * as THREE from "three";
import { detailedCar } from "../vendor/jevpilot/vehicle-model";
import { loadHeroCar } from "../vendor/jevpilot/model-assets";
import {
  cloneTrafficVehicle,
  loadTrafficFleet,
  type TrafficModelId,
} from "../vendor/jevpilot/traffic-assets";

/** Placeholder only until Tesla Model Y GLB loads. Ego mesh is Model Y — never a traffic kit car. */
export function createEgoVehicle() {
  return detailedCar("#e2e5e9");
}

function opaqueStandard(
  name: string,
  color: string,
  extras: THREE.MeshStandardMaterialParameters = {},
) {
  return new THREE.MeshStandardMaterial({
    name,
    color,
    metalness: 0.28,
    roughness: 0.42,
    transparent: false,
    opacity: 1,
    depthWrite: true,
    depthTest: true,
    side: THREE.FrontSide,
    ...extras,
  });
}

function addBox(
  group: THREE.Group,
  mat: THREE.Material,
  w: number,
  h: number,
  d: number,
  x: number,
  y: number,
  z: number,
) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  mesh.position.set(x, y, z);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.renderOrder = 2;
  group.add(mesh);
  return mesh;
}

/**
 * Capture-safe Model Y built from opaque boxes only — no glass quads, no
 * MeshPhysical transmission, no shared `physical()` cache. Headless Chromium
 * cannot decode the Draco GLB; the undraco 20-mesh ghost is refused.
 */
export function createOpaqueModelY() {
  const car = new THREE.Group();
  const paint = opaqueStandard("model-y-body-opaque", "#d5d9de", { metalness: 0.32, roughness: 0.36 });
  const glass = opaqueStandard("model-y-glass-opaque", "#141b22", { metalness: 0.2, roughness: 0.3 });
  const rubber = opaqueStandard("model-y-tire-opaque", "#121314", { metalness: 0.05, roughness: 0.95 });
  const lens = opaqueStandard("model-y-light-opaque", "#dce6f2", {
    metalness: 0.15,
    roughness: 0.25,
    emissive: "#c5d8ea",
    emissiveIntensity: 0.45,
  });
  const tail = opaqueStandard("model-y-tail-opaque", "#b01018", {
    metalness: 0.2,
    roughness: 0.35,
    emissive: "#8a0008",
    emissiveIntensity: 0.7,
  });
  const trim = opaqueStandard("model-y-trim-opaque", "#1c1e22", { metalness: 0.4, roughness: 0.5 });

  // ~4.75 × 1.92 × 1.62 Model Y silhouette. Cabin glass is a dark SOLID inset.
  addBox(car, paint, 1.92, 0.70, 4.75, 0, 0.58, 0);
  addBox(car, trim, 1.86, 0.12, 4.55, 0, 0.28, 0);
  addBox(car, paint, 1.72, 0.52, 2.35, 0, 1.14, 0.18);
  addBox(car, glass, 1.62, 0.38, 2.15, 0, 1.16, 0.16);
  addBox(car, paint, 1.58, 0.10, 1.55, 0, 1.46, 0.22);
  addBox(car, paint, 1.78, 0.16, 0.95, 0, 0.92, -1.55);
  addBox(car, paint, 1.70, 0.14, 0.72, 0, 0.90, 1.72);
  addBox(car, lens, 0.62, 0.12, 0.08, -0.52, 0.70, -2.36);
  addBox(car, lens, 0.62, 0.12, 0.08, 0.52, 0.70, -2.36);
  addBox(car, tail, 0.62, 0.12, 0.08, -0.52, 0.70, 2.36);
  addBox(car, tail, 0.62, 0.12, 0.08, 0.52, 0.70, 2.36);
  for (const z of [-1.48, 1.42]) {
    for (const x of [-0.84, 0.84]) {
      addBox(car, rubber, 0.30, 0.58, 0.58, x, 0.30, z);
    }
  }

  car.name = "tesla-model-y";
  car.userData.eyeHeight = 1.28;
  car.userData.eyeForward = 0.45;
  car.userData.wheelbase = 2.9;
  car.userData.width = 1.92;
  car.userData.depth = 4.75;
  car.userData.sourcedModel = true;
  car.userData.opaqueHull = true;
  return car;
}

/** Opaque NPC sedan — no glass. Dark-blue lead is the capture hazard car. */
export function createOpaqueSedan(color: string, label = "npc-sedan") {
  const car = new THREE.Group();
  const paint = opaqueStandard(`${label}-paint`, color, { metalness: 0.3, roughness: 0.4 });
  const glass = opaqueStandard(`${label}-glass`, "#151a20", { metalness: 0.18, roughness: 0.32 });
  const rubber = opaqueStandard(`${label}-tire`, "#121314", { metalness: 0.05, roughness: 0.95 });
  addBox(car, paint, 1.78, 0.58, 4.20, 0, 0.52, 0);
  addBox(car, paint, 1.58, 0.46, 2.05, 0, 1.02, 0.12);
  addBox(car, glass, 1.48, 0.32, 1.85, 0, 1.04, 0.10);
  addBox(car, paint, 1.42, 0.08, 1.25, 0, 1.30, 0.16);
  for (const z of [-1.28, 1.22]) {
    for (const x of [-0.78, 0.78]) {
      addBox(car, rubber, 0.26, 0.52, 0.52, x, 0.27, z);
    }
  }
  car.userData.kind = "vehicle";
  car.userData.label = label;
  car.userData.modelId = label;
  car.userData.depth = 4.2;
  car.userData.width = 1.78;
  return car;
}

export function auditOpaqueMaterials(root: THREE.Object3D) {
  let meshCount = 0;
  let transparentMeshes = 0;
  let minOpacity = 1;
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.material) return;
    meshCount += 1;
    const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const src of list) {
      const mat = src as THREE.MeshPhysicalMaterial;
      const opacity = typeof mat.opacity === "number" ? mat.opacity : 1;
      minOpacity = Math.min(minOpacity, opacity);
      const trans =
        mat.transparent === true ||
        opacity < 0.99 ||
        (typeof mat.transmission === "number" && mat.transmission > 0);
      if (trans) transparentMeshes += 1;
    }
  });
  return { meshCount, transparentMeshes, minOpacity };
}

const NPC_FALLBACK_COLORS = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085"];

/** Licensed GLB NPC fleet — NOT Model Y. Ghost = kinematic clone, NOT mesh opacity. */
export function createNpcVehicle(modelId: TrafficModelId) {
  return cloneTrafficVehicle(modelId);
}

/** Procedural NPC — no GLB (headless capture / fleet-unavailable fallback). */
export function createProceduralNpcVehicle(modelId: TrafficModelId) {
  const idx = Math.abs(modelId.split("").reduce((a, c) => a + c.charCodeAt(0), 0));
  return createOpaqueSedan(NPC_FALLBACK_COLORS[idx % NPC_FALLBACK_COLORS.length], modelId);
}

/** Kenney GLB when loaded, else procedural. */
export function createNpcVehicleFallback(modelId: TrafficModelId) {
  try {
    return createNpcVehicle(modelId);
  } catch {
    return createProceduralNpcVehicle(modelId);
  }
}

export { loadTrafficFleet };

export function createPedestrian(shirt = "#c27d55") {
  const group = new THREE.Group();
  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.18, 0.45, 6, 10),
    new THREE.MeshStandardMaterial({ color: shirt, transparent: false, opacity: 1, depthWrite: true }),
  );
  torso.position.y = 0.95;
  group.add(torso);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 10, 10),
    new THREE.MeshStandardMaterial({ color: "#dec060", transparent: false, opacity: 1, depthWrite: true }),
  );
  head.position.y = 1.45;
  group.add(head);
  group.userData.kind = "pedestrian";
  group.userData.label = "Pedestrian";
  return group;
}

export { loadHeroCar };
