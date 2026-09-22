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

/**
 * Capture-safe Model Y: crossover proportions, MeshStandardMaterial, no
 * transmission / glass opacity. Headless Chromium cannot decode the Draco
 * GLB; the undraco 20-mesh ghost is refused. Interactive browsers still
 * load the real GLB via loadHeroCar() + hardenEgoMaterials().
 */
export function createOpaqueModelY() {
  const car = detailedCar("#e1e4e8");
  const lengthScale = 4.75 / 4.16;
  car.scale.set(1.04, 1.18, lengthScale);
  car.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.material) return;
    const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const next = list.map((src) => {
      const phys = src as THREE.MeshPhysicalMaterial;
      const name = `${phys.name || ""}`.toLowerCase();
      const glass = name.includes("glass") || phys.transparent === true || (phys.opacity ?? 1) < 0.99;
      const lens = name.includes("light") || name.includes("lens") || name.includes("head") || name.includes("tail");
      return new THREE.MeshStandardMaterial({
        name: glass ? "model-y-glass-opaque" : phys.name || "model-y-body-opaque",
        color: glass ? (lens ? "#dce6f2" : "#151c24") : phys.color?.clone() ?? new THREE.Color("#e1e4e8"),
        metalness: glass ? 0.22 : Math.min(phys.metalness ?? 0.35, 0.55),
        roughness: glass ? 0.28 : Math.max(phys.roughness ?? 0.32, 0.22),
        emissive: phys.emissive?.clone() ?? new THREE.Color(0x000000),
        emissiveIntensity: phys.emissiveIntensity ?? 0,
        transparent: false,
        opacity: 1,
        depthWrite: true,
        depthTest: true,
        side: THREE.FrontSide,
      });
    });
    mesh.material = Array.isArray(mesh.material) ? next : next[0];
    mesh.renderOrder = 0;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
  });
  car.name = "tesla-model-y";
  car.userData.eyeHeight = 1.28;
  car.userData.eyeForward = 0.45;
  car.userData.wheelbase = 2.89;
  car.userData.width = 1.98;
  car.userData.depth = 4.75;
  car.userData.sourcedModel = true;
  car.userData.opaqueHull = true;
  return car;
}

const NPC_FALLBACK_COLORS = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085"];

/** Licensed GLB NPC fleet — NOT Model Y. Ghost = kinematic clone, NOT mesh opacity. */
export function createNpcVehicle(modelId: TrafficModelId) {
  return cloneTrafficVehicle(modelId);
}

/** Procedural NPC — no GLB (headless capture / fleet-unavailable fallback). */
export function createProceduralNpcVehicle(modelId: TrafficModelId) {
  const idx = Math.abs(modelId.split("").reduce((a, c) => a + c.charCodeAt(0), 0));
  const car = detailedCar(NPC_FALLBACK_COLORS[idx % NPC_FALLBACK_COLORS.length]);
  car.userData.kind = "vehicle";
  car.userData.label = modelId;
  car.userData.modelId = modelId;
  return car;
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
    new THREE.MeshStandardMaterial({ color: shirt }),
  );
  torso.position.y = 0.95;
  group.add(torso);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 10, 10),
    new THREE.MeshStandardMaterial({ color: "#dec060" }),
  );
  head.position.y = 1.45;
  group.add(head);
  group.userData.kind = "pedestrian";
  group.userData.label = "Pedestrian";
  return group;
}

export { loadHeroCar };
