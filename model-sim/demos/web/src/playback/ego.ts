import * as THREE from "three";
import { detailedCar } from "../vendor/jevpilot/vehicle-model";
import { loadHeroCar } from "../vendor/jevpilot/model-assets";
import {
  cloneTrafficVehicle,
  loadTrafficFleet,
  type TrafficModelId,
} from "../vendor/jevpilot/traffic-assets";

/** Procedural placeholder — swapped for Model Y GLB once loaded (JevPilot scene.js ~798–817). */
export function createEgoVehicle() {
  return detailedCar("#e2e5e9");
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
