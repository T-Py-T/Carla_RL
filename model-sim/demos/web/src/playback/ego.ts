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

/** Licensed GLB NPC fleet — NOT Model Y. Ghost = kinematic clone, NOT mesh opacity. */
export function createNpcVehicle(modelId: TrafficModelId) {
  return cloneTrafficVehicle(modelId);
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
