// @ts-nocheck — Research CC shortlist NPC GLB fleet (NOT Model Y; ego-only)
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { assetManager } from "./asset-loading";

/** Research CC shortlist — Kenney Car Kit CC0 + Khronos + OGA UAZ CC-BY. See public/models/traffic/ per pack. */
export const TRAFFIC_MODELS = [
  { id: "kenney-sedan", path: "/models/traffic/kenney-car-kit/sedan.glb", label: "Sedan", targetLength: 4.5 },
  {
    id: "kenney-hatchback-sports",
    path: "/models/traffic/kenney-car-kit/hatchback-sports.glb",
    label: "Sports hatchback",
    targetLength: 4.4,
  },
  { id: "kenney-van", path: "/models/traffic/kenney-car-kit/van.glb", label: "Van", targetLength: 4.9 },
  { id: "kenney-truck", path: "/models/traffic/kenney-car-kit/truck.glb", label: "Truck", targetLength: 5.4 },
  { id: "kenney-suv", path: "/models/traffic/kenney-car-kit/suv.glb", label: "SUV", targetLength: 4.8 },
  { id: "kenney-delivery", path: "/models/traffic/kenney-car-kit/delivery.glb", label: "Delivery van", targetLength: 5.6 },
  { id: "kenney-firetruck", path: "/models/traffic/kenney-car-kit/firetruck.glb", label: "Fire truck", targetLength: 6.2 },
  { id: "kenney-ambulance", path: "/models/traffic/kenney-car-kit/ambulance.glb", label: "Ambulance", targetLength: 5.8 },
  { id: "kenney-taxi", path: "/models/traffic/kenney-car-kit/taxi.glb", label: "Taxi", targetLength: 4.5 },
  {
    id: "khronos-toy-car",
    path: "/models/traffic/khronos-toy-car/toycar.glb",
    label: "Toy car",
    targetLength: 3.2,
  },
  {
    id: "khronos-milk-truck",
    path: "/models/traffic/khronos-milk-truck/truck.glb",
    label: "Milk truck",
    targetLength: 5.2,
  },
  {
    id: "oga-uaz-truck",
    path: "/models/traffic/oga-uaz-truck/uaz-truck.glb",
    label: "UAZ truck",
    targetLength: 5.0,
  },
] as const;

/** Sketchfab CC-BY shortlist — wired after `npm run fetch:sketchfab-traffic` (see scripts/sketchfab-traffic-models.json). */
export const SKETCHFAB_TRAFFIC_MODELS = [
  {
    id: "sketchfab-generic-sedan",
    path: "/models/traffic/sketchfab-generic-sedan/model.glb",
    label: "Generic sedan",
    targetLength: 4.5,
  },
  {
    id: "sketchfab-modern-sedan",
    path: "/models/traffic/sketchfab-modern-sedan/model.glb",
    label: "Modern sedan",
    targetLength: 4.6,
  },
  {
    id: "sketchfab-toyoace-van",
    path: "/models/traffic/sketchfab-toyoace-van/model.glb",
    label: "ToyoAce van",
    targetLength: 5.1,
  },
  {
    id: "sketchfab-renault-master-van",
    path: "/models/traffic/sketchfab-renault-master-van/model.glb",
    label: "Renault Master van",
    targetLength: 5.4,
  },
  {
    id: "sketchfab-gmc-school-bus",
    path: "/models/traffic/sketchfab-gmc-school-bus/model.glb",
    label: "School bus",
    targetLength: 10.5,
  },
  {
    id: "sketchfab-box-truck",
    path: "/models/traffic/sketchfab-box-truck/model.glb",
    label: "Box truck",
    targetLength: 7.0,
  },
  {
    id: "sketchfab-delivery-truck",
    path: "/models/traffic/sketchfab-delivery-truck/model.glb",
    label: "Delivery truck",
    targetLength: 6.0,
  },
] as const;

export type TrafficModelId = (typeof TRAFFIC_MODELS)[number]["id"];

const templates = new Map<TrafficModelId, THREE.Group>();
let fleetReady: Promise<Map<TrafficModelId, THREE.Group>> | undefined;

function normalizeTrafficModel(scene: THREE.Group, targetLength: number) {
  // Match Model Y heading — Kenney/Khronos GLBs face +Z; playback travels −Z.
  scene.rotation.set(0, -Math.PI / 2, 0);
  scene.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(scene);
  const size = bounds.getSize(new THREE.Vector3());
  const maxHoriz = Math.max(size.x, size.z);
  const scale = maxHoriz > 0 ? targetLength / maxHoriz : 1;
  scene.scale.setScalar(scale);
  scene.updateMatrixWorld(true);
  const fitted = new THREE.Box3().setFromObject(scene);
  const fittedCenter = fitted.getCenter(new THREE.Vector3());
  scene.position.set(-fittedCenter.x, -fitted.min.y, -fittedCenter.z);
  scene.updateMatrixWorld(true);
  const fittedSize = new THREE.Box3().setFromObject(scene).getSize(new THREE.Vector3());
  scene.userData.width = fittedSize.x;
  scene.userData.depth = fittedSize.z;
  scene.userData.sourcedModel = true;
  scene.traverse((node) => {
    if (node.isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
    }
  });
  return scene;
}

export async function loadTrafficFleet() {
  fleetReady ||= (async () => {
    const decoder = new DRACOLoader(assetManager).setDecoderPath("/draco/");
    const loader = new GLTFLoader(assetManager).setDRACOLoader(decoder);
    for (const spec of TRAFFIC_MODELS) {
      const { scene } = await loader.loadAsync(spec.path);
      const model = normalizeTrafficModel(scene, spec.targetLength);
      model.name = spec.id;
      templates.set(spec.id, model);
    }
    decoder.dispose();
    return templates;
  })();
  return fleetReady;
}

export function cloneTrafficVehicle(modelId: TrafficModelId) {
  const template = templates.get(modelId);
  if (!template) throw new Error(`Traffic model not loaded: ${modelId}`);
  const clone = template.clone(true);
  clone.traverse((node) => {
    if (node.isMesh) node.geometry = node.geometry.clone();
  });
  clone.userData.kind = "vehicle";
  clone.userData.label = TRAFFIC_MODELS.find((m) => m.id === modelId)?.label ?? "Vehicle";
  clone.userData.modelId = modelId;
  return clone;
}
