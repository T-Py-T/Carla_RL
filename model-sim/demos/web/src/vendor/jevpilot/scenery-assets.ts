// @ts-nocheck — Vendored from standardagents/jevpilot src/scenery-assets.js (street lamps)
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { assetManager } from "./asset-loading";
import { renderProfile } from "./render-profile";

const assets = new Map<string, Promise<{ geometry: THREE.BufferGeometry; material: THREE.Material }[]>>();

function loadAsset(name: string, file: string) {
  if (!assets.has(name)) {
    assets.set(
      name,
      new GLTFLoader(assetManager)
        .loadAsync(`/models/${name}/${file}.glb`)
        .then(({ scene }) => {
          scene.updateMatrixWorld(true);
          const bounds = new THREE.Box3().setFromObject(scene);
          const center = bounds.getCenter(new THREE.Vector3());
          const size = bounds.getSize(new THREE.Vector3());
          const normalize = new THREE.Matrix4()
            .makeScale(1 / size.y, 1 / size.y, 1 / size.y)
            .multiply(new THREE.Matrix4().makeTranslation(-center.x, -bounds.min.y, -center.z));
          const parts: { geometry: THREE.BufferGeometry; material: THREE.Material }[] = [];
          scene.traverse((mesh) => {
            if (!mesh.isMesh) return;
            const mat = mesh.material as THREE.MeshStandardMaterial;
            mat.envMapIntensity = 0.5;
            parts.push({
              geometry: mesh.geometry.clone().applyMatrix4(normalize.clone().multiply(mesh.matrixWorld)),
              material: mat,
            });
          });
          return parts;
        }),
    );
  }
  return assets.get(name)!;
}

export type StreetLightEdge = {
  ax: number;
  az: number;
  bx: number;
  bz: number;
  length: number;
};

/** Instanced Poly Haven street_lamp_01 along road edges (JevPilot scenery-assets.js). */
export async function placeStreetLamps(scene: THREE.Scene, edges: StreetLightEdge[]) {
  const parts = await loadAsset("street_lamp_01", "lamp");
  const locations: { x: number; z: number; height: number; rotation: number }[] = [];
  for (const edge of edges) {
    const h = Math.atan2(edge.bx - edge.ax, edge.az - edge.bz);
    for (let distance = 26; distance < edge.length - 20; distance += 42) {
      const side = Math.round(distance / 42) % 2 ? -1 : 1;
      locations.push({
        x: edge.ax + Math.sin(h) * distance + Math.cos(h) * 7.25 * side,
        z: edge.az - Math.cos(h) * distance + Math.sin(h) * 7.25 * side,
        height: 6.8,
        rotation: h + (side > 0 ? Math.PI / 2 : -Math.PI / 2),
      });
    }
  }
  const dummy = new THREE.Object3D();
  for (const part of parts) {
    const mesh = new THREE.InstancedMesh(part.geometry, part.material, locations.length);
    mesh.name = "street-lamp";
    locations.forEach((p, i) => {
      dummy.position.set(p.x, 0, p.z);
      dummy.rotation.set(0, p.rotation, 0);
      dummy.scale.setScalar(p.height);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }
}
