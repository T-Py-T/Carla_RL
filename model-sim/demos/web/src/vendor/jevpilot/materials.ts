/**
 * Vendored from standardagents/jevpilot — src/materials.js
 * Poly Haven CC0 PBR maps in public/textures/ (see public/textures/LICENSE.md).
 */
import * as THREE from "three";
import { assetManager } from "./asset-loading";
import { renderProfile } from "./render-profile";

export const materials = new Map<string, THREE.Material>();
const loader = new THREE.TextureLoader(assetManager);
const maps = new Map<string, THREE.Texture>();

function texture(name: string, kind: "color" | "normal" | "roughness") {
  const key = `${name}-${kind}`;
  if (!maps.has(key)) {
    const value = loader.load(`/textures/${key}.jpg`);
    value.wrapS = value.wrapT = THREE.RepeatWrapping;
    value.anisotropy = renderProfile.anisotropy;
    if (kind === "color") value.colorSpace = THREE.SRGBColorSpace;
    maps.set(key, value);
  }
  return maps.get(key)!;
}

export function pbr(name: string, tint = "#ffffff", scale = 3) {
  const key = `${name}:${tint}`;
  if (!materials.has(key)) {
    const mat = new THREE.MeshStandardMaterial({
      color: tint,
      map: texture(name, "color"),
      normalMap: texture(name, "normal"),
      roughnessMap: texture(name, "roughness"),
      normalScale: new THREE.Vector2(0.65, 0.65),
      roughness: 0.95,
      metalness: 0,
    });
    mat.userData.metersPerTile = scale;
    materials.set(key, mat);
  }
  return materials.get(key) as THREE.MeshStandardMaterial;
}

const COLOR_FAMILY: Record<string, [string, string, number]> = {
  "#73817e": ["asphalt", "#d2d5d9", 5],
  "#70817c": ["asphalt", "#c6cbd2", 5],
  "#d8d6c9": ["pavement", "#d7d4ca", 2.5],
};

export function material(color: string | THREE.Material): THREE.Material {
  if (color instanceof THREE.Material) return color;
  if (materials.has(color)) return materials.get(color)!;
  const family = COLOR_FAMILY[color];
  const value = family
    ? pbr(...family)
    : new THREE.MeshStandardMaterial({ color, roughness: 0.75, metalness: 0 });
  materials.set(color, value);
  return value;
}

export function physical(name: string, options: THREE.MeshPhysicalMaterialParameters): THREE.MeshPhysicalMaterial {
  if (!materials.has(name)) {
    materials.set(name, new THREE.MeshPhysicalMaterial(options));
  }
  return materials.get(name) as THREE.MeshPhysicalMaterial;
}

/** Meter-scaled UVs — maps keep grain size on every road block (JevPilot materials.js). */
export function metricUV(geometry: THREE.BufferGeometry, mat: THREE.Material) {
  const scale = (mat as THREE.MeshStandardMaterial).userData.metersPerTile as number | undefined;
  if (!scale) return geometry;
  const position = geometry.attributes.position;
  const normal = geometry.attributes.normal;
  const uv = new Float32Array(position.count * 2);
  for (let i = 0; i < position.count; i++) {
    const x = Math.abs(normal.getX(i));
    const y = Math.abs(normal.getY(i));
    const z = Math.abs(normal.getZ(i));
    uv[i * 2] = (x > y && x > z ? position.getZ(i) : position.getX(i)) / scale;
    uv[i * 2 + 1] = (y >= x && y >= z ? position.getZ(i) : position.getY(i)) / scale;
  }
  geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  return geometry;
}
