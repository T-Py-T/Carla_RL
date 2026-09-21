/**
 * Vendored from standardagents/jevpilot — src/scene.js (roads, crosswalks, signals)
 * and src/world.js (signalState). Attributed adaptation for Carla_RL FSD playback.
 */
import * as THREE from "three";
import { material, metricUV } from "./materials";

const COLORS = {
  sidewalk: "#d8d6c9",
  asphalt: "#73817e",
  dash: "#d5d7b4",
  curb: "#b6c0b0",
  marking: "#ecebd9",
  pole: "#566861",
  housing: "#344e47",
  lampOff: "#394d43",
  lampOn: ["#f0836b", "#f4cb69", "#afdf92"],
  lampEmissive: ["#98301d", "#ad770e", "#508e38"],
  lampDim: "#34483e",
};

function box(
  parent: THREE.Object3D,
  w: number,
  h: number,
  d: number,
  x: number,
  y: number,
  z: number,
  color: string,
) {
  const mat = material(color);
  const geo = new THREE.BoxGeometry(w, h, d);
  metricUV(geo, mat);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, y, z);
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function cyl(
  parent: THREE.Object3D,
  r: number,
  h: number,
  x: number,
  y: number,
  z: number,
  color: string,
  segments = 8,
) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, segments), material(color));
  mesh.position.set(x, y + h / 2, z);
  parent.add(mesh);
  return mesh;
}

function heading(from: { x: number; z: number }, to: { x: number; z: number }) {
  return Math.atan2(to.x - from.x, to.z - from.z);
}

function move(p: { x: number; z: number }, angle: number, dist: number) {
  return { x: p.x + Math.sin(angle) * dist, z: p.z + Math.cos(angle) * dist };
}

/** Port of JevPilot world.js signalState — 24s cycle, NS/EW offset phases. */
export function signalState(
  node: { offset: number },
  time: number,
  approach: number,
): { color: "red" | "amber" | "green"; walk: boolean; remaining: number } {
  const phase = (time + node.offset) % 24;
  const ns = Math.abs(Math.cos(approach)) > 0.5;
  if (phase >= 20) return { color: "red", walk: true, remaining: 24 - phase };
  if (ns) {
    return {
      color: phase < 8 ? "green" : phase < 10 ? "amber" : "red",
      walk: false,
      remaining: phase < 8 ? 8 - phase : phase < 10 ? 10 - phase : 24 - phase,
    };
  }
  return {
    color: phase >= 10 && phase < 18 ? "green" : phase >= 18 ? "amber" : "red",
    walk: false,
    remaining: phase < 10 ? 10 - phase : phase < 18 ? 18 - phase : 20 - phase,
  };
}

export type TrafficLightLamp = {
  mesh: THREE.Mesh;
  index: number;
  approach: number;
  offset: number;
};

/** One straight road segment — sidewalk, asphalt, lane markings, curb strips. */
export function buildStraightRoad(
  parent: THREE.Object3D,
  opts: {
    vertical: boolean;
    length: number;
    cx: number;
    cz: number;
    /** When true, draw multi-lane one-way dividers (no opposing-lane centerline). */
    oneWay?: boolean;
  },
) {
  const { vertical, length, cx, cz, oneWay = false } = opts;
  box(
    parent,
    vertical ? 16 : length + 0.2,
    0.26,
    vertical ? length + 0.2 : 16,
    cx,
    -0.12,
    cz,
    COLORS.sidewalk,
  );
  box(
    parent,
    vertical ? 12 : length + 0.3,
    0.1,
    vertical ? length + 0.3 : 12,
    cx,
    0.015,
    cz,
    COLORS.asphalt,
  );
  const half = length / 2;
  if (oneWay) {
    // Four same-direction lanes — dashed dividers at ±3 m (12 m pavement).
    const dividerOffsets = [-3, 0, 3];
    for (const offset of dividerOffsets) {
      for (let k = 12; k < length - 10; k += 7) {
        const t = k - half;
        box(
          parent,
          vertical ? 0.08 : 2.6,
          0.02,
          vertical ? 2.6 : 0.08,
          vertical ? cx + offset : cx - half + k,
          0.081,
          vertical ? cz + t : cz + offset,
          COLORS.dash,
        );
      }
    }
    // Directional chevrons — all lanes travel −Z on the vertical main road.
    for (let k = 18; k < length - 14; k += 22) {
      const t = k - half;
      for (const lane of [-4.5, -1.5, 1.5, 4.5]) {
        box(
          parent,
          vertical ? 0.55 : 0.9,
          0.02,
          vertical ? 0.9 : 0.55,
          vertical ? cx + lane : cx - half + k,
          0.082,
          vertical ? cz + t : cz + lane,
          COLORS.marking,
        );
      }
    }
  } else {
    for (let k = 12; k < length - 10; k += 8) {
      const t = k - half;
      box(
        parent,
        vertical ? 0.13 : 3.2,
        0.02,
        vertical ? 3.2 : 0.13,
        vertical ? cx : cx - half + k,
        0.081,
        vertical ? cz + t : cz,
        COLORS.dash,
      );
    }
  }
  for (const dir of [-1, 1]) {
    box(
      parent,
      vertical ? 0.1 : length - 18,
      0.015,
      vertical ? length - 18 : 0.1,
      vertical ? cx + dir * 5.5 : cx,
      0.077,
      vertical ? cz : cz + dir * 5.5,
      COLORS.curb,
    );
  }
}

/** Intersection pad, crosswalk stripes, and white stop bars. */
export function buildIntersection(
  parent: THREE.Object3D,
  cx: number,
  cz: number,
  neighbors: { dx: number; dz: number }[],
) {
  box(parent, 12.1, 0.1, 12.1, cx, 0.018, cz, COLORS.asphalt);
  for (const { dx, dz } of neighbors) {
    for (let k = -4.5; k <= 4.5; k += 1.5) {
      box(
        parent,
        dx ? 1.8 : 0.7,
        0.021,
        dx ? 0.7 : 1.8,
        cx + dx * 7 + (dz ? k : 0),
        0.081,
        cz + dz * 7 + (dx ? k : 0),
        COLORS.marking,
      );
    }
    box(
      parent,
      dx ? 0.22 : 5,
      0.025,
      dx ? 5 : 0.22,
      cx + dx * 10 + (dz ? dz * 3 : 0),
      0.083,
      cz + dz * 10 + (dx ? -dx * 3 : 0),
      COLORS.marking,
    );
  }
}

function stopLabel(text: string, bg: string, fg: string) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = fg;
  ctx.font = "bold 72px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Procedural stop sign — JevPilot scene.js stop_sign branch. */
export function createStopSign(parent: THREE.Object3D, x: number, z: number, approach: number, height = 2.8) {
  const group = new THREE.Group();
  group.position.set(x, 0, z);
  group.rotation.y = -approach;
  cyl(group, 0.08, height, 0, 0, 0, COLORS.pole);
  const sign = new THREE.Mesh(
    new THREE.CylinderGeometry(0.65, 0.65, 0.1, 8),
    material("#c4715b"),
  );
  sign.rotation.x = Math.PI / 2;
  sign.position.set(0, 2.45, 0);
  group.add(sign);
  const text = new THREE.Mesh(
    new THREE.PlaneGeometry(0.95, 0.43),
    new THREE.MeshBasicMaterial({
      map: stopLabel("STOP", "#c4715b", "#fff4df"),
      side: THREE.DoubleSide,
    }),
  );
  text.position.set(0, 2.45, 0.065);
  group.add(text);
  parent.add(group);
  return group;
}

export function createTrafficLight(
  parent: THREE.Object3D,
  x: number,
  z: number,
  approach: number,
  offset = 0,
  height = 4.8,
): TrafficLightLamp[] {
  const group = new THREE.Group();
  group.position.set(x, 0, z);
  group.rotation.y = -approach;
  cyl(group, 0.08, height, 0, 0, 0, COLORS.pole);
  box(group, 0.65, 1.65, 0.38, 0, 4.2, 0, COLORS.housing);
  const lamps: TrafficLightLamp[] = [];
  for (let i = 0; i < 3; i++) {
    const lamp = new THREE.Mesh(
      new THREE.SphereGeometry(0.17, 10, 8),
      new THREE.MeshStandardMaterial({
        color: COLORS.lampOff,
        emissive: "#000000",
      }),
    );
    lamp.position.set(0, 4.73 - i * 0.5, 0.22);
    group.add(lamp);
    lamps.push({ mesh: lamp, index: i, approach, offset });
  }
  parent.add(group);
  return lamps;
}

/** Place signal heads at each approach corner (JevPilot world.js placement). */
export function placeIntersectionSignals(
  parent: THREE.Object3D,
  node: { x: number; z: number; offset: number },
  neighborPoints: { x: number; z: number }[],
): TrafficLightLamp[] {
  const lamps: TrafficLightLamp[] = [];
  for (const other of neighborPoints) {
    const h = heading(other, node);
    const p = move(move(node, h, -9), h + Math.PI / 2, 6.9);
    lamps.push(...createTrafficLight(parent, p.x, p.z, h, node.offset));
  }
  return lamps;
}

export function updateTrafficLights(lamps: TrafficLightLamp[], timeSeconds: number) {
  const node = { offset: 0 };
  for (const l of lamps) {
    node.offset = l.offset;
    const c = signalState(node, timeSeconds, l.approach).color;
    const on = (["red", "amber", "green"] as const)[l.index] === c;
    const mat = l.mesh.material as THREE.MeshStandardMaterial;
    mat.color.set(on ? COLORS.lampOn[l.index] : COLORS.lampDim);
    mat.emissive.set(on ? COLORS.lampEmissive[l.index] : "#000000");
    mat.emissiveIntensity = on ? 0.9 : 0;
  }
}

export type TownRoadLayout = {
  mainRoad: { length: number; cx: number; cz: number };
  crossRoad: { length: number; cx: number; cz: number };
  intersection: { cx: number; cz: number; offset?: number };
};

/** Build playback town road network — main + cross + intersection signals. */
export function buildTownRoadNetwork(
  parent: THREE.Object3D,
  layout: TownRoadLayout,
): TrafficLightLamp[] {
  const { mainRoad, crossRoad, intersection } = layout;
  buildStraightRoad(parent, {
    vertical: true,
    length: mainRoad.length,
    cx: mainRoad.cx,
    cz: mainRoad.cz,
    oneWay: true,
  });
  buildStraightRoad(parent, {
    vertical: false,
    length: crossRoad.length,
    cx: crossRoad.cx,
    cz: crossRoad.cz,
  });
  const neighbors = [
    { dx: 0, dz: 1 },
    { dx: 0, dz: -1 },
    { dx: 1, dz: 0 },
    { dx: -1, dz: 0 },
  ];
  buildIntersection(parent, intersection.cx, intersection.cz, neighbors);
  const node = { x: intersection.cx, z: intersection.cz, offset: intersection.offset ?? 0 };
  const neighborPoints = [
    { x: intersection.cx, z: intersection.cz + 20 },
    { x: intersection.cx, z: intersection.cz - 20 },
  ];
  const lamps = placeIntersectionSignals(parent, node, neighborPoints);
  // Cross-road approaches use stop signs (JevPilot world.js control: "stop").
  createStopSign(parent, intersection.cx + 6.9, intersection.cz - 9, -Math.PI / 2);
  createStopSign(parent, intersection.cx - 6.9, intersection.cz - 9, Math.PI / 2);
  return lamps;
}
