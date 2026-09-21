/**
 * Vendored ribbon shaders from standardagents/jevpilot — src/road-vectors.js
 * Playback adapter: drives ribbons from local poses/paths/probs (no Jev decision API).
 */
import * as THREE from "three";

const VECTOR_STEPS = 24;

type CarPose = { x: number; z: number; heading: number; width: number; depth: number };

function ribbonMaterial(color: string, alpha = 0.65) {
  return new THREE.ShaderMaterial({
    uniforms: {
      tint: { value: new THREE.Color(color) },
      alpha: { value: alpha },
      time: { value: 0 },
      pulse: { value: 0 },
      ego: { value: new THREE.Vector3() },
      body: { value: new THREE.Vector2() },
    },
    vertexShader: `
      attribute float progress;
      varying float vProgress;
      varying vec2 vWorld;
      void main() {
        vProgress = progress;
        vWorld = (modelMatrix * vec4(position, 1.0)).xz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 tint;
      uniform float alpha;
      uniform float time;
      uniform float pulse;
      uniform vec3 ego;
      uniform vec2 body;
      varying float vProgress;
      varying vec2 vWorld;
      void main() {
        vec2 delta = vWorld - ego.xy;
        float right = dot(delta, vec2(cos(ego.z), sin(ego.z)));
        float ahead = dot(delta, vec2(sin(ego.z), -cos(ego.z)));
        if (abs(right) < body.x && abs(ahead) < body.y) discard;
        float fade = (1.0 - smoothstep(0.72, 1.0, vProgress)) * smoothstep(0.015, 0.08, vProgress);
        float scan = 1.0 - pulse * (0.5 + 0.5 * sin(vProgress * 18.0 - time * 6.0));
        gl_FragColor = vec4(tint, alpha * fade * scan);
      }
    `,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
}

function createRibbon(color: string, alpha: number) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array((VECTOR_STEPS + 1) * 6), 3).setUsage(THREE.DynamicDrawUsage),
  );
  const progress: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= VECTOR_STEPS; i++) progress.push(i / VECTOR_STEPS, i / VECTOR_STEPS);
  geometry.setAttribute("progress", new THREE.Float32BufferAttribute(progress, 1));
  for (let i = 0; i < VECTOR_STEPS; i++) {
    const n = i * 2;
    indices.push(n, n + 1, n + 2, n + 1, n + 3, n + 2);
  }
  geometry.setIndex(indices);
  const mesh = new THREE.Mesh(geometry, ribbonMaterial(color, alpha));
  mesh.frustumCulled = false;
  mesh.renderOrder = 4;
  return mesh;
}

function updateRibbon(mesh: THREE.Mesh, points: THREE.Vector3[], width: number, y: number) {
  const attr = mesh.geometry.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < points.length; i++) {
    const before = points[Math.max(0, i - 1)];
    const after = points[Math.min(points.length - 1, i + 1)];
    const dx = after.x - before.x;
    const dz = after.z - before.z;
    const len = Math.hypot(dx, dz) || 1;
    attr.setXYZ(i * 2, points[i].x - (dz / len) * width, y, points[i].z + (dx / len) * width);
    attr.setXYZ(i * 2 + 1, points[i].x + (dz / len) * width, y, points[i].z - (dx / len) * width);
  }
  attr.needsUpdate = true;
}

function pathPoints(car: THREE.Object3D, steer: number): THREE.Vector3[] {
  const points: THREE.Vector3[] = [];
  let x = car.position.x;
  let z = car.position.z - 2.8;
  let heading = car.rotation.y;
  for (let i = 0; i <= VECTOR_STEPS; i++) {
    heading += steer;
    z -= Math.cos(heading) * 1.35;
    x += Math.sin(heading) * 1.35;
    points.push(new THREE.Vector3(x, 0.12, z));
  }
  return points;
}

export class PlaybackPathVectors {
  private group = new THREE.Group();
  private selected = createRibbon("#007aff", 0.9);
  private selectedGlow = createRibbon("#38bcd6", 0.15);
  private candidates: Array<{ line: THREE.Mesh; label: HTMLSpanElement; color: string }> = [];
  private time = 0;

  constructor(scene: THREE.Scene, labelLayer: HTMLElement) {
    scene.add(this.group);
    this.selectedGlow.renderOrder = 3;
    this.selected.renderOrder = 5;
    this.group.add(this.selectedGlow, this.selected);
    const configs = [
      { color: "#48a5ff", steer: 0 },
      { color: "#e6a34b", steer: -0.09 },
      { color: "#e6a34b", steer: 0.09 },
    ];
    for (const cfg of configs) {
      const line = createRibbon(cfg.color, 0.65);
      const label = document.createElement("span");
      label.className = "vector-label";
      label.hidden = true;
      labelLayer.append(label);
      this.candidates.push({ line, label, color: cfg.color });
      this.group.add(line);
    }
  }

  setVisible(visible: boolean) {
    this.group.visible = visible;
    for (const c of this.candidates) c.label.hidden = !visible;
  }

  update(
    car: THREE.Object3D,
    camera: THREE.Camera,
    width: number,
    height: number,
    dt: number,
    action: number,
    probabilities: number[],
  ) {
    this.time += dt;
    const pose: CarPose = {
      x: car.position.x,
      z: car.position.z,
      heading: car.rotation.y,
      width: (car.userData.width as number) ?? 1.9,
      depth: (car.userData.depth as number) ?? 4.16,
    };
    for (const mesh of this.group.children) {
      const mat = (mesh as THREE.Mesh).material as THREE.ShaderMaterial;
      mat.uniforms.ego.value.set(pose.x, pose.z, pose.heading);
      mat.uniforms.body.value.set(pose.width / 2 + 0.12, pose.depth / 2 + 0.15);
      mat.uniforms.time.value = this.time * 0.001;
      mat.uniforms.pulse.value = 0.35;
    }

    const steers = [0, -0.09, 0.09];
    let selectedPoints: THREE.Vector3[] | null = null;

    this.candidates.forEach((item, index) => {
      const points = pathPoints(car, steers[index]);
      const selected = index === action && action !== 3;
      item.line.visible = !selected;
      if (selected) selectedPoints = points;
      else updateRibbon(item.line, points, 0.055, 0.22);

      const prob = Math.round((probabilities[index] ?? 0) * 100);
      const mid = points[Math.floor(points.length * 0.55)];
      const screen = new THREE.Vector3(mid.x, 0.8, mid.z).project(camera);
      item.label.hidden = screen.z > 1 || screen.z < 0;
      item.label.textContent = `${prob}%`;
      item.label.classList.toggle("selected", selected);
      item.label.style.transform = `translate(${(screen.x * 0.5 + 0.5) * width}px, ${(-screen.y * 0.5 + 0.5) * height}px) translate(-50%,-50%)`;
    });

    if (selectedPoints) {
      updateRibbon(this.selected, selectedPoints, 0.2, 0.25);
      updateRibbon(this.selectedGlow, selectedPoints, 0.5, 0.195);
      (this.selected.material as THREE.ShaderMaterial).uniforms.tint.value.set("#007aff");
      (this.selectedGlow.material as THREE.ShaderMaterial).uniforms.tint.value.set("#38bcd6");
    }
    this.selected.visible = this.selectedGlow.visible = !!selectedPoints;

    if (action === 3) {
      const brakePoints = pathPoints(car, 0);
      updateRibbon(this.selected, brakePoints, 0.16, 0.26);
      updateRibbon(this.selectedGlow, brakePoints, 0.34, 0.2);
      (this.selected.material as THREE.ShaderMaterial).uniforms.tint.value.set("#e86940");
      (this.selectedGlow.material as THREE.ShaderMaterial).uniforms.tint.value.set("#e86940");
      this.selected.visible = this.selectedGlow.visible = true;
    }
  }
}
