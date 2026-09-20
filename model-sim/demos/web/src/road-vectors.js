import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";

const VECTOR_STEPS = 24;

function ribbonMaterial(color, alpha = 0.65) {
  return new THREE.ShaderMaterial({
    uniforms: {
      tint: { value: new THREE.Color(color) },
      alpha: { value: alpha },
    },
    vertexShader: `
      attribute float progress;
      varying float vProgress;
      void main() {
        vProgress = progress;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 tint;
      uniform float alpha;
      varying float vProgress;
      void main() {
        float fade = (1.0 - smoothstep(0.72, 1.0, vProgress)) * smoothstep(0.02, 0.1, vProgress);
        gl_FragColor = vec4(tint, alpha * fade);
      }
    `,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
}

function createRibbon(color, alpha) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array((VECTOR_STEPS + 1) * 6), 3).setUsage(
      THREE.DynamicDrawUsage,
    ),
  );
  const progress = [];
  const indices = [];
  for (let i = 0; i <= VECTOR_STEPS; i++) {
    progress.push(i / VECTOR_STEPS, i / VECTOR_STEPS);
  }
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

function updateRibbon(mesh, points, width, y) {
  const attr = mesh.geometry.attributes.position;
  for (let i = 0; i < points.length; i++) {
    const before = points[Math.max(0, i - 1)];
    const after = points[Math.min(points.length - 1, i + 1)];
    const dx = after.x - before.x;
    const dz = after.z - before.z;
    const len = Math.hypot(dx, dz) || 1;
    attr.setXYZ(
      i * 2,
      points[i].x - (dz / len) * width,
      y,
      points[i].z + (dx / len) * width,
    );
    attr.setXYZ(
      i * 2 + 1,
      points[i].x + (dz / len) * width,
      y,
      points[i].z - (dx / len) * width,
    );
  }
  attr.needsUpdate = true;
}

export class PathVectors {
  constructor(scene, labelLayer) {
    this.group = new THREE.Group();
    scene.add(this.group);
    this.labelLayer = labelLayer;
    this.candidates = [];
    this.selected = createRibbon("#007aff", 0.92);
    this.selectedGlow = createRibbon("#38bcd6", 0.22);
    this.selectedGlow.renderOrder = 3;
    this.selected.renderOrder = 5;
    this.group.add(this.selectedGlow, this.selected);
    for (let i = 0; i < 3; i++) {
      const line = createRibbon("#48a5ff", 0.55);
      const label = document.createElement("span");
      label.className = "vector-label";
      label.hidden = true;
      labelLayer.append(label);
      this.candidates.push({ line, label, color: "#48a5ff" });
      this.group.add(line);
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
    this.labelLayer.hidden = !visible;
  }

  update(car, camera, width, height, action, probabilities) {
    const configs = [
      { action: 0, color: "#48a5ff", steer: 0.0 },
      { action: 1, color: "#e6a34b", steer: -0.09 },
      { action: 2, color: "#e6a34b", steer: 0.09 },
    ];
    let selectedPoints = null;

    configs.forEach((cfg, index) => {
      const points = this._pathPoints(car, cfg.steer);
      const item = this.candidates[index];
      const selected = cfg.action === action;
      item.line.visible = !selected;
      if (selected) {
        selectedPoints = points;
      } else {
        updateRibbon(item.line, points, 0.06, 0.24);
        item.line.material.uniforms.tint.value.set(cfg.color);
        item.line.material.uniforms.alpha.value = 0.58;
      }
      const prob = Math.round((probabilities[cfg.action] ?? 0) * 100);
      const mid = points[Math.floor(points.length * 0.55)];
      const screen = new THREE.Vector3(mid.x, 0.8, mid.z).project(camera);
      item.label.hidden = screen.z > 1 || screen.z < 0;
      item.label.textContent = `${prob}%`;
      item.label.classList.toggle("selected", selected);
      item.label.style.transform = `translate(${(screen.x * 0.5 + 0.5) * width}px, ${(-screen.y * 0.5 + 0.5) * height}px) translate(-50%,-50%)`;
    });

    if (selectedPoints) {
      updateRibbon(this.selected, selectedPoints, 0.2, 0.26);
      updateRibbon(this.selectedGlow, selectedPoints, 0.42, 0.2);
    }
    this.selected.visible = this.selectedGlow.visible = !!selectedPoints;

    if (action === 3) {
      const brakePoints = this._pathPoints(car, 0);
      updateRibbon(this.selected, brakePoints, 0.16, 0.26);
      updateRibbon(this.selectedGlow, brakePoints, 0.34, 0.2);
      this.selected.material.uniforms.tint.value.set("#e86940");
      this.selectedGlow.material.uniforms.tint.value.set("#e86940");
    }
  }

  _pathPoints(car, steer) {
    const points = [];
    let x = car.position.x;
    let z = car.position.z + 2.5;
    let heading = car.rotation.y;
    for (let i = 0; i <= VECTOR_STEPS; i++) {
      heading += steer;
      z += Math.cos(heading) * 1.35;
      x += Math.sin(heading) * 1.35;
      points.push(new THREE.Vector3(x, 0.12, z));
    }
    return points;
  }
}
