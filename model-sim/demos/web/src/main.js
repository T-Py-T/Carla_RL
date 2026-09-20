import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";

const PATH_COLORS = {
  forward: 0x48a5ff,
  selected: 0x007aff,
  glow: 0x38bcd6,
  amber: 0xe6a34b,
  brake: 0xe86940,
};

const MANEUVER = {
  0: "Continue straight",
  1: "Bear left",
  2: "Bear right",
  3: "Slow for hazard",
};

export class JevTownDemo {
  constructor({ width = 960, height = 540, mount = document.getElementById("app") } = {}) {
    this.width = width;
    this.height = height;
    this.stepIndex = 0;
    this.distance = 0;
    this.speedKmh = 24;
    this.action = 0;
    this.heading = 0;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#b7c9db");
    this.scene.fog = new THREE.Fog("#b7c9db", 40, 180);

    this.camera = new THREE.PerspectiveCamera(72, width / height, 0.1, 400);
    this.camera.position.set(0, 1.35, -2.1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setSize(width, height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(this.renderer.domElement);

    this._buildWorld();
    this._buildPaths();
    this._bindHud();
  }

  _buildWorld() {
    const light = new THREE.HemisphereLight("#eef4fb", "#70817c", 1.05);
    this.scene.add(light);
    const sun = new THREE.DirectionalLight("#ffffff", 0.85);
    sun.position.set(20, 40, 10);
    this.scene.add(sun);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(260, 260),
      new THREE.MeshStandardMaterial({ color: "#d8d6c9" }),
    );
    ground.rotation.x = -Math.PI / 2;
    this.scene.add(ground);

    const road = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 260),
      new THREE.MeshStandardMaterial({ color: "#70817c" }),
    );
    road.rotation.x = -Math.PI / 2;
    road.position.y = 0.02;
    this.scene.add(road);

    for (let i = 0; i < 28; i++) {
      const z = -120 + i * 9;
      const side = i % 2 === 0 ? -1 : 1;
      const w = 4 + (i % 5);
      const h = 6 + (i % 7);
      const building = new THREE.Mesh(
        new THREE.BoxGeometry(w, h, 6),
        new THREE.MeshStandardMaterial({ color: i % 3 ? "#8d9cab" : "#7f93a3" }),
      );
      building.position.set(side * (8 + (i % 4)), h / 2, z);
      this.scene.add(building);
    }

    this.car = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1.8, 0.7, 3.8),
      new THREE.MeshStandardMaterial({ color: "#f4f6f8", metalness: 0.2, roughness: 0.35 }),
    );
    body.position.y = 0.55;
    this.car.add(body);
    this.scene.add(this.car);
  }

  _buildPaths() {
    this.pathGroup = new THREE.Group();
    this.scene.add(this.pathGroup);
    this.pathMeshes = [];
    for (let action = 0; action < 4; action++) {
      const geometry = new THREE.BufferGeometry();
      const material = new THREE.LineBasicMaterial({
        color: action === 0 ? PATH_COLORS.forward : action === 3 ? PATH_COLORS.brake : PATH_COLORS.amber,
        transparent: true,
        opacity: action === 0 ? 0.95 : 0.65,
        linewidth: 1,
      });
      const line = new THREE.Line(geometry, material);
      this.pathGroup.add(line);
      this.pathMeshes.push(line);
    }
    this.glow = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: PATH_COLORS.glow, transparent: true, opacity: 0.25 }),
    );
    this.pathGroup.add(this.glow);
  }

  _pathPoints(action) {
    const points = [];
    let x = 0;
    let z = 2;
    let heading = 0;
    const steer = { 0: 0, 1: -0.08, 2: 0.08, 3: 0 }[action] ?? 0;
    for (let i = 0; i < 28; i++) {
      heading += steer;
      z += Math.cos(heading) * 1.4;
      x += Math.sin(heading) * 1.4;
      points.push(new THREE.Vector3(x, 0.08, z));
    }
    return points;
  }

  _updatePaths(selected) {
    this.pathMeshes.forEach((line, action) => {
      const points = this._pathPoints(action);
      line.geometry.setFromPoints(points);
      line.material.opacity = action === selected ? 0.98 : 0.55;
      line.material.color.setHex(
        action === selected
          ? PATH_COLORS.selected
          : action === 3
            ? PATH_COLORS.brake
            : action === 0
              ? PATH_COLORS.forward
              : PATH_COLORS.amber,
      );
    });
    const glowPoints = this._pathPoints(selected);
    this.glow.geometry.setFromPoints(glowPoints);
  }

  _bindHud() {
    this.hud = {
      maneuver: document.getElementById("next-maneuver"),
      distance: document.getElementById("turn-distance"),
      remaining: document.getElementById("remaining"),
      speed: document.getElementById("speed"),
      state: document.getElementById("pilot-state"),
      context: document.getElementById("context-message"),
      json: document.getElementById("json-content"),
      turnIcon: document.getElementById("turn-icon"),
      buttons: [...document.querySelectorAll(".candidate-button")],
    };
  }

  _policy() {
    const weave = Math.sin(this.stepIndex / 18) * 0.25;
    const q = [
      1.2 + Math.min(this.speedKmh / 40, 1),
      0.35 + Math.max(weave, 0),
      0.35 + Math.max(-weave, 0),
      this.speedKmh > 55 ? 1.4 : -0.2,
    ];
    const action = q.indexOf(Math.max(...q));
    const exp = q.map((v) => Math.exp(v));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map((v) => v / sum);
    return { action, q, probs };
  }

  step() {
    const { action, q, probs } = this._policy();
    this.action = action;
    const steer = { 0: 0, 1: -0.03, 2: 0.03, 3: 0 }[action] ?? 0;
    const accel = action === 3 ? -2.5 : 1.1;
    this.speedKmh = Math.max(0, Math.min(65, this.speedKmh + accel * 0.08));
    this.heading += steer;
    this.distance += this.speedKmh / 3.6 / 10;
    this.car.position.z -= this.speedKmh / 3.6 / 10;
    this.car.rotation.y = this.heading;
    this.camera.position.x = Math.sin(this.heading) * 0.35;
    this.camera.position.z = this.car.position.z - 2.1;
    this.camera.lookAt(this.car.position.x, 1.0, this.car.position.z + 12);
    this._updatePaths(action);

    const labels = ["FORWARD", "LEFT", "RIGHT", "BRAKE"];
    this.hud.maneuver.textContent = MANEUVER[action];
    this.hud.distance.textContent = `${Math.round(this.distance)} m ahead`;
    this.hud.remaining.textContent = `${Math.max(0, 420 - Math.round(this.distance))} m left`;
    this.hud.speed.textContent = String(Math.round(this.speedKmh));
    this.hud.state.textContent = "Local playback";
    this.hud.context.textContent = `${labels[action]} · p=${Math.round(probs[action] * 100)}%`;
    this.hud.turnIcon.textContent = action === 1 ? "←" : action === 2 ? "→" : "↑";
    this.hud.buttons.forEach((btn, idx) => btn.classList.toggle("selected", idx === action));

    const payload = {
      step: this.stepIndex,
      town: "Town03",
      mode: "threejs",
      maneuver: MANEUVER[action],
      action: { index: action, label: labels[action] },
      vehicle: { speed_kmh: +this.speedKmh.toFixed(1), speed_limit_kmh: 50 },
      q_values: Object.fromEntries(q.map((v, i) => [String(i), +v.toFixed(3)])),
      probabilities: Object.fromEntries(probs.map((v, i) => [String(i), +v.toFixed(3)])),
    };
    this.hud.json.innerHTML = JSON.stringify(payload, null, 2)
      .replace(/"(.*?)":/g, '<span class="json-key">"$1":</span>')
      .replace(/: "(.*?)"/g, ': <span class="json-string">"$1"</span>')
      .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>');

    this.stepIndex += 1;
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}

window.JevTownDemo = JevTownDemo;

window.stepSimulation = () => {
  if (!window.demo) {
    window.demo = new JevTownDemo({
      width: Number(new URLSearchParams(location.search).get("w") || 960),
      height: Number(new URLSearchParams(location.search).get("h") || 540),
    });
  }
  window.demo.step();
  return true;
};
