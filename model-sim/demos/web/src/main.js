import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";
import { PathVectors } from "./road-vectors.js";

const MANEUVER = {
  0: "Continue straight",
  1: "Bear left",
  2: "Bear right",
  3: "Slow for hazard",
};

const LABELS = ["FORWARD", "LEFT", "RIGHT", "BRAKE"];

export class JevTownDemo {
  constructor(options = {}) {
    const mount = options.mount ?? document.getElementById("app");
    this.width = options.width ?? 960;
    this.height = options.height ?? 540;
    this.plain = options.plain ?? false;
    this.stepIndex = 0;
    this.distance = 0;
    this.speedKmh = 24;
    this.action = 0;
    this.heading = 0;
    this.activeTab = "decision";

    if (this.plain) {
      document.body.classList.add("plain-mode");
    }

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#b7c9db");
    this.scene.fog = new THREE.Fog("#b7c9db", 35, 150);

    this.camera = new THREE.PerspectiveCamera(68, this.width / this.height, 0.1, 300);
    this.camera.position.set(0, 1.25, 0.8);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setSize(this.width, this.height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(this.renderer.domElement);

    this._buildWorld();
    this.vectors = new PathVectors(this.scene, document.getElementById("vector-labels"));
    this.vectors.setVisible(!this.plain);
    this._bindHud();
    this._bindTabs();
  }

  _buildWorld() {
    this.scene.add(new THREE.HemisphereLight("#eef4fb", "#70817c", 1.1));
    const sun = new THREE.DirectionalLight("#ffffff", 0.9);
    sun.position.set(16, 36, 8);
    this.scene.add(sun);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(240, 240),
      new THREE.MeshStandardMaterial({ color: "#d8d6c9" }),
    );
    ground.rotation.x = -Math.PI / 2;
    this.scene.add(ground);

    const road = new THREE.Mesh(
      new THREE.PlaneGeometry(11, 240),
      new THREE.MeshStandardMaterial({ color: "#70817c", roughness: 0.95 }),
    );
    road.rotation.x = -Math.PI / 2;
    road.position.y = 0.03;
    this.scene.add(road);

    for (let i = 0; i < 32; i++) {
      const z = -110 + i * 8;
      for (const side of [-1, 1]) {
        const w = 3.5 + (i % 4);
        const h = 5 + (i % 6);
        const building = new THREE.Mesh(
          new THREE.BoxGeometry(w, h, 5.5),
          new THREE.MeshStandardMaterial({
            color: i % 2 ? "#8d9cab" : "#7f93a3",
            roughness: 0.85,
          }),
        );
        building.position.set(side * (7.5 + (i % 3)), h / 2, z);
        this.scene.add(building);
      }
    }

    this.car = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1.85, 0.65, 4.0),
      new THREE.MeshStandardMaterial({ color: "#f4f6f8", metalness: 0.15, roughness: 0.35 }),
    );
    body.position.y = 0.52;
    this.car.add(body);
    this.scene.add(this.car);
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

  _bindTabs() {
    document.querySelectorAll(".json-tabs button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".json-tabs button").forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        this.activeTab = button.dataset.tab;
        this._renderJson(this._lastPayload ?? {});
      });
    });
  }

  _policy() {
    const weave = Math.sin(this.stepIndex / 16) * 0.35;
    const q = [
      1.25 + Math.min(this.speedKmh / 38, 1.1),
      0.4 + Math.max(weave, 0),
      0.4 + Math.max(-weave, 0),
      this.speedKmh > 52 ? 1.5 : -0.15,
    ];
    const exp = q.map((v) => Math.exp(v / 1.05));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map((v) => v / sum);
    const action = probs.indexOf(Math.max(...probs));
    return { action, q, probs };
  }

  _renderJson(payload) {
    this._lastPayload = payload;
    const view =
      this.activeTab === "decision"
        ? {
            maneuver: payload.maneuver,
            action: payload.action,
            probabilities: payload.probabilities,
            selected_path: LABELS[payload.action?.index ?? 0],
          }
        : payload;
    const text = JSON.stringify(view, null, 2);
    this.hud.json.innerHTML = text
      .replace(/"(.*?)":/g, '<span class="json-key">"$1":</span>')
      .replace(/: "(.*?)"/g, ': <span class="json-string">"$1"</span>')
      .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
      .replace(/: (true|false|null)/g, ': <span class="json-bool">$1</span>');
  }

  step() {
    const { action, q, probs } = this._policy();
    this.action = action;
    const steer = { 0: 0, 1: -0.035, 2: 0.035, 3: 0 }[action] ?? 0;
    const accel = action === 3 ? -2.8 : 1.15;
    this.speedKmh = Math.max(0, Math.min(65, this.speedKmh + accel * 0.08));
    this.heading += steer;
    this.distance += this.speedKmh / 3.6 / 10;
    this.car.position.z -= this.speedKmh / 3.6 / 10;
    this.car.rotation.y = this.heading;

    // Driver ego camera — hood height, looking down the road.
    this.camera.position.set(
      this.car.position.x + Math.sin(this.heading) * 0.15,
      1.22,
      this.car.position.z + 0.85,
    );
    this.camera.lookAt(
      this.car.position.x + Math.sin(this.heading) * 2,
      0.95,
      this.car.position.z + 16,
    );

    if (!this.plain) {
      this.vectors.update(this.car, this.camera, this.width, this.height, action, probs);
      this.hud.maneuver.textContent = MANEUVER[action];
      this.hud.distance.textContent = `${Math.round(this.distance)} m ahead`;
      this.hud.remaining.textContent = `${Math.max(0, 420 - Math.round(this.distance))} m left`;
      this.hud.speed.textContent = String(Math.round(this.speedKmh));
      this.hud.state.textContent = "Local playback";
      this.hud.context.textContent = `${LABELS[action]} · p=${Math.round(probs[action] * 100)}%`;
      this.hud.turnIcon.textContent = action === 1 ? "←" : action === 2 ? "→" : "↑";
      this.hud.buttons.forEach((btn, idx) => btn.classList.toggle("selected", idx === action));

      const payload = {
        step: this.stepIndex,
        town: "Town03",
        mode: "threejs",
        maneuver: MANEUVER[action],
        action: { index: action, label: LABELS[action] },
        vehicle: { speed_kmh: +this.speedKmh.toFixed(1), speed_limit_kmh: 50 },
        q_values: Object.fromEntries(q.map((v, i) => [String(i), +v.toFixed(3)])),
        probabilities: Object.fromEntries(probs.map((v, i) => [String(i), +v.toFixed(3)])),
      };
      this._renderJson(payload);
    }

    this.stepIndex += 1;
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}

window.JevTownDemo = JevTownDemo;

window.createDemo = (params = {}) => {
  if (window.demo) {
    window.demo.renderer.dispose();
  }
  const query = new URLSearchParams(location.search);
  window.demo = new JevTownDemo({
    width: Number(params.width ?? query.get("w") ?? 960),
    height: Number(params.height ?? query.get("h") ?? 540),
    plain: params.plain ?? query.get("plain") === "1",
  });
  return window.demo;
};

window.stepSimulation = () => {
  if (!window.demo) window.createDemo();
  window.demo.step();
  return true;
};

window.renderPlainFrame = () => {
  window.createDemo({ plain: true });
  for (let i = 0; i < 12; i++) window.demo.step();
  return true;
};
