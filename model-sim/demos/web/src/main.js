import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js";
import { PathVectors } from "./road-vectors.js";
import { createEgoVehicle } from "./ego-vehicle.js";
import { buildTown, TrafficSystem } from "./world.js";
import { PerceptionViz } from "./perception-viz.js";

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
    this.width = options.width ?? 1280;
    this.height = options.height ?? 720;
    this.plain = options.plain ?? false;
    this.stepIndex = 0;
    this.distance = 0;
    this.speedKmh = 28;
    this.action = 0;
    this.heading = 0;
    this.activeTab = "decision";
    this._camTarget = new THREE.Vector3();
    this._camPos = new THREE.Vector3();

    if (this.plain) document.body.classList.add("plain-mode");

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(52, this.width / this.height, 0.1, 400);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setSize(this.width, this.height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(1);
    mount.appendChild(this.renderer.domElement);

    buildTown(this.scene);
    this.traffic = new TrafficSystem(this.scene);
    this.car = createEgoVehicle();
    this.scene.add(this.car);

    this.perception = new PerceptionViz(this.scene);
    this.perception.setVisible(!this.plain);
    this.vectors = new PathVectors(this.scene, document.getElementById("vector-labels"));
    this.vectors.setVisible(!this.plain);
    this._bindHud();
    this._bindTabs();

    this._camPos.set(0, 3.2, 8.5);
    this._camTarget.set(0, 1.0, -12);
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
    const weave = Math.sin(this.stepIndex / 14) * 0.4;
    const q = [
      1.3 + Math.min(this.speedKmh / 36, 1.2),
      0.42 + Math.max(weave, 0),
      0.42 + Math.max(-weave, 0),
      this.speedKmh > 54 ? 1.55 : -0.1,
    ];
    const exp = q.map((v) => Math.exp(v / 1.02));
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
            perception: payload.perception,
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

  _updateChaseCamera() {
    const back = new THREE.Vector3(0, 2.85, 8.2);
    back.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.car.rotation.y);
    const desired = this.car.position.clone().add(back);
    this._camPos.lerp(desired, 0.12);
    this.camera.position.copy(this._camPos);

    const look = new THREE.Vector3(0, 0.9, -16);
    look.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.car.rotation.y);
    this._camTarget.lerp(this.car.position.clone().add(look), 0.15);
    this.camera.lookAt(this._camTarget);
  }

  step() {
    const { action, q, probs } = this._policy();
    this.action = action;
    const steer = { 0: 0, 1: -0.032, 2: 0.032, 3: 0 }[action] ?? 0;
    const accel = action === 3 ? -3.0 : 1.2;
    this.speedKmh = Math.max(0, Math.min(65, this.speedKmh + accel * 0.085));
    this.heading += steer;
    this.distance += this.speedKmh / 3.6 / 10;
    this.car.position.z -= this.speedKmh / 3.6 / 10;
    this.car.rotation.y = this.heading;

    const actors = this.traffic.step(this.car, this.stepIndex);
    this._updateChaseCamera();

    if (!this.plain) {
      this.perception.update(this.car, [this.car, ...actors]);
      this.vectors.update(this.car, this.camera, this.width, this.height, action, probs);

      this.hud.maneuver.textContent = MANEUVER[action];
      this.hud.distance.textContent = `${Math.round(this.distance)} m ahead`;
      this.hud.remaining.textContent = `${Math.max(0, 420 - Math.round(this.distance))} m left`;
      this.hud.speed.textContent = String(Math.round(this.speedKmh));
      this.hud.state.textContent = "Local FSD playback";
      this.hud.context.textContent = `${LABELS[action]} · p=${Math.round(probs[action] * 100)}%`;
      this.hud.turnIcon.textContent = action === 1 ? "←" : action === 2 ? "→" : "↑";
      this.hud.buttons.forEach((btn, idx) => btn.classList.toggle("selected", idx === action));

      const nearby = actors.filter((a) => a.position.distanceTo(this.car.position) < 40).length;
      const payload = {
        step: this.stepIndex,
        town: "Town03",
        mode: "threejs-chase",
        maneuver: MANEUVER[action],
        action: { index: action, label: LABELS[action] },
        vehicle: { speed_kmh: +this.speedKmh.toFixed(1), speed_limit_kmh: 50 },
        perception: { tracks: nearby, lidar_points: 3200, scanner: "local-sim" },
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
  if (window.demo?.renderer) window.demo.renderer.dispose();
  if (window.demo?.perception) window.demo.perception.dispose();
  const query = new URLSearchParams(location.search);
  window.demo = new JevTownDemo({
    width: Number(params.width ?? query.get("w") ?? 1280),
    height: Number(params.height ?? query.get("h") ?? 720),
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
  for (let i = 0; i < 15; i++) window.demo.step();
  return true;
};
