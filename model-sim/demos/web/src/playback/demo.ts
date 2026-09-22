import * as THREE from "three";
import { PlaybackPathVectors } from "../vendor/jevpilot/road-vectors";
import { loadHeroCar, updateHeroWheels } from "../vendor/jevpilot/model-assets";
import { renderProfile } from "../vendor/jevpilot/render-profile";
import { createEgoVehicle, loadTrafficFleet } from "./ego";
import { updateTrafficLights } from "../vendor/jevpilot/jevpilot-road";
import { buildTown, EGO_LANE_X, TrafficSystem } from "./world";
import { PerceptionViz } from "./perception-viz";
import { SensorAdapter } from "./sensor-adapter";

const MANEUVER: Record<number, string> = {
  0: "Continue straight",
  1: "Bear left",
  2: "Bear right",
  3: "Slow for hazard",
};

const LABELS = ["FORWARD", "LEFT", "RIGHT", "BRAKE"];

export interface DemoOptions {
  mount?: HTMLElement;
  width?: number;
  height?: number;
  plain?: boolean;
  procedural?: boolean;
  proceduralTraffic?: boolean;
}

export class JevTownDemo {
  width: number;
  height: number;
  plain: boolean;
  proceduralTraffic: boolean;
  captureMode: boolean;
  stepIndex = 0;
  distance = 0;
  speedKmh = 40;
  action = 0;
  heading = 0;
  activeTab = "decision";
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  traffic: TrafficSystem;
  car: THREE.Group;
  perception: PerceptionViz;
  sensor: SensorAdapter;
  vectors: PlaybackPathVectors;
  ready: Promise<void>;
  private _camTarget = new THREE.Vector3();
  private _camPos = new THREE.Vector3();
  private _heroCar: THREE.Group | null = null;
  private _lastPayload: Record<string, unknown> = {};
  private _lastFrameMs = performance.now();
  /** Last TrafficSystem.leadGap (scripted playback input — not a learned policy). */
  private _leadGap = Infinity;
  private hud!: {
    maneuver: HTMLElement;
    distance: HTMLElement;
    remaining: HTMLElement;
    speed: HTMLElement;
    state: HTMLElement;
    context: HTMLElement;
    json: HTMLElement;
    turnIcon: HTMLElement;
    buttons: HTMLButtonElement[];
  };

  constructor(options: DemoOptions = {}) {
    const mount = options.mount ?? document.getElementById("app")!;
    this.width = options.width ?? 1280;
    this.height = options.height ?? 720;
    this.plain = options.plain ?? false;
    this.proceduralTraffic =
      options.proceduralTraffic ??
      new URLSearchParams(location.search).get("proceduralTraffic") === "1";
    this.captureMode =
      new URLSearchParams(location.search).get("capture") === "1";

    if (this.plain) document.body.classList.add("plain-mode");
    if (this.captureMode) document.body.classList.add("capture-mode");

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(52, this.width / this.height, 0.1, 400);
    this.renderer = new THREE.WebGLRenderer({
      antialias: renderProfile.antialias,
      preserveDrawingBuffer: true,
      logarithmicDepthBuffer: true,
    });
    this.renderer.setSize(this.width, this.height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(Math.min(renderProfile.pixelRatio, 2));
    mount.appendChild(this.renderer.domElement);

    buildTown(this.scene);
    this.traffic = new TrafficSystem(this.scene);
    this.car = createEgoVehicle();
    this.car.position.set(EGO_LANE_X, 0, 0);
    this.scene.add(this.car);

    this.perception = new PerceptionViz(this.scene);
    this.sensor = new SensorAdapter();
    this.perception.setVisible(!this.plain);
    this.vectors = new PlaybackPathVectors(this.scene, document.getElementById("vector-labels")!);
    this.vectors.setVisible(!this.plain);
    this._bindHud();
    this._bindTabs();

    this._camPos.set(0, 3.2, 8.5);
    this._camTarget.set(0, 1.0, -12);

    const playerGroup = this.car;
    // Ego is Tesla Model Y only. `procedural=1` is a headless fallback — do not
    // skip the GLB just because NPC traffic is procedural.
    const skipHeroModel =
      options.procedural === true || new URLSearchParams(location.search).get("procedural") === "1";
    const heroReady = skipHeroModel
      ? Promise.resolve()
      : loadHeroCar()
          .then((model) => {
            if (this.car !== playerGroup) {
              model.traverse((mesh) => (mesh as THREE.Mesh).geometry?.dispose());
              return;
            }
            playerGroup.traverse((mesh) => (mesh as THREE.Mesh).geometry?.dispose());
            playerGroup.clear();
            playerGroup.add(model);
            this._heroCar = model;
            playerGroup.userData.sourcedModel = true;
            playerGroup.userData.eyeHeight = model.userData.eyeHeight;
            playerGroup.userData.eyeForward = model.userData.eyeForward;
          })
          .catch((error) => console.warn("Model Y unavailable, keeping procedural ego", error));

    const trafficReady = this.proceduralTraffic
      ? Promise.resolve()
      : loadTrafficFleet().catch((e) => {
          console.warn("Traffic GLB fleet unavailable, using procedural NPCs", e);
          return null;
        });

    this.ready = Promise.all([heroReady, trafficReady]).then(() => {
      this.traffic.spawnInitial(this.proceduralTraffic);
    });
  }

  private _bindHud() {
    this.hud = {
      maneuver: document.getElementById("next-maneuver")!,
      distance: document.getElementById("turn-distance")!,
      remaining: document.getElementById("remaining")!,
      speed: document.getElementById("speed")!,
      state: document.getElementById("pilot-state")!,
      context: document.getElementById("context-message")!,
      json: document.getElementById("json-content")!,
      turnIcon: document.getElementById("turn-icon")!,
      buttons: [...document.querySelectorAll<HTMLButtonElement>(".candidate-button")],
    };
  }

  private _bindTabs() {
    document.querySelectorAll<HTMLButtonElement>(".json-tabs button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".json-tabs button").forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        this.activeTab = button.dataset.tab ?? "decision";
        this._renderJson(this._lastPayload);
      });
    });
  }

  private _policy() {
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

  private _renderJson(payload: Record<string, unknown>) {
    this._lastPayload = payload;
    const view =
      this.activeTab === "decision"
        ? {
            maneuver: payload.maneuver,
            action: payload.action,
            probabilities: payload.probabilities,
            perception: payload.perception,
            selected_path: LABELS[(payload.action as { index?: number })?.index ?? 0],
          }
        : payload;
    const text = JSON.stringify(view, null, 2);
    (this.hud.json as HTMLElement).innerHTML = text
      .replace(/"(.*?)":/g, '<span class="json-key">"$1":</span>')
      .replace(/: "(.*?)"/g, ': <span class="json-string">"$1"</span>')
      .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
      .replace(/: (true|false|null)/g, ': <span class="json-bool">$1</span>');
  }

  private _updateChaseCamera() {
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
    const now = performance.now();
    const dt = now - this._lastFrameMs;
    this._lastFrameMs = now;

    const { action, q, probs } = this._policy();
    this.action = action;
    const steer = ({ 0: 0, 1: -0.032, 2: 0.032, 3: 0 } as Record<number, number>)[action] ?? 0;
    const traffic = this.traffic.step(this.car, this.stepIndex);
    const actors = traffic.actors;
    const leadGap = traffic.leadGap ?? Infinity;
    this._leadGap = leadGap;

    // Scripted playback from TrafficSystem.leadGap — NOT a learned / sensor closed-loop policy.
    // Same #113-class curve in captureMode (no weaker floor / no skipped BRAKE HUD).
    let accel = action === 3 ? -3.0 : 1.2;
    if (leadGap < 24) {
      accel = Math.min(accel, -2.0);
      this.speedKmh = Math.min(this.speedKmh, Math.max(8, (leadGap - 4) * 2.6));
    }
    if (leadGap < 14) {
      accel = -4.5;
      this.action = 3;
    }
    // Hold short of the lead so ego cannot pierce once gap drops below the 8 km/h floor.
    if (leadGap < 8) {
      accel = -8;
      this.speedKmh = Math.min(this.speedKmh, Math.max(0, (leadGap - 6) * 4));
      this.action = 3;
    }

    const delta = this.speedKmh / 3.6 / 10;
    this.speedKmh = Math.max(0, Math.min(65, this.speedKmh + accel * 0.085));
    this.heading += steer;
    this.distance += delta;
    this.car.position.z -= delta;
    this.car.rotation.y = this.heading;

    if (this._heroCar) {
      updateHeroWheels(this._heroCar, delta, steer * 8);
    }

    const lights = this.scene.userData.trafficLights;
    if (lights?.length) updateTrafficLights(lights, now / 1000);

    this._updateChaseCamera();

    if (!this.plain) {
      const sensorFrame = this.sensor.synthesize(this.car, [this.car, ...actors]);
      this.perception.update(this.car, sensorFrame);
      this.vectors.update(this.car, this.camera, this.width, this.height, dt, this.action, probs);

      this.hud.maneuver.textContent = MANEUVER[this.action];
      this.hud.distance.textContent = `${Math.round(this.distance)} m ahead`;
      this.hud.remaining.textContent = `${Math.max(0, 420 - Math.round(this.distance))} m left`;
      this.hud.speed.textContent = String(Math.round(this.speedKmh));
      this.hud.state.textContent = "Scripted playback";
      this.hud.context.textContent = `${LABELS[this.action]} · leadGap (not a model)`;
      this.hud.turnIcon.textContent = this.action === 1 ? "←" : this.action === 2 ? "→" : "↑";
      document.body.classList.toggle("is-braking", this.action === 3);
      (this.hud.buttons as HTMLButtonElement[]).forEach((btn, idx) =>
        btn.classList.toggle("selected", idx === this.action),
      );

      this._renderJson({
        step: this.stepIndex,
        town: "Town03",
        mode: "threejs-chase",
        maneuver: MANEUVER[this.action],
        action: { index: this.action, label: LABELS[this.action] },
        vehicle: { speed_kmh: +this.speedKmh.toFixed(1), speed_limit_kmh: 50 },
        perception: this.sensor.frameStats(sensorFrame),
        honesty: {
          control: "scripted-leadGap",
          sensor: "synthetic-adapter",
          closed_loop: false,
          later: "sensor→control, loaded model decisions, closed-loop RL",
        },
        q_values: Object.fromEntries(q.map((v, i) => [String(i), +v.toFixed(3)])),
        probabilities: Object.fromEntries(probs.map((v, i) => [String(i), +v.toFixed(3)])),
      });
    }

    this.stepIndex += 1;
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /** Headless capture telemetry. */
  motionSample() {
    const perception = this._lastPayload.perception as { tracks?: unknown[] } | undefined;
    return {
      egoZ: this.car.position.z,
      step: this.stepIndex,
      tracks: perception?.tracks?.length ?? 0,
      speedKmh: this.speedKmh,
      action: this.action,
      leadGap: this._leadGap,
      egoModel: this._heroCar?.name ?? "procedural-placeholder",
    };
  }
}
