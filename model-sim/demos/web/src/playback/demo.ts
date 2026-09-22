import * as THREE from "three";
import { PlaybackPathVectors } from "../vendor/jevpilot/road-vectors";
import { loadHeroCar, updateHeroWheels } from "../vendor/jevpilot/model-assets";
import { renderProfile } from "../vendor/jevpilot/render-profile";
import { auditOpaqueMaterials, createOpaqueModelY, loadTrafficFleet } from "./ego";
import { updateTrafficLights } from "../vendor/jevpilot/jevpilot-road";
import {
  buildTown,
  BUMPER_CLEAR_M,
  EGO_HALF_LENGTH,
  EGO_LANE_X,
  NPC_HALF_LENGTH,
  TrafficSystem,
} from "./world";
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
  /** Last TrafficSystem bumper leadGap (scripted playback input — not a learned policy). */
  private _leadGap = Infinity;
  private _leadZ: number | null = null;
  private _overlap = false;
  private _bumperGap = Infinity;
  private _transparentMeshes = 0;
  private _egoMeshCount = 0;
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
      // Log depth + MeshPhysical glass reads as a whole-body ghost in SwiftShader.
      logarithmicDepthBuffer: !this.captureMode,
    });
    this.renderer.setSize(this.width, this.height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(Math.min(renderProfile.pixelRatio, 2));
    mount.appendChild(this.renderer.domElement);

    buildTown(this.scene);
    this.traffic = new TrafficSystem(this.scene);
    this.car = new THREE.Group();
    this.car.position.set(EGO_LANE_X, 0, 0);
    this.scene.add(this.car);
    this._installOpaqueEgo();

    this.perception = new PerceptionViz(this.scene);
    this.sensor = new SensorAdapter();
    this.perception.setVisible(!this.plain);
    this.vectors = new PlaybackPathVectors(this.scene, document.getElementById("vector-labels")!);
    // Capture hides path ribbons — they read as a second ghost mesh through the hull.
    this.vectors.setVisible(!this.plain && !this.captureMode);
    this._bindHud();
    this._bindTabs();
    if (this.captureMode) this._applyCaptureHud();

    if (this.captureMode) {
      this._camPos.set(EGO_LANE_X + 3.6, 2.85, 8.8);
      this._camTarget.set(EGO_LANE_X - 0.4, 0.7, -10);
    } else {
      this._camPos.set(EGO_LANE_X, 2.55, 7.4);
      this._camTarget.set(EGO_LANE_X, 0.95, -14);
    }

    const playerGroup = this.car;
    // Capture never mounts the Draco GLB / undraco ghost — that swap is what
    // turned the body translucent ~1s into the clip. Interactive browsers may
    // still replace the opaque hull with a hardened Model Y GLB.
    const skipHeroModel =
      this.captureMode ||
      options.procedural === true ||
      new URLSearchParams(location.search).get("procedural") === "1";
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
          .catch((error) => console.warn("Model Y GLB unavailable, keeping opaque hull", error));

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

  private _installOpaqueEgo() {
    const model = createOpaqueModelY();
    this.car.clear();
    this.car.add(model);
    this._heroCar = model;
    this.car.userData.sourcedModel = true;
    this.car.userData.eyeHeight = model.userData.eyeHeight;
    this.car.userData.eyeForward = model.userData.eyeForward;
    this.car.userData.depth = model.userData.depth;
    this.car.userData.width = model.userData.width;
    const audit = auditOpaqueMaterials(this.car);
    this._transparentMeshes = audit.transparentMeshes;
    this._egoMeshCount = audit.meshCount;
  }

  /** Product labels only — no “scripted / synthetic / not a model” on the clip. */
  private _applyCaptureHud() {
    const brand = document.querySelector<HTMLElement>(".topbar strong");
    const sub = document.querySelector<HTMLElement>(".topbar span");
    if (brand) brand.textContent = "Carla RL · FSD";
    if (sub) sub.textContent = "Model Y";
    if (this.hud.state) this.hud.state.textContent = "FSD";
    if (this.hud.context) this.hud.context.textContent = "Model Y";
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
    // Capture uses a 3/4 rear-right chase so the bumper-to-lead gap is on
    // screen. A dead-behind cam stacked the lead box through the ego hull.
    const back = this.captureMode
      ? new THREE.Vector3(3.6, 2.85, 8.8)
      : new THREE.Vector3(0, 2.55, 7.4);
    back.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.car.rotation.y);
    const desired = this.car.position.clone().add(back);
    const look = this.captureMode
      ? new THREE.Vector3(-0.5, 0.7, -9)
      : new THREE.Vector3(0, 0.95, -14);
    look.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.car.rotation.y);
    const lookTarget = this.car.position.clone().add(look);
    if (this.captureMode || this.stepIndex < 2) {
      this._camPos.copy(desired);
      this._camTarget.copy(lookTarget);
    } else {
      this._camPos.lerp(desired, 0.2);
      this._camTarget.lerp(lookTarget, 0.24);
    }
    this.camera.position.copy(this._camPos);
    this.camera.lookAt(this._camTarget);
  }

  private _meshOverlap(actors: THREE.Object3D[]) {
    this.car.updateMatrixWorld(true);
    const egoBox = new THREE.Box3().setFromObject(this.car);
    let overlap = false;
    for (const mesh of actors) {
      if (mesh.userData.kind === "pedestrian") continue;
      mesh.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(mesh);
      if (!box.isEmpty() && egoBox.intersectsBox(box)) overlap = true;
    }
    return overlap;
  }

  step() {
    const now = performance.now();
    const dt = now - this._lastFrameMs;
    this._lastFrameMs = now;

    const { action, q, probs } = this._policy();
    this.action = action;
    const traffic = this.traffic.step(this.car, this.stepIndex);
    const actors = traffic.actors;
    const leadGap = traffic.leadGap ?? Infinity;
    const leadZ = typeof traffic.leadZ === "number" ? traffic.leadZ : null;
    this._leadGap = leadGap;
    this._leadZ = leadZ;
    this._bumperGap = leadGap;

    // Scripted playback from bumper leadGap — NOT a learned / sensor closed-loop policy.
    // Same #113-class curve in captureMode: pose the mesh, do not only paint HUD BRK.
    let accel = this.action === 3 ? -3.0 : 1.2;
    if (leadGap < 28) accel = Math.min(accel, -1.8);
    if (leadGap < 20) {
      accel = Math.min(accel, -2.8);
      this.speedKmh = Math.min(this.speedKmh, Math.max(14, (leadGap - BUMPER_CLEAR_M) * 2.4));
    }
    if (leadGap < 14) {
      accel = -5.0;
      this.action = 3;
      this.speedKmh = Math.min(this.speedKmh, Math.max(6, (leadGap - BUMPER_CLEAR_M) * 2.0));
    }
    if (leadGap < 8) {
      accel = -8;
      this.action = 3;
      this.speedKmh = Math.min(this.speedKmh, Math.max(0, (leadGap - BUMPER_CLEAR_M) * 3.2));
    }

    // Integrate speed first so this frame's pose uses the capped value.
    this.speedKmh = Math.max(0, Math.min(65, this.speedKmh + accel * 0.085));

    // Steer AFTER the BRAKE override — policy LEFT/RIGHT must not yaw the hull
    // into the adjacent dark sedan / red pickup while HUD says BRK.
    const steer =
      this.action === 3
        ? 0
        : ({ 0: 0, 1: -0.032, 2: 0.032, 3: 0 } as Record<number, number>)[this.action] ?? 0;
    if (this.captureMode || this.action === 3) {
      this.heading = 0;
    } else {
      this.heading += steer;
    }

    let delta = this.speedKmh / 3.6 / 10;
    if (leadZ !== null) {
      const stopZ = leadZ + NPC_HALF_LENGTH + EGO_HALF_LENGTH + BUMPER_CLEAR_M;
      const nextZ = this.car.position.z - delta;
      if (nextZ < stopZ) {
        delta = Math.max(0, this.car.position.z - stopZ);
        this.car.position.z = stopZ;
        this.speedKmh = 0;
        this.action = 3;
      } else {
        this.car.position.z = nextZ;
      }
    } else {
      this.car.position.z -= delta;
    }
    this.distance += delta;
    this.car.position.x = EGO_LANE_X;
    this.car.rotation.y = this.heading;
    this._bumperGap = leadZ !== null
      ? this.car.position.z - leadZ - EGO_HALF_LENGTH - NPC_HALF_LENGTH
      : leadGap;
    this._overlap = this._meshOverlap(actors);
    const audit = auditOpaqueMaterials(this.car);
    this._transparentMeshes = audit.transparentMeshes;
    this._egoMeshCount = audit.meshCount;

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
      if (this.captureMode) {
        this.hud.state.textContent = "FSD";
        this.hud.context.textContent = "Model Y";
      } else {
        this.hud.state.textContent = "Scripted playback";
        this.hud.context.textContent = `${LABELS[this.action]} · leadGap (not a model)`;
      }
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
      egoX: this.car.position.x,
      heading: this.heading,
      step: this.stepIndex,
      tracks: perception?.tracks?.length ?? 0,
      speedKmh: this.speedKmh,
      action: this.action,
      leadGap: this._bumperGap,
      bumperGap: this._bumperGap,
      overlap: this._overlap,
      transparentMeshes: this._transparentMeshes,
      egoMeshCount: this._egoMeshCount,
      egoModel: this._heroCar?.name ?? "procedural-placeholder",
    };
  }
}
