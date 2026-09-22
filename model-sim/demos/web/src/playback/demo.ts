import * as THREE from "three";
import { PlaybackPathVectors } from "../vendor/jevpilot/road-vectors";
import { loadHeroCar, updateHeroWheels } from "../vendor/jevpilot/model-assets";
import { renderProfile } from "../vendor/jevpilot/render-profile";
import { auditOpaqueMaterials, createOpaqueModelY, loadTrafficFleet } from "./ego";
import { updateTrafficLights } from "../vendor/jevpilot/jevpilot-road";
import {
  buildTown,
  EGO_LANE_X,
  TrafficSystem,
} from "./world";
import { PerceptionViz } from "./perception-viz";
import { SensorAdapter } from "./sensor-adapter";
import { EgoController } from "./ego-controller";

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
  /** Last fused sensor forward gap (control feedback — not TrafficSystem cheat). */
  private _forwardGap = Infinity;
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

  /** Product labels only — no maintainer honesty sticker on the clip. */
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

  private _renderJson(payload: Record<string, unknown>) {
    this._lastPayload = payload;
    const view =
      this.activeTab === "decision"
        ? {
            maneuver: payload.maneuver,
            action: payload.action,
            probabilities: payload.probabilities,
            perception: payload.perception,
            control_loop: payload.control_loop,
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

    const traffic = this.traffic.step(this.car, this.stepIndex);
    const actors = traffic.actors;

    // ── SENSE: synthesize sensor frame from scene actors (before control) ──
    const sensorFrame = this.sensor.synthesize(this.car, [this.car, ...actors]);
    const obs = EgoController.observe(sensorFrame, this.car);
    this._forwardGap = obs.forwardGapM;

    // ── DECIDE: longitudinal (+ minimal lateral) from fused observations ──
    const decision = EgoController.decide(obs, this.speedKmh);
    this.action = decision.action;

    // ── ACTUATE: integrate speed / pose from decision + track bumper clamp ──
    const result = EgoController.actuate(
      this.car,
      decision,
      obs,
      this.speedKmh,
      this.heading,
      this.captureMode,
    );
    this.speedKmh = result.speedKmh;
    this.heading = result.heading;
    this.action = result.action;
    this.car.position.z = result.positionZ;
    this.distance += result.delta;
    this.car.position.x = EGO_LANE_X;
    this.car.rotation.y = this.heading;
    this._bumperGap = result.forwardGapM;

    this._overlap = this._meshOverlap(actors);
    const audit = auditOpaqueMaterials(this.car);
    this._transparentMeshes = audit.transparentMeshes;
    this._egoMeshCount = audit.meshCount;

    if (this._heroCar) {
      updateHeroWheels(this._heroCar, result.delta, decision.steer * 8);
    }

    const lights = this.scene.userData.trafficLights;
    if (lights?.length) updateTrafficLights(lights, now / 1000);

    this._updateChaseCamera();

    if (!this.plain) {
      this.perception.update(this.car, sensorFrame);
      this.vectors.update(this.car, this.camera, this.width, this.height, dt, this.action, decision.probs);

      this.hud.maneuver.textContent = MANEUVER[this.action];
      this.hud.distance.textContent = `${Math.round(this.distance)} m ahead`;
      this.hud.remaining.textContent = `${Math.max(0, 420 - Math.round(this.distance))} m left`;
      this.hud.speed.textContent = String(Math.round(this.speedKmh));
      if (this.captureMode) {
        this.hud.state.textContent = "FSD";
        this.hud.context.textContent = "Model Y";
      } else {
        this.hud.state.textContent = "FSD";
        this.hud.context.textContent = `${LABELS[this.action]} · sensor feedback`;
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
        control_loop: {
          sense: "SensorAdapter.synthesize → controlObs",
          decide: "EgoController.decide (rule-based, not RL checkpoint)",
          actuate: "EgoController.actuate → pose/speed",
          closed_loop: true,
          loaded_policy: false,
        },
        q_values: Object.fromEntries(decision.q.map((v, i) => [String(i), +v.toFixed(3)])),
        probabilities: Object.fromEntries(decision.probs.map((v, i) => [String(i), +v.toFixed(3)])),
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
    const controlLoop = this._lastPayload.control_loop as { closed_loop?: boolean } | undefined;
    return {
      egoZ: this.car.position.z,
      egoX: this.car.position.x,
      heading: this.heading,
      step: this.stepIndex,
      tracks: perception?.tracks?.length ?? 0,
      speedKmh: this.speedKmh,
      action: this.action,
      leadGap: this._bumperGap,
      forwardGap: this._forwardGap,
      bumperGap: this._bumperGap,
      closedLoop: controlLoop?.closed_loop ?? false,
      overlap: this._overlap,
      transparentMeshes: this._transparentMeshes,
      egoMeshCount: this._egoMeshCount,
      egoModel: this._heroCar?.name ?? "procedural-placeholder",
    };
  }
}
