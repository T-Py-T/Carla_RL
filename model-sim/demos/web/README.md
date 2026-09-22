# FSD town playback demo

Interactive Three.js chase-cam used by PR #115. **Portfolio viewers: this is not a trained DQN driver.**

## Honesty (what this is / is not)

| Now | Later (not this demo) |
|-----|------------------------|
| Ego **brake / accel / BRK** is **sensor-feedback** via `EgoController` — fused LiDAR, proximity zones, and track detections from `SensorAdapter` | **Loaded RL checkpoint** (DQN / policy weights) |
| Closed-loop **sense → decide → actuate** on every frame (`demo.ts` → `ego-controller.ts`) | CARLA live sensor stream |
| Perception overlay source is `synthetic-adapter` (actor poses → structured returns) | Python↔Three.js bridge |

Honesty for maintainers lives here and in the PR body — **not** on the portfolio clip. `?capture=1` hides the honesty banner / JSON sticker and keeps product HUD only (Carla RL · FSD, Model Y, BRK, Slow for hazard, speed). Interactive (`npm run dev`) still shows the maintainer banner.

## Control loop

```
Traffic actors → SensorAdapter.synthesize() → controlObs
  → EgoController.decide() → EgoController.actuate() → ego pose / speed
```

Rule-based longitudinal thresholds on `forwardGapM` (fused track bumper + forward LiDAR). Minimal lateral from asymmetric proximity. **No `.h5` / `.keras` / ONNX weights loaded.**

## Run

```bash
cd model-sim/demos/web
npm install
npm run dev
```

Headless proofs (Playwright viewport, no IDE chrome):

```bash
npm run capture:proof
npm run verify:proof
```

Proofs land in `docs/pr-115-artifacts/` (binary-safe `.gitattributes`).

URL flags: `?capture=1` (portfolio HUD — no honesty sticker; opaque Model Y hull + snapped chase cam + bumper clamp), `?proceduralTraffic=1` (generic NPC fallback), `?plain=1` (no overlay/HUD). **Do not pass `procedural=1` unless you intentionally skip the Model Y mesh.**

Headless Chromium cannot instantiate the bundled Draco WASM, and the Node undraco path yields a ~20-mesh ghost (whole-body translucent). `capture=1` therefore mounts `createOpaqueModelY()` for the full clip — body stays opaque; sensor fans/rings may be translucent. Interactive browsers still load `public/models/model-y/model-y.glb` and run `hardenEgoMaterials()`.
