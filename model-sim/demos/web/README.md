# FSD town playback demo

Interactive Three.js chase-cam used by PR #114. **Portfolio viewers: this is not a trained driver.**

## Honesty (what this is / is not)

| Now | Later (not this demo) |
|-----|------------------------|
| Ego **slow / BRAKE / BRK** is **scripted** from `TrafficSystem.leadGap` in `src/playback/demo.ts` | Real **sensor→control** |
| Perception overlay is `SensorAdapter.synthesize()` — source `synthetic-adapter` — **viz from traffic actor poses** | **Loaded model** decisions shown in the HUD |
| Model Y mesh is the **controlled vehicle only**. Other cars are generic Kenney/etc. traffic | **Closed-loop RL** |

The HUD banner, top bar, and JSON `honesty` object repeat the same facts so the clip cannot be mistaken for a policy.

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

URL flags: `?capture=1` (solid HUD for screenshots; opaque Model Y hull + snapped chase cam + bumper clamp), `?proceduralTraffic=1` (generic NPC fallback), `?plain=1` (no overlay/HUD). **Do not pass `procedural=1` unless you intentionally skip the Model Y mesh.**

Headless Chromium cannot instantiate the bundled Draco WASM, and the Node undraco path yields a ~20-mesh ghost (whole-body translucent). `capture=1` therefore mounts `createOpaqueModelY()` for the full clip — body stays opaque; sensor fans/rings may be translucent. Interactive browsers still load `public/models/model-y/model-y.glb` and run `hardenEgoMaterials()`. Ego pose/brake is **scripted** from bumper `leadGap` (the mesh actually stops short of the lead). Not closed-loop.
