# Python Carla_RL → Three.js Scene Shim (documented only — NOT P0)

Legal / asset provenance: `model-sim/demos/web/LEGAL.md`

**Status:** HARD blocked until Taylor clears car asset polish (PR #112 stream).

## Goal (later)

Bridge `model-sim` Python playback (CARLA episodes, offline town, sensor frames) into the vendored JevPilot Three.js presentation layer without Jev decision scripting.

## Proposed contract

```typescript
interface PlaybackFrame {
  ego: { x: number; z: number; heading: number; speedKmh: number };
  traffic: Array<{ id: string; x: number; z: number; heading: number; color: string; kind: "vehicle" | "pedestrian" }>;
  paths: { action: number; probabilities: number[]; steers?: number[] };
  // perception: deferred — do not wire real sensor meshes until asset gate clears
}
```

## Python side (future)

- `demos/fsd_playback.py` emits JSON frames or WebSocket stream
- `carla_rl/offline_town.py` poses → `PlaybackFrame.ego` + `traffic`
- `local_policy.py` probs → `PlaybackFrame.paths`

## Three.js side (current)

- `src/playback/demo.ts` — local policy adapter (poses/paths/probs)
- `src/vendor/jevpilot/*` — presentation only (Model Y GLB, detailedCar, HUD, ribbon shaders)

## Out of scope for this shim doc

- Real LiDAR/camera meshes
- Jev driver / `TYPESAFE_API_KEY`
- Building/scenery upgrade (stays crude)
