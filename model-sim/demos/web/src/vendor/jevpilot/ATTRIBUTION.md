# JevPilot presentation reference — provenance

**Upstream:** https://github.com/standardagents/jevpilot · https://jevpilot.standardagents.ai

## TypeScript modules (attributed adaptation)

These files are TypeScript ports/adaptations of the JevPilot demo presentation layer for Carla_RL FSD playback visual parity. **No Jev decision API** is wired.

| Carla_RL file | JevPilot source (main branch) |
|---------------|-------------------------------|
| `materials.ts` | `src/materials.js` |
| `vehicle-model.ts` | `src/vehicle-model.js` |
| `model-assets.ts` | `src/model-assets.js` |
| `road-vectors.ts` | `src/road-vectors.js` (ribbon GLSL; playback adapter replaces planning.js) |
| `render-profile.ts` | `src/render-profile.js` |
| `asset-loading.ts` | `src/asset-loading.js` |
| `jevpilot-road.ts` | `src/scene.js` (roads, sidewalks, crosswalks, stop bars, traffic lights, stop signs) + `src/world.js` (`signalState`) |
| `scenery-assets.ts` | `src/scenery-assets.js` (instanced street lamps) |
| `../../styles/jevpilot.css` | `src/style.css` (HUD chrome subset used by our `index.html`) |

## Poly Haven CC0 assets (copied from JevPilot `public/`)

| Asset | License | Location |
|-------|---------|----------|
| asphalt_02 maps | [CC0](https://polyhaven.com/license) | `public/textures/asphalt-*.jpg` — see `public/textures/LICENSE.md` |
| concrete_pavement maps | CC0 | `public/textures/pavement-*.jpg` |
| street_lamp_01 | CC0 | `public/models/street_lamp_01/lamp.glb` — see `LICENSE.md` |

## License status (important)

- The JevPilot repository **does not publish a root LICENSE** for application source code.
- **License-clear assets copied:** Model Y GLB (CC BY 4.0), Poly Haven textures/lamp (CC0) — see paths above.
- **Not copied:** HDR `daylight.hdr`, tree/shrub GLBs, other vehicle GLBs.

Use is limited to attributed visual reference for this portfolio demo unless Standard Agents grants separate terms.

## Model Y (ego GLB)

CC BY 4.0 — credit **763468712** / Sketchfab. Full text: `public/models/model-y/LICENSE.txt`.
