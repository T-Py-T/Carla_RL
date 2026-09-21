# Legal provenance — FSD web demo (PR #112 car polish)

Taylor / Firstmate: **reuse only within upstream licenses.** This file is the PR provenance index.

## Bundled assets (license-clear)

| Asset | License | Location | Attribution |
|-------|---------|----------|-------------|
| **Tesla Model Y 2021 GLB** (ego only) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/model-y/model-y.glb` | `public/models/model-y/ATTRIBUTION.md`, `LICENSE.txt` — artist **763468712** on Sketchfab (credit text in LICENSE.txt **must not be stripped**) |
| **Kenney Car Kit** (NPC cars/trucks/vans/emergency) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | `public/models/traffic/kenney-car-kit/` | `ATTRIBUTION.md`, `LICENSE.txt` — [kenney.nl/assets/car-kit](https://kenney.nl/assets/car-kit), verified on download |
| **Khronos ToyCar** (NPC) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | `public/models/traffic/khronos-toy-car/` | `ATTRIBUTION.md` — glTF Sample Assets |
| **Khronos Cesium Milk Truck** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/khronos-milk-truck/` | `ATTRIBUTION.md`, `LICENSE.txt` — glTF Sample Assets |
| **OGA UAZ Truck** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/oga-uaz-truck/` | `ATTRIBUTION.md`, `LICENSE.txt` — brylie / OpenGameArt, Sketchfab ANDREO12 |
| **Sketchfab Generic Sedan** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/sketchfab-generic-sedan/` | MMC Works — `ATTRIBUTION.md`, `LICENSE.txt` |
| **Sketchfab Modern Sedan** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/sketchfab-modern-sedan/` | Chazbc |
| **Sketchfab ToyoAce Van** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/sketchfab-toyoace-van/` | ROH3D |
| **Sketchfab Renault Master Van** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/sketchfab-renault-master-van/` | Nieve5677 |
| **Sketchfab GMC School Bus** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/sketchfab-gmc-school-bus/` | Nieve5677 |
| **Sketchfab Box Truck** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/sketchfab-box-truck/` | jamesli8 |
| **Sketchfab Delivery Truck** (NPC) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `public/models/traffic/sketchfab-delivery-truck/` | MrPrisma3D |
| **Poly Haven asphalt_02** (road surface) | [CC0 1.0](https://polyhaven.com/license) | `public/textures/asphalt-*.jpg` | `public/textures/LICENSE.md` — via JevPilot bundle |
| **Poly Haven concrete_pavement** (sidewalks) | CC0 | `public/textures/pavement-*.jpg` | same |
| **Poly Haven street_lamp_01** | CC0 | `public/models/street_lamp_01/lamp.glb` | `LICENSE.md` — JevPilot `scenery-assets.js` placement |
| **Draco decoder** | Apache 2.0 (Google) | `public/draco/` | Bundled for GLB decompression; standard Three.js Draco path |
| **three.js** | MIT | npm dependency | See `package.json` |

Stable doc mirror: `models/model-y/ATTRIBUTION.md` + `LICENSE.txt` (no binary duplicate).

## JevPilot visual reference (application source)

Reference demo: [standardagents/jevpilot](https://github.com/standardagents/jevpilot) · [jevpilot.standardagents.ai](https://jevpilot.standardagents.ai)

| Item | Status |
|------|--------|
| JevPilot **application source** (`src/*.js`, `src/style.css`) | **No root SPDX license** in upstream repo (`package.json` is `"private": true`). TypeScript modules under `src/vendor/jevpilot/` (including `jevpilot-road.ts` — road surface, lane lines, sidewalks, stop bars, traffic-light cycling) and `src/styles/jevpilot.css` are **attributed adaptations** for portfolio visual parity — see `src/vendor/jevpilot/ATTRIBUTION.md`. |
| JevPilot **public/** assets we bundle | `model-y` (ego), Poly Haven textures + `street_lamp_01` (CC0). NPC traffic uses Kenney/Sketchfab shortlist — **not** Model Y clones. |
| Jev driver / `TYPESAFE_API_KEY` | **Not used** |

Before commercial redistribution beyond portfolio demo, confirm application-code terms with Standard Agents if required.

## Perception overlay (PR #114)

- **Synthetic sensor adapter** (`src/playback/sensor-adapter.ts`): structured LiDAR wedge, proximity zones, persistent track IDs — derived from traffic actor poses (not CARLA sensor stream).
- CARLA live sensor streaming remains future work (`docs/python-three-shim.md`).

## Authorship

Cloud Agent commits on branch `username/car-asset-polish-6e4a` are **not** final author. Parent rewrites to Taylor identity on Mac before merge.

## CC BY 4.0 credit (Model Y) — retain verbatim

```
This work is based on "Tesla Model Y 2021" (https://sketchfab.com/3d-models/tesla-model-y-2021-c0a86cac582d4b33aba0fb1b1912d970) by 763468712 (https://sketchfab.com/763468712) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
Adaptations by Tina 3D Tesla: normalization, connected mesh-island separation, polygon reduction, studio materials. Mesh islands are not verified Tesla service parts.
```
