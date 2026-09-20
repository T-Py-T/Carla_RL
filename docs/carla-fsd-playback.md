# 3D Town FSD Playback (P0)

Taylor-approved P0 direction: **3D town ego-camera** playback with JevPilot / FSD look-and-feel — not the rejected PR #110 2D highway-env HUD.

Gold-standard references (visual only; no Jev driver wiring):

- https://github.com/standardagents/jevpilot
- https://jevpilot.standardagents.ai

## P0 “zero layer” (JevPilot / Tesla FSD bar)

Three.js **chase-cam** town playback — NOT highway-env top-down.

| Requirement | P0 |
|-------------|-----|
| 3D chase cam through town | Behind-ego follow cam, white sedan, Town03-style blocks + intersection |
| Surroundings | NPC cars, pedestrians, buildings, trees, lamps |
| FSD viz feel | Stylized LiDAR point cloud + scan ring + wireframe detection boxes |
| Path predictions | JevPilot ribbon shaders + `.vector-label` probability badges |
| Decision HUD | `.navigation-hud`, `.bottom-hud`, `#json-dialog` |
| Local only | No Jev driver / no `TYPESAFE_API_KEY` |

**Later (NOT P0):** CARLA sensor mesh, real point clouds, live scanner.

## Quick start (clean checkout, no CARLA required)

```bash
cd model-sim
uv sync --locked --extra apple-gpu --extra dev
uv run python demos/fsd_playback.py --mode offline --headless --steps 240
```

## Interactive Three.js demo (browser)

```bash
cd model-sim/demos/web
npm install
# open index.html in a browser, or:
npx serve .
```

Headless proof capture (uses Puppeteer + WebGL):

```bash
cd model-sim/demos/web && npm install
cd ../..
uv run python demos/capture_proof.py --renderer threejs --output-dir /opt/cursor/artifacts/carla-fsd-p0
```

## CARLA 3D playback (Linux + NVIDIA GPU)

```bash
cd model-sim
chmod +x docker/setup_carla.sh && ./docker/setup_carla.sh
# Install matching CARLA 0.9.15 Python API egg
uv run python demos/fsd_playback.py --mode carla --town Town03 --headless --save-video /tmp/carla_fsd.mp4
```

## Makefile shortcuts

```bash
make fsd-playback-offline
make fsd-capture-proof
make carla-up
```

## Visual delta vs rejected PR #110

PR #110 was a **2D top-down highway-env pygame HUD** (Frogger-style). This P0 is a **perspective 3D town ego-camera** with JevPilot-style projected path ribbons and glass HUD panels.
