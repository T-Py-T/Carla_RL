# 3D Town FSD Playback (P0)

This deliverable replaces the rejected PR #110 2D highway-env HUD with a **3D town ego-camera** playback experience inspired by JevPilot / FSD demos:

- Perspective ego RGB view (CARLA when available; offline town renderer otherwise)
- Projected candidate path overlays in the camera frame
- Decision readout UI (action label, Q-values, softmax probabilities, JSON inspector)
- Local rule-based policy only — no Jev driver wiring, no `TYPESAFE_API_KEY`

## Quick start (clean checkout, no CARLA required)

```bash
cd model-sim
uv sync --locked --extra apple-gpu --extra dev
uv run python demos/fsd_playback.py --mode offline --headless --steps 240
```

This runs the CPU offline town renderer with full FSD overlays. Use `--save-video /tmp/demo.mp4` to capture MP4 proof locally.

## CARLA 3D playback (Linux + NVIDIA GPU)

1. Start CARLA:

```bash
cd model-sim
chmod +x docker/setup_carla.sh
./docker/setup_carla.sh
```

2. Install the CARLA Python API egg matching simulator `0.9.15` (from the CARLA release `PythonAPI/carla/dist/` folder):

```bash
export CARLA_EGG=/path/to/carla-*-linux-x86_64.egg
uv pip install "$CARLA_EGG"
```

3. Run the demo:

```bash
uv run python demos/fsd_playback.py --mode carla --town Town03 --headless --save-video /tmp/carla_fsd.mp4
```

`--mode auto` (default) connects to CARLA when the API is installed and RPC is reachable; otherwise it falls back to offline mode.

## Capture hosted proof artifacts

```bash
uv run python demos/capture_proof.py --output-dir /opt/cursor/artifacts/carla-fsd-p0
```

Produces:

- `before_plain_ego.png` — plain ego frame (no overlays)
- `after_fsd_overlay.png` — ego frame with projected paths + decision UI
- `fsd_town_playback_demo.gif` — animated rollout
- `fsd_town_playback_demo.mp4` — video rollout

## Makefile shortcuts

```bash
make fsd-playback-offline
make fsd-capture-proof
make carla-up          # requires Docker + NVIDIA
```

## Visual delta vs rejected PR #110

PR #110 was a **top-down 2D highway-env pygame HUD** (Frogger-style). This P0 is a **perspective 3D town ego-camera** with path lines projected into the forward view and a decision panel — matching the JevPilot FSD demo direction Taylor approved.
