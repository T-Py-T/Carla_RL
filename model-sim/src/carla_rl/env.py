"""Minimal CARLA ego-camera environment for FSD playback."""

from __future__ import annotations

import random
import time
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import settings

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    carla = None


ACTION_CONTROL = {
    0: carla.VehicleControl(throttle=settings.THROTTLE, steer=0.0)
    if carla
    else None,
    1: carla.VehicleControl(throttle=settings.THROTTLE, steer=-settings.STEER)
    if carla
    else None,
    2: carla.VehicleControl(throttle=settings.THROTTLE, steer=settings.STEER)
    if carla
    else None,
    3: carla.VehicleControl(throttle=0.0, brake=settings.BRAKE) if carla else None,
}


class CarlaPlaybackEnv:
    """Connect to CARLA, spawn an ego vehicle with an RGB camera, and step discretely."""

    def __init__(
        self,
        host: str = settings.CARLA_HOST,
        port: int = settings.CARLA_PORT,
        town: str = settings.DEFAULT_TOWN,
        img_width: int = settings.IMG_WIDTH,
        img_height: int = settings.IMG_HEIGHT,
    ) -> None:
        if carla is None:
            raise RuntimeError(
                "carla Python API not installed. Install the egg matching your "
                "simulator version or run with --offline for proof capture."
            )
        self.host = host
        self.port = port
        self.town = town
        self.img_width = img_width
        self.img_height = img_height
        self.client: Any = None
        self.world: Any = None
        self.vehicle: Any = None
        self.sensor: Any = None
        self.collision_sensor: Any = None
        self._actors: List[Any] = []
        self._frame_queue: Queue = Queue(maxsize=1)
        self._collision = False
        self._last_frame: Optional[np.ndarray] = None

    @staticmethod
    def is_available() -> bool:
        return carla is not None

    def connect(self) -> None:
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(settings.CARLA_TIMEOUT_SEC)
        if self.town:
            self.world = self.client.load_world(self.town)
        else:
            self.world = self.client.get_world()

        if settings.SYNC_MODE:
            sync_settings = self.world.get_settings()
            sync_settings.synchronous_mode = True
            sync_settings.fixed_delta_seconds = settings.FIXED_DELTA_SECONDS
            self.world.apply_settings(sync_settings)

    def reset(self) -> np.ndarray:
        self._cleanup()
        self._collision = False
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, transform)
        self._actors.append(self.vehicle)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.img_width))
        camera_bp.set_attribute("image_size_y", str(self.img_height))
        camera_bp.set_attribute("fov", str(settings.CAMERA_FOV))
        camera_transform = carla.Transform(
            carla.Location(x=settings.CAMERA_OFFSET_X, z=settings.CAMERA_OFFSET_Z)
        )
        self.sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.sensor.listen(self._on_image)
        self._actors.append(self.sensor)

        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda _: self._set_collision())
        self._actors.append(self.collision_sensor)

        self._spawn_traffic(min(settings.NPC_COUNT, 20))
        self._tick_until_frame(timeout=8.0)
        return self.get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        control = ACTION_CONTROL.get(action, ACTION_CONTROL[0])
        self.vehicle.apply_control(control)
        if settings.SYNC_MODE:
            self.world.tick()
        else:
            time.sleep(settings.FIXED_DELTA_SECONDS)
        self._tick_until_frame(timeout=2.0)

        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * float(np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        transform = self.vehicle.get_transform()
        lane_offset = transform.location.y % 3.5 - 1.75
        reward = speed_kmh / 60.0
        done = self._collision
        info = {
            "speed_kmh": speed_kmh,
            "lane_offset": lane_offset,
            "crashed": self._collision,
        }
        return self.get_observation(), reward, done, info

    def get_observation(self) -> np.ndarray:
        if self._last_frame is None:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        return self._last_frame.copy()

    def close(self) -> None:
        self._cleanup()
        if self.world is not None and settings.SYNC_MODE:
            settings_obj = self.world.get_settings()
            settings_obj.synchronous_mode = False
            self.world.apply_settings(settings_obj)

    def _spawn_traffic(self, count: int) -> None:
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for transform in spawn_points[:count]:
            bp = random.choice(blueprints)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_autopilot(True)
                self._actors.append(actor)

    def _on_image(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3][:, :, ::-1].copy()
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                pass
        self._frame_queue.put(bgr)

    def _set_collision(self) -> None:
        self._collision = True

    def _tick_until_frame(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._last_frame = self._frame_queue.get(timeout=0.2)
                return
            except Empty:
                if settings.SYNC_MODE:
                    self.world.tick()

    def _cleanup(self) -> None:
        for actor in reversed(self._actors):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self._actors = []
        self.vehicle = None
        self.sensor = None
        self.collision_sensor = None
