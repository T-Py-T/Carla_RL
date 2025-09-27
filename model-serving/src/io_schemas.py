"""
Pydantic schemas for CarlaRL Policy-as-a-Service API.

This module defines all input/output schemas for the FastAPI endpoints,
ensuring type safety and validation for the serving infrastructure.
"""

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Observation(BaseModel):
    """
    Single observation from the CARLA environment.

    Represents the current state of the vehicle including speed, steering,
    and sensor readings (cameras, lidar, etc.) as flattened arrays.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "speed": 25.5,
                "steering": 0.1,
                "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
            }
        }
    )

    speed: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Vehicle speed in km/h"
    )
    steering: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Current steering angle (-1.0 = full left, 1.0 = full right)"
    )
    sensors: list[float] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Flattened sensor data (camera pixels, lidar points, etc.)"
    )

    @field_validator('sensors')
    @classmethod
    def validate_sensors(cls, v: list[float]) -> list[float]:
        """Ensure all sensor values are finite and within reasonable bounds."""
        for i, val in enumerate(v):
            if not (-1000.0 <= val <= 1000.0):
                raise ValueError(f"Sensor value at index {i} out of bounds: {val}")
        return v


class Action(BaseModel):
    """
    Action output from the RL policy.

    Represents the control commands for the vehicle: throttle, brake, and steering.
    All values are normalized between -1.0 and 1.0 or 0.0 and 1.0 as appropriate.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "throttle": 0.7,
                "brake": 0.0,
                "steer": 0.1
            }
        }
    )

    throttle: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Throttle intensity (0.0 = no throttle, 1.0 = full throttle)"
    )
    brake: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Brake intensity (0.0 = no brake, 1.0 = full brake)"
    )
    steer: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Steering angle (-1.0 = full left, 1.0 = full right)"
    )


class PredictRequest(BaseModel):
    """
    Request payload for the /predict endpoint.

    Contains batch of observations and optional deterministic mode flag.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "observations": [
                    {
                        "speed": 25.5,
                        "steering": 0.1,
                        "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
                    }
                ],
                "deterministic": True
            }
        }
    )

    observations: list[Observation] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Batch of observations to process"
    )
    deterministic: bool | None = Field(
        default=False,
        description="Whether to use deterministic inference (reproducible outputs)"
    )


class PredictResponse(BaseModel):
    """
    Response payload from the /predict endpoint.

    Contains predicted actions, model version, and timing information.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "actions": [
                    {
                        "throttle": 0.7,
                        "brake": 0.0,
                        "steer": 0.1
                    }
                ],
                "version": "v0.1.0",
                "timingMs": 8.5,
                "deterministic": True
            }
        }
    )

    actions: list[Action] = Field(
        ...,
        description="Predicted actions corresponding to input observations"
    )
    version: str = Field(
        ...,
        description="Model version used for inference"
    )
    timingMs: float = Field(
        ...,
        ge=0.0,
        description="Inference time in milliseconds"
    )
    deterministic: bool = Field(
        ...,
        description="Whether deterministic inference was used"
    )


class HealthResponse(BaseModel):
    """Response schema for /healthz endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "version": "v0.1.0",
                "git": "abc123ef",
                "device": "cpu",
                "timestamp": 1695825600.0
            }
        }
    )

    status: str = Field(
        ...,
        description="Service health status"
    )
    version: str = Field(
        ...,
        description="Service version"
    )
    git: str = Field(
        ...,
        description="Git commit SHA"
    )
    device: str = Field(
        ...,
        description="Inference device (cpu/cuda)"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Response timestamp"
    )


class MetadataResponse(BaseModel):
    """Response schema for /metadata endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "modelName": "carla-ppo",
                "version": "v0.1.0",
                "device": "cpu",
                "inputShape": [5],
                "actionSpace": {
                    "throttle": [0.0, 1.0],
                    "brake": [0.0, 1.0],
                    "steer": [-1.0, 1.0]
                }
            }
        }
    )

    modelName: str = Field(
        ...,
        description="Name of the loaded model"
    )
    version: str = Field(
        ...,
        description="Model version"
    )
    device: str = Field(
        ...,
        description="Inference device"
    )
    inputShape: list[int] = Field(
        ...,
        description="Expected input tensor shape"
    )
    actionSpace: dict[str, list[float]] = Field(
        ...,
        description="Action space bounds for each action dimension"
    )


class WarmupResponse(BaseModel):
    """Response schema for /warmup endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "warmed",
                "timingMs": 150.2,
                "device": "cpu"
            }
        }
    )

    status: str = Field(
        ...,
        description="Warmup status"
    )
    timingMs: float = Field(
        ...,
        ge=0.0,
        description="Warmup time in milliseconds"
    )
    device: str = Field(
        ...,
        description="Device used for warmup"
    )


class ErrorResponse(BaseModel):
    """Standard error response schema for all endpoints."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid observation data",
                "details": {
                    "field": "sensors",
                    "value": "too_large"
                },
                "timestamp": 1695825600.0
            }
        }
    )

    error: str = Field(
        ...,
        description="Error type/category"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error context and debugging information"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Error timestamp"
    )
