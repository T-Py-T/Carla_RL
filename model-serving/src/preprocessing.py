"""
Feature preprocessing pipeline for CarlaRL Policy-as-a-Service.

This module handles preprocessing of observations to maintain train-serve parity
and provides utilities for feature transformation and normalization.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import PreprocessingError
from .io_schemas import Observation


class FeaturePreprocessor:
    """
    Base class for feature preprocessing with train-serve parity.

    Ensures consistent feature transformation between training and serving.
    """

    def __init__(self):
        self.is_fitted = False
        self.feature_names = []
        self.input_shape = None
        self.output_shape = None

    def fit(self, observations: list[Observation]) -> "FeaturePreprocessor":
        """
        Fit preprocessor to training data.

        Args:
            observations: List of observations for fitting

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def transform(self, observations: list[Observation]) -> np.ndarray:
        """
        Transform observations to feature matrix.

        Args:
            observations: List of observations to transform

        Returns:
            Feature matrix (batch_size, feature_dim)
        """
        raise NotImplementedError("Subclasses must implement transform method")

    def fit_transform(self, observations: list[Observation]) -> np.ndarray:
        """
        Fit preprocessor and transform observations.

        Args:
            observations: List of observations

        Returns:
            Transformed feature matrix
        """
        return self.fit(observations).transform(observations)

    def save(self, path: str | Path) -> None:
        """
        Save preprocessor to file.

        Args:
            path: Path to save preprocessor
        """
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise PreprocessingError(f"Failed to save preprocessor to {path}: {str(e)}")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        """
        Load preprocessor from file.

        Args:
            path: Path to load preprocessor from

        Returns:
            Loaded preprocessor instance
        """
        try:
            with open(path, "rb") as f:
                preprocessor = pickle.load(f)

            if not isinstance(preprocessor, FeaturePreprocessor):
                raise PreprocessingError(
                    f"Loaded object is not a FeaturePreprocessor: {type(preprocessor)}"
                )

            return preprocessor

        except Exception as e:
            raise PreprocessingError(f"Failed to load preprocessor from {path}: {str(e)}")


class StandardFeaturePreprocessor(FeaturePreprocessor):
    """
    Standard feature preprocessor with normalization and scaling.

    Extracts features from observations and applies normalization.
    """

    def __init__(
        self,
        normalize_speed: bool = True,
        normalize_steering: bool = True,
        normalize_sensors: bool = True,
        sensor_clip_range: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.normalize_speed = normalize_speed
        self.normalize_steering = normalize_steering
        self.normalize_sensors = normalize_sensors
        self.sensor_clip_range = sensor_clip_range or (-10.0, 10.0)

        # Statistics for normalization
        self.speed_stats = {"mean": 0.0, "std": 1.0}
        self.steering_stats = {"mean": 0.0, "std": 1.0}
        self.sensor_stats = {"mean": 0.0, "std": 1.0}

        # Feature configuration
        self.feature_names = ["speed", "steering", "sensors"]

    def _extract_features(self, observations: list[Observation]) -> dict[str, np.ndarray]:
        """
        Extract raw features from observations.

        Args:
            observations: List of observations

        Returns:
            Dictionary of feature arrays
        """
        try:
            speeds = np.array([obs.speed for obs in observations], dtype=np.float32)
            steerings = np.array([obs.steering for obs in observations], dtype=np.float32)

            # Handle variable-length sensor arrays
            max_sensor_len = max(len(obs.sensors) for obs in observations)
            sensors = np.zeros((len(observations), max_sensor_len), dtype=np.float32)

            for i, obs in enumerate(observations):
                sensor_len = len(obs.sensors)
                sensors[i, :sensor_len] = obs.sensors[:max_sensor_len]
                # Pad with zeros if needed (already initialized to zeros)

            return {"speed": speeds, "steering": steerings, "sensors": sensors}

        except Exception as e:
            raise PreprocessingError(
                f"Failed to extract features: {str(e)}",
                details={"num_observations": len(observations)},
            )

    def fit(self, observations: list[Observation]) -> "StandardFeaturePreprocessor":
        """
        Fit preprocessor statistics to observations.

        Args:
            observations: Training observations

        Returns:
            Self for method chaining
        """
        if not observations:
            raise PreprocessingError("Cannot fit preprocessor to empty observation list")

        try:
            features = self._extract_features(observations)

            # Compute normalization statistics
            if self.normalize_speed:
                self.speed_stats["mean"] = float(np.mean(features["speed"]))
                self.speed_stats["std"] = float(np.std(features["speed"]) + 1e-8)

            if self.normalize_steering:
                self.steering_stats["mean"] = float(np.mean(features["steering"]))
                self.steering_stats["std"] = float(np.std(features["steering"]) + 1e-8)

            if self.normalize_sensors:
                # Clip sensors before computing stats
                clipped_sensors = np.clip(
                    features["sensors"], self.sensor_clip_range[0], self.sensor_clip_range[1]
                )
                self.sensor_stats["mean"] = float(np.mean(clipped_sensors))
                self.sensor_stats["std"] = float(np.std(clipped_sensors) + 1e-8)

            # Set input/output shapes
            sample_transformed = self.transform(observations[:1])
            self.input_shape = [len(observations[0].sensors) + 2]  # sensors + speed + steering
            self.output_shape = list(sample_transformed.shape[1:])

            self.is_fitted = True
            return self

        except Exception as e:
            raise PreprocessingError(
                f"Failed to fit preprocessor: {str(e)}",
                details={"num_observations": len(observations)},
            )

    def transform(self, observations: list[Observation]) -> np.ndarray:
        """
        Transform observations to normalized feature matrix.

        Args:
            observations: List of observations to transform

        Returns:
            Normalized feature matrix (batch_size, feature_dim)
        """
        if not observations:
            raise PreprocessingError("Cannot transform empty observation list")

        try:
            features = self._extract_features(observations)

            # Normalize features
            speed = features["speed"]
            steering = features["steering"]
            sensors = features["sensors"]

            if self.normalize_speed and self.is_fitted:
                speed = (speed - self.speed_stats["mean"]) / self.speed_stats["std"]

            if self.normalize_steering and self.is_fitted:
                steering = (steering - self.steering_stats["mean"]) / self.steering_stats["std"]

            if self.normalize_sensors and self.is_fitted:
                # Clip sensors to range
                sensors = np.clip(sensors, self.sensor_clip_range[0], self.sensor_clip_range[1])
                sensors = (sensors - self.sensor_stats["mean"]) / self.sensor_stats["std"]

            # Concatenate all features
            feature_matrix = np.column_stack(
                [speed.reshape(-1, 1), steering.reshape(-1, 1), sensors]
            )

            return feature_matrix.astype(np.float32)

        except Exception as e:
            raise PreprocessingError(
                f"Failed to transform observations: {str(e)}",
                details={"num_observations": len(observations), "is_fitted": self.is_fitted},
            )

    def get_feature_info(self) -> dict[str, Any]:
        """
        Get information about preprocessor configuration and statistics.

        Returns:
            Dictionary with preprocessor information
        """
        return {
            "type": "StandardFeaturePreprocessor",
            "is_fitted": self.is_fitted,
            "normalize_speed": self.normalize_speed,
            "normalize_steering": self.normalize_steering,
            "normalize_sensors": self.normalize_sensors,
            "sensor_clip_range": self.sensor_clip_range,
            "speed_stats": self.speed_stats,
            "steering_stats": self.steering_stats,
            "sensor_stats": self.sensor_stats,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "feature_names": self.feature_names,
        }


class MinimalPreprocessor(FeaturePreprocessor):
    """
    Minimal preprocessor that just converts observations to feature matrix.

    No normalization or scaling applied - useful for models that expect raw features.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = ["speed", "steering", "sensors"]

    def fit(self, observations: list[Observation]) -> "MinimalPreprocessor":
        """
        Fit preprocessor (no-op for minimal preprocessor).

        Args:
            observations: Training observations (unused)

        Returns:
            Self for method chaining
        """
        if observations:
            # Set shapes based on first observation
            sample_transformed = self.transform(observations[:1])
            self.input_shape = [len(observations[0].sensors) + 2]
            self.output_shape = list(sample_transformed.shape[1:])

        self.is_fitted = True
        return self

    def transform(self, observations: list[Observation]) -> np.ndarray:
        """
        Transform observations to raw feature matrix.

        Args:
            observations: List of observations to transform

        Returns:
            Raw feature matrix (batch_size, feature_dim)
        """
        if not observations:
            raise PreprocessingError("Cannot transform empty observation list")

        try:
            # Extract features without normalization
            speeds = np.array([obs.speed for obs in observations], dtype=np.float32)
            steerings = np.array([obs.steering for obs in observations], dtype=np.float32)

            # Handle variable-length sensor arrays
            max_sensor_len = max(len(obs.sensors) for obs in observations)
            sensors = np.zeros((len(observations), max_sensor_len), dtype=np.float32)

            for i, obs in enumerate(observations):
                sensor_len = len(obs.sensors)
                sensors[i, :sensor_len] = obs.sensors[:max_sensor_len]

            # Concatenate all features
            feature_matrix = np.column_stack(
                [speeds.reshape(-1, 1), steerings.reshape(-1, 1), sensors]
            )

            return feature_matrix.astype(np.float32)

        except Exception as e:
            raise PreprocessingError(
                f"Failed to transform observations: {str(e)}",
                details={"num_observations": len(observations)},
            )


def create_preprocessor(config: dict[str, Any]) -> FeaturePreprocessor:
    """
    Create preprocessor from configuration.

    Args:
        config: Preprocessor configuration dictionary

    Returns:
        Configured preprocessor instance

    Raises:
        PreprocessingError: If configuration is invalid
    """
    preprocessor_type = config.get("type", "standard").lower()

    if preprocessor_type == "standard":
        return StandardFeaturePreprocessor(
            normalize_speed=config.get("normalize_speed", True),
            normalize_steering=config.get("normalize_steering", True),
            normalize_sensors=config.get("normalize_sensors", True),
            sensor_clip_range=config.get("sensor_clip_range", (-10.0, 10.0)),
        )
    elif preprocessor_type == "minimal":
        return MinimalPreprocessor()
    else:
        raise PreprocessingError(
            f"Unknown preprocessor type: {preprocessor_type}",
            details={"available_types": ["standard", "minimal"]},
        )


def to_feature_matrix(observations: list[Observation]) -> np.ndarray:
    """
    Convert observations to feature matrix using minimal preprocessing.

    Utility function for quick conversion without fitted preprocessor.

    Args:
        observations: List of observations

    Returns:
        Feature matrix (batch_size, feature_dim)
    """
    preprocessor = MinimalPreprocessor()
    return preprocessor.fit_transform(observations)


def validate_preprocessing_parity(
    train_preprocessor: FeaturePreprocessor,
    serve_preprocessor: FeaturePreprocessor,
    test_observations: list[Observation],
    tolerance: float = 1e-6,
) -> bool:
    """
    Validate that training and serving preprocessors produce identical results.

    Args:
        train_preprocessor: Preprocessor used during training
        serve_preprocessor: Preprocessor used during serving
        test_observations: Test observations for validation
        tolerance: Numerical tolerance for comparison

    Returns:
        True if preprocessors produce identical results

    Raises:
        PreprocessingError: If parity validation fails
    """
    try:
        train_features = train_preprocessor.transform(test_observations)
        serve_features = serve_preprocessor.transform(test_observations)

        if train_features.shape != serve_features.shape:
            raise PreprocessingError(
                "Shape mismatch between train and serve features",
                details={"train_shape": train_features.shape, "serve_shape": serve_features.shape},
            )

        max_diff = np.max(np.abs(train_features - serve_features))

        if max_diff > tolerance:
            raise PreprocessingError(
                f"Feature parity validation failed: max difference {max_diff} > tolerance {tolerance}",
                details={
                    "max_difference": float(max_diff),
                    "tolerance": tolerance,
                    "mean_difference": float(np.mean(np.abs(train_features - serve_features))),
                },
            )

        return True

    except Exception as e:
        if isinstance(e, PreprocessingError):
            raise
        else:
            raise PreprocessingError(
                f"Parity validation failed: {str(e)}",
                details={"num_test_observations": len(test_observations)},
            )
