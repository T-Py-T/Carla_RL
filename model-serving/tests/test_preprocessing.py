"""
Unit tests for preprocessing pipeline in CarlaRL Policy-as-a-Service.

Tests feature preprocessing, normalization, and train-serve parity validation.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.exceptions import PreprocessingError
from src.io_schemas import Observation
from src.preprocessing import (
    FeaturePreprocessor,
    MinimalPreprocessor,
    StandardFeaturePreprocessor,
    create_preprocessor,
    to_feature_matrix,
    validate_preprocessing_parity,
)


class TestFeaturePreprocessorBase:
    """Test cases for base FeaturePreprocessor class."""

    def test_base_preprocessor_abstract_methods(self):
        """Test that base class raises NotImplementedError for abstract methods."""
        preprocessor = FeaturePreprocessor()

        with pytest.raises(NotImplementedError):
            preprocessor.fit([])

        with pytest.raises(NotImplementedError):
            preprocessor.transform([])

    def test_base_preprocessor_save_load(self):
        """Test save/load functionality for preprocessor."""
        # Use MinimalPreprocessor for testing since base class is abstract
        preprocessor = MinimalPreprocessor()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_preprocessor.pkl"

            # Save preprocessor
            preprocessor.save(temp_path)
            assert temp_path.exists()

            # Load preprocessor
            loaded_preprocessor = FeaturePreprocessor.load(temp_path)
            assert isinstance(loaded_preprocessor, MinimalPreprocessor)

    def test_base_preprocessor_save_error(self):
        """Test save error handling."""
        preprocessor = MinimalPreprocessor()

        # Try to save to invalid path
        with pytest.raises(PreprocessingError):
            preprocessor.save("/invalid/path/preprocessor.pkl")

    def test_base_preprocessor_load_error(self):
        """Test load error handling."""
        # Try to load nonexistent file
        with pytest.raises(PreprocessingError):
            FeaturePreprocessor.load("nonexistent.pkl")

    def test_base_preprocessor_load_invalid_object(self):
        """Test loading invalid object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "invalid.pkl"

            # Save non-preprocessor object
            import pickle

            with open(temp_path, "wb") as f:
                pickle.dump({"not": "preprocessor"}, f)

            with pytest.raises(PreprocessingError) as exc_info:
                FeaturePreprocessor.load(temp_path)

            assert "not a FeaturePreprocessor" in str(exc_info.value)


class TestStandardFeaturePreprocessor:
    """Test cases for StandardFeaturePreprocessor."""

    def create_test_observations(self, n: int = 5) -> list[Observation]:
        """Create test observations for testing."""
        observations = []
        for i in range(n):
            obs = Observation(
                speed=20.0 + i * 5.0,  # 20, 25, 30, 35, 40
                steering=-0.2 + i * 0.1,  # -0.2, -0.1, 0.0, 0.1, 0.2
                sensors=[0.1 * j + i * 0.1 for j in range(3)],  # Variable sensor values
            )
            observations.append(obs)
        return observations

    def test_standard_preprocessor_creation(self):
        """Test StandardFeaturePreprocessor creation."""
        preprocessor = StandardFeaturePreprocessor()

        assert preprocessor.normalize_speed is True
        assert preprocessor.normalize_steering is True
        assert preprocessor.normalize_sensors is True
        assert preprocessor.is_fitted is False

    def test_standard_preprocessor_creation_custom_config(self):
        """Test StandardFeaturePreprocessor with custom configuration."""
        preprocessor = StandardFeaturePreprocessor(
            normalize_speed=False,
            normalize_steering=True,
            normalize_sensors=False,
            sensor_clip_range=(-5.0, 5.0),
        )

        assert preprocessor.normalize_speed is False
        assert preprocessor.normalize_steering is True
        assert preprocessor.normalize_sensors is False
        assert preprocessor.sensor_clip_range == (-5.0, 5.0)

    def test_standard_preprocessor_fit(self):
        """Test fitting StandardFeaturePreprocessor."""
        observations = self.create_test_observations()
        preprocessor = StandardFeaturePreprocessor()

        result = preprocessor.fit(observations)

        assert result is preprocessor  # Should return self
        assert preprocessor.is_fitted is True
        assert preprocessor.speed_stats["mean"] == 30.0  # Mean of [20, 25, 30, 35, 40]
        assert preprocessor.speed_stats["std"] > 0

    def test_standard_preprocessor_fit_empty(self):
        """Test fitting with empty observation list."""
        preprocessor = StandardFeaturePreprocessor()

        with pytest.raises(PreprocessingError) as exc_info:
            preprocessor.fit([])

        assert "empty observation list" in str(exc_info.value)

    def test_standard_preprocessor_transform(self):
        """Test transforming observations."""
        observations = self.create_test_observations()
        preprocessor = StandardFeaturePreprocessor()

        # Fit first
        preprocessor.fit(observations)

        # Transform
        features = preprocessor.transform(observations)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(observations)
        assert features.shape[1] == 2 + 3  # speed + steering + 3 sensors
        assert features.dtype == np.float32

    def test_standard_preprocessor_transform_without_fit(self):
        """Test transforming without fitting (should still work but no normalization)."""
        observations = self.create_test_observations()
        preprocessor = StandardFeaturePreprocessor()

        features = preprocessor.transform(observations)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(observations)

    def test_standard_preprocessor_transform_empty(self):
        """Test transforming empty observation list."""
        preprocessor = StandardFeaturePreprocessor()

        with pytest.raises(PreprocessingError) as exc_info:
            preprocessor.transform([])

        assert "empty observation list" in str(exc_info.value)

    def test_standard_preprocessor_fit_transform(self):
        """Test fit_transform method."""
        observations = self.create_test_observations()
        preprocessor = StandardFeaturePreprocessor()

        features = preprocessor.fit_transform(observations)

        assert preprocessor.is_fitted is True
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(observations)

    def test_standard_preprocessor_normalization(self):
        """Test that normalization actually normalizes features."""
        observations = self.create_test_observations()
        preprocessor = StandardFeaturePreprocessor()

        # Fit and transform
        features = preprocessor.fit_transform(observations)

        # Check that speed column (index 0) is normalized
        speed_column = features[:, 0]
        assert abs(np.mean(speed_column)) < 1e-6  # Should be close to 0
        assert abs(np.std(speed_column) - 1.0) < 1e-6  # Should be close to 1

    def test_standard_preprocessor_sensor_clipping(self):
        """Test sensor clipping functionality."""
        # Create observations with extreme sensor values
        observations = [Observation(speed=25.0, steering=0.0, sensors=[-100.0, 100.0, 0.0])]

        preprocessor = StandardFeaturePreprocessor(sensor_clip_range=(-10.0, 10.0))
        features = preprocessor.fit_transform(observations)

        # Sensor values should be clipped
        sensor_features = features[0, 2:]  # Skip speed and steering
        assert np.all(sensor_features >= -10.0)
        assert np.all(sensor_features <= 10.0)

    def test_standard_preprocessor_get_feature_info(self):
        """Test getting feature information."""
        observations = self.create_test_observations()
        preprocessor = StandardFeaturePreprocessor()
        preprocessor.fit(observations)

        info = preprocessor.get_feature_info()

        assert info["type"] == "StandardFeaturePreprocessor"
        assert info["is_fitted"] is True
        assert "speed_stats" in info
        assert "steering_stats" in info
        assert "sensor_stats" in info
        assert "input_shape" in info
        assert "output_shape" in info


class TestMinimalPreprocessor:
    """Test cases for MinimalPreprocessor."""

    def create_test_observations(self, n: int = 3) -> list[Observation]:
        """Create test observations for testing."""
        observations = []
        for i in range(n):
            obs = Observation(
                speed=20.0 + i,
                steering=i * 0.1,
                sensors=[0.1 * j for j in range(2)],  # 2 sensors
            )
            observations.append(obs)
        return observations

    def test_minimal_preprocessor_creation(self):
        """Test MinimalPreprocessor creation."""
        preprocessor = MinimalPreprocessor()

        assert preprocessor.feature_names == ["speed", "steering", "sensors"]
        assert preprocessor.is_fitted is False

    def test_minimal_preprocessor_fit(self):
        """Test fitting MinimalPreprocessor."""
        observations = self.create_test_observations()
        preprocessor = MinimalPreprocessor()

        result = preprocessor.fit(observations)

        assert result is preprocessor
        assert preprocessor.is_fitted is True
        assert preprocessor.input_shape is not None
        assert preprocessor.output_shape is not None

    def test_minimal_preprocessor_transform(self):
        """Test transforming with MinimalPreprocessor."""
        observations = self.create_test_observations()
        preprocessor = MinimalPreprocessor()

        features = preprocessor.transform(observations)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(observations)
        assert features.shape[1] == 2 + 2  # speed + steering + 2 sensors
        assert features.dtype == np.float32

        # Verify no normalization (raw values preserved)
        assert features[0, 0] == 20.0  # First speed value
        assert features[1, 0] == 21.0  # Second speed value

    def test_minimal_preprocessor_variable_sensor_length(self):
        """Test handling of variable sensor lengths."""
        observations = [
            Observation(speed=20.0, steering=0.0, sensors=[1.0, 2.0]),
            Observation(speed=25.0, steering=0.1, sensors=[3.0, 4.0, 5.0]),  # 3 sensors
            Observation(speed=30.0, steering=0.2, sensors=[6.0]),  # 1 sensor
        ]

        preprocessor = MinimalPreprocessor()
        features = preprocessor.transform(observations)

        # Should pad to max sensor length (3)
        assert features.shape == (3, 2 + 3)  # speed + steering + 3 sensors

        # Check padding with zeros
        assert features[0, 4] == 0.0  # Padded sensor for first observation
        assert features[2, 3] == 0.0  # Padded sensor for third observation
        assert features[2, 4] == 0.0  # Padded sensor for third observation


class TestPreprocessorFactory:
    """Test cases for preprocessor factory function."""

    def test_create_preprocessor_standard(self):
        """Test creating standard preprocessor."""
        config = {"type": "standard"}
        preprocessor = create_preprocessor(config)

        assert isinstance(preprocessor, StandardFeaturePreprocessor)

    def test_create_preprocessor_standard_with_config(self):
        """Test creating standard preprocessor with custom config."""
        config = {
            "type": "standard",
            "normalize_speed": False,
            "normalize_steering": True,
            "sensor_clip_range": [-5.0, 5.0],
        }
        preprocessor = create_preprocessor(config)

        assert isinstance(preprocessor, StandardFeaturePreprocessor)
        assert preprocessor.normalize_speed is False
        assert preprocessor.normalize_steering is True
        assert preprocessor.sensor_clip_range == [-5.0, 5.0]

    def test_create_preprocessor_minimal(self):
        """Test creating minimal preprocessor."""
        config = {"type": "minimal"}
        preprocessor = create_preprocessor(config)

        assert isinstance(preprocessor, MinimalPreprocessor)

    def test_create_preprocessor_default(self):
        """Test creating preprocessor with default type."""
        config = {}  # No type specified
        preprocessor = create_preprocessor(config)

        assert isinstance(preprocessor, StandardFeaturePreprocessor)

    def test_create_preprocessor_unknown_type(self):
        """Test creating preprocessor with unknown type."""
        config = {"type": "unknown"}

        with pytest.raises(PreprocessingError) as exc_info:
            create_preprocessor(config)

        assert "Unknown preprocessor type" in str(exc_info.value)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_to_feature_matrix(self):
        """Test to_feature_matrix utility function."""
        observations = [
            Observation(speed=20.0, steering=0.0, sensors=[1.0, 2.0]),
            Observation(speed=25.0, steering=0.1, sensors=[3.0, 4.0]),
        ]

        features = to_feature_matrix(observations)

        assert isinstance(features, np.ndarray)
        assert features.shape == (2, 4)  # 2 observations, 4 features
        assert features[0, 0] == 20.0  # First speed
        assert features[1, 1] == 0.1  # Second steering

    def test_validate_preprocessing_parity_success(self):
        """Test successful preprocessing parity validation."""
        observations = [
            Observation(speed=20.0, steering=0.0, sensors=[1.0, 2.0]),
            Observation(speed=25.0, steering=0.1, sensors=[3.0, 4.0]),
        ]

        # Create two identical preprocessors
        train_preprocessor = MinimalPreprocessor()
        serve_preprocessor = MinimalPreprocessor()

        train_preprocessor.fit(observations)
        serve_preprocessor.fit(observations)

        result = validate_preprocessing_parity(train_preprocessor, serve_preprocessor, observations)

        assert result is True

    def test_validate_preprocessing_parity_shape_mismatch(self):
        """Test parity validation with shape mismatch."""
        observations = [Observation(speed=20.0, steering=0.0, sensors=[1.0, 2.0])]

        # Create preprocessors that will produce different shapes
        train_preprocessor = Mock()
        serve_preprocessor = Mock()

        train_preprocessor.transform.return_value = np.array([[1, 2, 3]])
        serve_preprocessor.transform.return_value = np.array([[1, 2]])  # Different shape

        with pytest.raises(PreprocessingError) as exc_info:
            validate_preprocessing_parity(train_preprocessor, serve_preprocessor, observations)

        assert "Shape mismatch" in str(exc_info.value)

    def test_validate_preprocessing_parity_value_mismatch(self):
        """Test parity validation with value mismatch."""
        observations = [Observation(speed=20.0, steering=0.0, sensors=[1.0, 2.0])]

        # Create preprocessors that produce different values
        train_preprocessor = Mock()
        serve_preprocessor = Mock()

        train_preprocessor.transform.return_value = np.array([[1.0, 2.0, 3.0]])
        serve_preprocessor.transform.return_value = np.array([[1.0, 2.0, 3.1]])  # Slight difference

        with pytest.raises(PreprocessingError) as exc_info:
            validate_preprocessing_parity(
                train_preprocessor, serve_preprocessor, observations, tolerance=1e-6
            )

        assert "parity validation failed" in str(exc_info.value)

    def test_validate_preprocessing_parity_within_tolerance(self):
        """Test parity validation within tolerance."""
        observations = [Observation(speed=20.0, steering=0.0, sensors=[1.0, 2.0])]

        # Create preprocessors with small difference within tolerance
        train_preprocessor = Mock()
        serve_preprocessor = Mock()

        train_preprocessor.transform.return_value = np.array([[1.0, 2.0, 3.0]])
        serve_preprocessor.transform.return_value = np.array(
            [[1.0, 2.0, 3.000001]]
        )  # Very small diff

        result = validate_preprocessing_parity(
            train_preprocessor, serve_preprocessor, observations, tolerance=1e-5
        )

        assert result is True


class TestPreprocessingEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_preprocessing_with_nan_values(self):
        """Test preprocessing with NaN values in observations."""
        observations = [Observation(speed=float("nan"), steering=0.0, sensors=[1.0, 2.0])]

        preprocessor = StandardFeaturePreprocessor()

        # Should handle NaN gracefully or raise appropriate error
        with pytest.raises(PreprocessingError):
            preprocessor.fit_transform(observations)

    def test_preprocessing_with_infinite_values(self):
        """Test preprocessing with infinite values."""
        observations = [Observation(speed=float("inf"), steering=0.0, sensors=[1.0, 2.0])]

        preprocessor = StandardFeaturePreprocessor()

        # Should handle infinity gracefully or raise appropriate error
        with pytest.raises(PreprocessingError):
            preprocessor.fit_transform(observations)

    def test_preprocessing_with_empty_sensors(self):
        """Test preprocessing with empty sensor arrays."""
        # This should be caught by Pydantic validation, but test graceful handling
        observations = [
            Observation(speed=20.0, steering=0.0, sensors=[1.0])  # At least one sensor required
        ]

        preprocessor = MinimalPreprocessor()
        features = preprocessor.transform(observations)

        # Should handle gracefully
        assert features.shape == (1, 3)  # speed + steering + 1 sensor

    def test_preprocessing_large_batch(self):
        """Test preprocessing with large batch size."""
        # Create large batch
        observations = []
        for i in range(1000):
            obs = Observation(
                speed=20.0 + i * 0.01,
                steering=(i % 100) * 0.01,
                sensors=[0.1 * j for j in range(5)],
            )
            observations.append(obs)

        preprocessor = StandardFeaturePreprocessor()
        features = preprocessor.fit_transform(observations)

        assert features.shape == (1000, 7)  # speed + steering + 5 sensors
        assert features.dtype == np.float32
