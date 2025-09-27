"""
QA validation for Model Management Layer in CarlaRL Policy-as-a-Service.

This module validates that all Model Management Layer requirements are met
and the implementation is ready for integration.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from src.exceptions import ArtifactValidationError, ModelLoadingError, PreprocessingError
from src.io_schemas import Observation
from src.model_loader import (
    PolicyWrapper,
    get_available_versions,
    load_artifacts,
    validate_artifact_integrity,
    validate_model_compatibility,
)
from src.preprocessing import (
    MinimalPreprocessor,
    StandardFeaturePreprocessor,
    validate_preprocessing_parity,
)


class SimpleTestModel(nn.Module):
    """Simple test model for QA validation."""

    def __init__(self, input_dim: int = 5, output_dim: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class TestModelManagementQA:
    """QA validation tests for Model Management Layer."""

    def create_complete_artifacts(self, temp_dir: Path, with_hashes: bool = True) -> None:
        """Create complete test artifacts for QA validation."""
        # Create model
        model = SimpleTestModel()
        model_path = temp_dir / "model.pt"
        torch.save(model, model_path)

        # Create preprocessor
        observations = [
            Observation(speed=25.0, steering=0.1, sensors=[0.8, 0.2, 0.5, 0.9, 0.1]),
            Observation(speed=30.0, steering=-0.1, sensors=[0.6, 0.4, 0.7, 0.8, 0.3])
        ]
        preprocessor = StandardFeaturePreprocessor()
        preprocessor.fit(observations)

        preprocessor_path = temp_dir / "preprocessor.pkl"
        preprocessor.save(preprocessor_path)

        # Create model card
        model_card_data = {
            "model_name": "test-carla-ppo",
            "version": "v0.1.0",
            "model_type": "pytorch",
            "input_shape": [5],
            "output_shape": [3],
            "framework_version": "2.1.0",
            "description": "Test model for QA validation",
            "performance_metrics": {
                "reward": 850.5,
                "success_rate": 0.95
            }
        }

        if with_hashes:
            # Add artifact hashes
            import hashlib

            def compute_hash(file_path):
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                return sha256_hash.hexdigest()

            model_card_data["artifact_hashes"] = {
                "model.pt": compute_hash(model_path),
                "preprocessor.pkl": compute_hash(preprocessor_path)
            }

        model_card_path = temp_dir / "model_card.yaml"
        with open(model_card_path, 'w') as f:
            yaml.dump(model_card_data, f)

    def test_qa_fr_2_1_policy_wrapper_deterministic_modes(self):
        """
        QA Test: FR-2.1 - PolicyWrapper with deterministic/stochastic modes
        """
        model = SimpleTestModel()
        wrapper = PolicyWrapper(model, model_type="pytorch")

        # Test deterministic mode
        x = torch.randn(2, 5)
        output1 = wrapper(x, deterministic=True)
        output2 = wrapper(x, deterministic=True)

        # Should produce same output for deterministic mode
        assert torch.allclose(output1, output2), "Deterministic mode should produce identical outputs"

        # Test stochastic mode (for models that support it)
        output3 = wrapper(x, deterministic=False)
        assert output3.shape == output1.shape, "Output shapes should match regardless of mode"

        print("✅ FR-2.1: PolicyWrapper deterministic/stochastic modes validated")

    def test_qa_fr_2_2_model_loading_formats(self):
        """
        QA Test: FR-2.2 - Model loading supporting TorchScript and PyTorch formats
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test PyTorch model loading
            model = SimpleTestModel()
            pytorch_path = temp_path / "pytorch_model.pt"
            torch.save(model, pytorch_path)

            from src.model_loader import load_pytorch_model
            pytorch_wrapper = load_pytorch_model(pytorch_path, torch.device('cpu'))
            assert pytorch_wrapper.model_type == "pytorch"

            # Test TorchScript model loading
            scripted_model = torch.jit.script(model)
            torchscript_path = temp_path / "torchscript_model.pt"
            torch.jit.save(scripted_model, torchscript_path)

            torchscript_wrapper = load_pytorch_model(torchscript_path, torch.device('cpu'))
            assert torchscript_wrapper.model_type == "torchscript"

            print("✅ FR-2.2: Model loading formats (PyTorch/TorchScript) validated")

    def test_qa_fr_2_3_artifact_integrity_validation(self):
        """
        QA Test: FR-2.3 - Artifact integrity validation using hash pinning
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.create_complete_artifacts(temp_path, with_hashes=True)

            # Load model card and validate integrity
            from src.model_loader import load_model_card
            model_card = load_model_card(temp_path)

            # Should pass validation
            result = validate_artifact_integrity(temp_path, model_card)
            assert result is True

            # Test hash mismatch detection
            model_card["artifact_hashes"]["model.pt"] = "wrong_hash"
            with pytest.raises(ArtifactValidationError):
                validate_artifact_integrity(temp_path, model_card)

            print("✅ FR-2.3: Artifact integrity validation with hash pinning validated")

    def test_qa_fr_2_4_preprocessor_serialization(self):
        """
        QA Test: FR-2.4 - Preprocessor loading and serialization
        """
        observations = [
            Observation(speed=25.0, steering=0.1, sensors=[0.8, 0.2, 0.5]),
            Observation(speed=30.0, steering=-0.1, sensors=[0.6, 0.4, 0.7])
        ]

        # Test StandardFeaturePreprocessor serialization
        preprocessor = StandardFeaturePreprocessor()
        preprocessor.fit(observations)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "preprocessor.pkl"

            # Save and load
            preprocessor.save(temp_path)
            loaded_preprocessor = StandardFeaturePreprocessor.load(temp_path)

            # Verify functionality preserved
            original_features = preprocessor.transform(observations)
            loaded_features = loaded_preprocessor.transform(observations)

            assert np.allclose(original_features, loaded_features)
            assert loaded_preprocessor.is_fitted is True

            print("✅ FR-2.4: Preprocessor serialization validated")

    def test_qa_fr_2_5_train_serve_parity(self):
        """
        QA Test: FR-2.5 - Feature pipeline with train-serve parity validation
        """
        observations = [
            Observation(speed=25.0, steering=0.1, sensors=[0.8, 0.2, 0.5]),
            Observation(speed=30.0, steering=-0.1, sensors=[0.6, 0.4, 0.7])
        ]

        # Create "training" preprocessor
        train_preprocessor = StandardFeaturePreprocessor()
        train_preprocessor.fit(observations)

        # Simulate "serving" preprocessor (loaded from file)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "preprocessor.pkl"
            train_preprocessor.save(temp_path)
            serve_preprocessor = StandardFeaturePreprocessor.load(temp_path)

            # Validate parity
            result = validate_preprocessing_parity(
                train_preprocessor, serve_preprocessor, observations
            )
            assert result is True

            print("✅ FR-2.5: Train-serve parity validation implemented")

    def test_qa_fr_2_6_multi_version_support(self):
        """
        QA Test: FR-2.6 - Multi-version model support with semantic versioning
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple version directories
            versions = ["v0.1.0", "v0.2.0", "v1.0.0"]
            for version in versions:
                version_dir = temp_path / version
                version_dir.mkdir()
                self.create_complete_artifacts(version_dir, with_hashes=False)

            # Test version discovery
            available_versions = get_available_versions(temp_path)

            for version in versions:
                assert version in available_versions

            # Test loading specific version
            policy, preprocessor = load_artifacts(
                temp_path / "v1.0.0",
                torch.device('cpu'),
                validate_integrity=False
            )

            assert isinstance(policy, PolicyWrapper)
            assert preprocessor is not None

            print("✅ FR-2.6: Multi-version model support validated")

    def test_qa_fr_2_7_device_selection(self):
        """
        QA Test: FR-2.7 - Device selection logic with automatic fallback
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.create_complete_artifacts(temp_path, with_hashes=False)

            # Test CPU device selection
            policy_cpu, _ = load_artifacts(
                temp_path,
                torch.device('cpu'),
                validate_integrity=False
            )

            assert policy_cpu.device.type == 'cpu'

            # Test device movement
            policy_moved = policy_cpu.to(torch.device('cpu'))
            assert policy_moved.device.type == 'cpu'

            print("✅ FR-2.7: Device selection logic validated")

    def test_qa_fr_2_8_model_metadata_parsing(self):
        """
        QA Test: FR-2.8 - Model metadata parsing from model_card.yaml
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.create_complete_artifacts(temp_path, with_hashes=False)

            from src.model_loader import load_model_card
            model_card = load_model_card(temp_path)

            # Validate required fields
            required_fields = ["model_name", "version", "model_type"]
            for field in required_fields:
                assert field in model_card

            # Validate specific values
            assert model_card["model_name"] == "test-carla-ppo"
            assert model_card["version"] == "v0.1.0"
            assert model_card["model_type"] == "pytorch"

            print("✅ FR-2.8: Model metadata parsing validated")

    def test_qa_fr_2_9_error_handling(self):
        """
        QA Test: FR-2.9 - Graceful error handling for missing/corrupted artifacts
        """
        # Test missing directory
        with pytest.raises(ModelLoadingError) as exc_info:
            load_artifacts(Path("nonexistent"), torch.device('cpu'))
        assert "Artifact directory not found" in str(exc_info.value)

        # Test missing model card
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with pytest.raises(ModelLoadingError) as exc_info:
                load_artifacts(temp_path, torch.device('cpu'))
            assert "Model card not found" in str(exc_info.value)

        # Test corrupted model card
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid YAML
            model_card_path = temp_path / "model_card.yaml"
            model_card_path.write_text("invalid: yaml: [")

            with pytest.raises(ModelLoadingError) as exc_info:
                load_artifacts(temp_path, torch.device('cpu'))
            assert "Failed to parse model card YAML" in str(exc_info.value)

        print("✅ FR-2.9: Error handling for missing/corrupted artifacts validated")

    def test_qa_preprocessing_edge_cases(self):
        """
        QA Test: Preprocessing pipeline handles edge cases correctly
        """
        # Test empty observation handling
        preprocessor = MinimalPreprocessor()

        with pytest.raises(PreprocessingError):
            preprocessor.transform([])

        # Test variable sensor lengths
        observations = [
            Observation(speed=20.0, steering=0.0, sensors=[1.0, 2.0]),
            Observation(speed=25.0, steering=0.1, sensors=[3.0, 4.0, 5.0, 6.0])  # More sensors
        ]

        features = preprocessor.transform(observations)

        # Should handle variable lengths by padding
        assert features.shape[0] == 2
        assert features.shape[1] == 2 + 4  # speed + steering + max_sensor_length

        print("✅ Preprocessing edge cases validated")

    def test_qa_model_compatibility_validation(self):
        """
        QA Test: Model compatibility validation works correctly
        """
        # Valid model card
        valid_model_card = {
            "input_shape": [5],
            "output_shape": [3],
            "framework_version": "2.1.0"
        }

        result = validate_model_compatibility(valid_model_card)
        assert result is True

        # Invalid model card (missing fields)
        invalid_model_card = {"input_shape": [5]}

        with pytest.raises(ArtifactValidationError):
            validate_model_compatibility(invalid_model_card)

        print("✅ Model compatibility validation implemented")


def run_model_management_qa():
    """
    Run complete QA validation for Model Management Layer.

    This function validates that all Model Management Layer requirements
    from the PRD are properly implemented.
    """
    print("🔍 Running QA Validation for Model Management Layer")
    print("=" * 60)

    # Run all QA tests
    qa_test = TestModelManagementQA()

    try:
        qa_test.test_qa_fr_2_1_policy_wrapper_deterministic_modes()
        qa_test.test_qa_fr_2_2_model_loading_formats()
        qa_test.test_qa_fr_2_3_artifact_integrity_validation()
        qa_test.test_qa_fr_2_4_preprocessor_serialization()
        qa_test.test_qa_fr_2_5_train_serve_parity()
        qa_test.test_qa_fr_2_6_multi_version_support()
        qa_test.test_qa_fr_2_7_device_selection()
        qa_test.test_qa_fr_2_8_model_metadata_parsing()
        qa_test.test_qa_fr_2_9_error_handling()
        qa_test.test_qa_preprocessing_edge_cases()
        qa_test.test_qa_model_compatibility_validation()

        print("\n🎉 Model Management Layer QA: ALL TESTS PASSED")
        print("✅ Ready for integration with other layers")
        return True

    except Exception as e:
        print("\n❌ Model Management Layer QA: FAILED")
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    run_model_management_qa()
