"""
Unit tests for artifact integrity validation system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.versioning.integrity_validator import (
    IntegrityValidationError,
    IntegrityValidator,
    ModelLoaderIntegrityMixin,
)
from src.versioning.artifact_manager import ArtifactManager


class TestIntegrityValidator:
    """Test IntegrityValidator functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def artifact_manager(self, temp_dir):
        """Create ArtifactManager instance for testing."""
        return ArtifactManager(temp_dir / "artifacts")

    @pytest.fixture
    def integrity_validator(self, artifact_manager):
        """Create IntegrityValidator instance for testing."""
        return IntegrityValidator(artifact_manager)

    @pytest.fixture
    def sample_artifacts(self, temp_dir):
        """Create sample artifact files for testing."""
        artifacts_dir = temp_dir / "sample_artifacts"
        artifacts_dir.mkdir()

        # Create subdirectory first
        subdir = artifacts_dir / "subdir"
        subdir.mkdir()

        # Create sample files
        (artifacts_dir / "model.pt").write_text("model data")
        (artifacts_dir / "config.yaml").write_text("config: test")
        (subdir / "preprocessor.pkl").write_text("preprocessor data")

        return artifacts_dir

    @pytest.fixture
    def pinned_artifacts(self, artifact_manager, sample_artifacts):
        """Create pinned artifacts for testing."""
        manifest = artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)
        return manifest

    def test_integrity_validator_initialization(self, artifact_manager):
        """Test IntegrityValidator initialization."""
        validator = IntegrityValidator(artifact_manager)

        assert validator.artifact_manager == artifact_manager
        assert validator._validation_cache == {}

    def test_validate_model_artifacts_success(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test successful model artifact validation."""
        is_valid, report = integrity_validator.validate_model_artifacts("v1.2.3", sample_artifacts)

        assert is_valid is True
        assert report["version"] == "v1.2.3"
        assert report["total_artifacts"] == 3
        assert report["valid_artifacts"] == 3
        assert report["invalid_artifacts"] == 0
        assert report["missing_artifacts"] == 0
        assert len(report["errors"]) == 0
        assert report["is_valid"] is True

    def test_validate_model_artifacts_with_required_artifacts(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validation with specific required artifacts."""
        required_artifacts = ["model.pt", "config.yaml"]

        is_valid, report = integrity_validator.validate_model_artifacts(
            "v1.2.3", sample_artifacts, required_artifacts
        )

        assert is_valid is True
        assert report["total_artifacts"] == 2
        assert report["valid_artifacts"] == 2
        assert "model.pt" in report["artifacts"]
        assert "config.yaml" in report["artifacts"]
        assert "subdir/preprocessor.pkl" not in report["artifacts"]

    def test_validate_model_artifacts_missing_file(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validation with missing file."""
        # Remove a file
        (sample_artifacts / "model.pt").unlink()

        with pytest.raises(IntegrityValidationError) as exc_info:
            integrity_validator.validate_model_artifacts("v1.2.3", sample_artifacts)

        assert "Artifact file missing" in str(exc_info.value)
        assert exc_info.value.version == "v1.2.3"
        assert "model.pt" in exc_info.value.failed_artifacts

    def test_validate_model_artifacts_corrupted_file(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validation with corrupted file."""
        # Corrupt a file
        (sample_artifacts / "model.pt").write_text("corrupted data")

        with pytest.raises(IntegrityValidationError) as exc_info:
            integrity_validator.validate_model_artifacts("v1.2.3", sample_artifacts)

        assert "Hash mismatch" in str(exc_info.value)
        assert exc_info.value.version == "v1.2.3"
        assert "model.pt" in exc_info.value.failed_artifacts

    def test_validate_model_artifacts_nonexistent_version(
        self, integrity_validator, sample_artifacts
    ):
        """Test validation with nonexistent version."""
        with pytest.raises(IntegrityValidationError) as exc_info:
            integrity_validator.validate_model_artifacts("v999.999.999", sample_artifacts)

        assert "No manifest found" in str(exc_info.value)
        assert exc_info.value.version == "v999.999.999"

    def test_validate_model_artifacts_non_strict_mode(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validation in non-strict mode."""
        # Corrupt a file
        (sample_artifacts / "model.pt").write_text("corrupted data")

        is_valid, report = integrity_validator.validate_model_artifacts(
            "v1.2.3", sample_artifacts, strict_mode=False
        )

        assert is_valid is False
        assert report["invalid_artifacts"] == 1
        assert len(report["errors"]) > 0
        assert report["artifacts"]["model.pt"]["status"] == "invalid"

    def test_validate_required_artifacts(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validate_required_artifacts method."""
        required_artifacts = ["model.pt", "config.yaml"]

        is_valid, report = integrity_validator.validate_required_artifacts(
            "v1.2.3", sample_artifacts, required_artifacts
        )

        assert is_valid is True
        assert report["total_artifacts"] == 2
        assert report["valid_artifacts"] == 2

    def test_quick_validation_success(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test quick validation success."""
        critical_artifacts = ["model.pt"]

        result = integrity_validator.quick_validation(
            "v1.2.3", sample_artifacts, critical_artifacts
        )

        assert result is True

    def test_quick_validation_failure(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test quick validation failure."""
        # Corrupt a file
        (sample_artifacts / "model.pt").write_text("corrupted data")

        critical_artifacts = ["model.pt"]

        result = integrity_validator.quick_validation(
            "v1.2.3", sample_artifacts, critical_artifacts
        )

        assert result is False

    def test_get_validation_summary(self, integrity_validator):
        """Test validation summary generation."""
        report = {
            "version": "v1.2.3",
            "total_artifacts": 3,
            "valid_artifacts": 2,
            "invalid_artifacts": 1,
            "missing_artifacts": 0,
            "errors": ["Hash mismatch for model.pt"],
        }

        summary = integrity_validator.get_validation_summary(report)

        assert "v1.2.3" in summary
        assert "2/3 artifacts valid" in summary
        assert "1 invalid" in summary

    def test_validation_cache(self, integrity_validator, sample_artifacts, pinned_artifacts):
        """Test validation caching."""
        # First validation
        is_valid1, _ = integrity_validator.validate_model_artifacts("v1.2.3", sample_artifacts)
        assert is_valid1 is True

        # Second validation should use cache
        is_valid2, report2 = integrity_validator.validate_model_artifacts(
            "v1.2.3", sample_artifacts
        )
        assert is_valid2 is True
        assert report2.get("cached") is True

        # Check cache stats
        stats = integrity_validator.get_cache_stats()
        assert stats["cached_validations"] == 1

    def test_clear_validation_cache(self, integrity_validator, sample_artifacts, pinned_artifacts):
        """Test clearing validation cache."""
        # Validate to populate cache
        integrity_validator.validate_model_artifacts("v1.2.3", sample_artifacts)

        # Check cache is populated
        stats = integrity_validator.get_cache_stats()
        assert stats["cached_validations"] == 1

        # Clear cache
        integrity_validator.clear_validation_cache()

        # Check cache is empty
        stats = integrity_validator.get_cache_stats()
        assert stats["cached_validations"] == 0

    def test_validation_with_missing_required_artifacts(
        self, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validation with missing required artifacts."""
        required_artifacts = ["model.pt", "nonexistent.pt"]

        with pytest.raises(IntegrityValidationError) as exc_info:
            integrity_validator.validate_model_artifacts(
                "v1.2.3", sample_artifacts, required_artifacts
            )

        assert "Required artifact not found in manifest" in str(exc_info.value)
        assert "nonexistent.pt" in exc_info.value.failed_artifacts


class TestModelLoaderIntegrityMixin:
    """Test ModelLoaderIntegrityMixin functionality."""

    class TestModelLoader(ModelLoaderIntegrityMixin):
        """Test model loader class."""

        def __init__(self):
            super().__init__()
            self.loaded_models = {}

    @pytest.fixture
    def test_loader(self):
        """Create test model loader instance."""
        return self.TestModelLoader()

    @pytest.fixture
    def mock_validator(self):
        """Create mock integrity validator."""
        validator = Mock(spec=IntegrityValidator)
        validator.validate_required_artifacts.return_value = (True, {"is_valid": True})
        return validator

    def test_mixin_initialization(self, test_loader):
        """Test mixin initialization."""
        assert test_loader._integrity_validator is None
        assert test_loader._validation_enabled is True

    def test_set_integrity_validator(self, test_loader, mock_validator):
        """Test setting integrity validator."""
        test_loader.set_integrity_validator(mock_validator)

        assert test_loader._integrity_validator == mock_validator

    def test_enable_integrity_validation(self, test_loader):
        """Test enabling/disabling integrity validation."""
        test_loader.enable_integrity_validation(False)
        assert test_loader._validation_enabled is False

        test_loader.enable_integrity_validation(True)
        assert test_loader._validation_enabled is True

    def test_validate_before_loading_success(self, test_loader, mock_validator):
        """Test successful validation before loading."""
        test_loader.set_integrity_validator(mock_validator)

        result = test_loader.validate_before_loading(
            "v1.2.3", Path("/test/artifacts"), ["model.pt"]
        )

        assert result is True
        mock_validator.validate_required_artifacts.assert_called_once()

    def test_validate_before_loading_disabled(self, test_loader, mock_validator):
        """Test validation when disabled."""
        test_loader.set_integrity_validator(mock_validator)
        test_loader.enable_integrity_validation(False)

        result = test_loader.validate_before_loading(
            "v1.2.3", Path("/test/artifacts"), ["model.pt"]
        )

        assert result is True
        mock_validator.validate_required_artifacts.assert_not_called()

    def test_validate_before_loading_no_validator(self, test_loader):
        """Test validation with no validator set."""
        result = test_loader.validate_before_loading(
            "v1.2.3", Path("/test/artifacts"), ["model.pt"]
        )

        assert result is True

    def test_validate_before_loading_failure_strict(self, test_loader, mock_validator):
        """Test validation failure in strict mode."""
        test_loader.set_integrity_validator(mock_validator)
        mock_validator.validate_required_artifacts.side_effect = IntegrityValidationError(
            "Validation failed", "v1.2.3", ["model.pt"]
        )

        with pytest.raises(IntegrityValidationError):
            test_loader.validate_before_loading(
                "v1.2.3", Path("/test/artifacts"), ["model.pt"], strict_mode=True
            )

    def test_validate_before_loading_failure_non_strict(self, test_loader, mock_validator):
        """Test validation failure in non-strict mode."""
        test_loader.set_integrity_validator(mock_validator)
        mock_validator.validate_required_artifacts.return_value = (False, {"is_valid": False})

        result = test_loader.validate_before_loading(
            "v1.2.3", Path("/test/artifacts"), ["model.pt"], strict_mode=False
        )

        assert result is False

    def test_validate_before_loading_exception_non_strict(self, test_loader, mock_validator):
        """Test exception handling in non-strict mode."""
        test_loader.set_integrity_validator(mock_validator)
        mock_validator.validate_required_artifacts.side_effect = Exception("Unexpected error")

        result = test_loader.validate_before_loading(
            "v1.2.3", Path("/test/artifacts"), ["model.pt"], strict_mode=False
        )

        assert result is False
