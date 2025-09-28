"""
Unit tests for artifact management and integrity validation system.
"""

import tempfile
from pathlib import Path

import pytest

from src.versioning.artifact_manager import (
    ArtifactIntegrityError,
    ArtifactManager,
    ArtifactManifest,
)
from src.versioning.semantic_version import SemanticVersion


class TestArtifactManifest:
    """Test ArtifactManifest functionality."""

    def test_manifest_creation(self):
        """Test creating a manifest."""
        manifest = ArtifactManifest(
            version="v1.2.3",
            artifacts={"model.pt": "abc123", "config.yaml": "def456"},
            model_type="policy",
            description="Test model",
        )

        assert manifest.version == "v1.2.3"
        assert manifest.artifacts == {"model.pt": "abc123", "config.yaml": "def456"}
        assert manifest.model_type == "policy"
        assert manifest.description == "Test model"

    def test_manifest_to_dict(self):
        """Test converting manifest to dictionary."""
        manifest = ArtifactManifest(
            version="v1.2.3", artifacts={"model.pt": "abc123"}, model_type="policy"
        )

        data = manifest.to_dict()

        assert data["version"] == "v1.2.3"
        assert data["artifacts"] == {"model.pt": "abc123"}
        assert data["model_type"] == "policy"
        assert "created_at" in data
        assert "dependencies" in data
        assert "metadata" in data

    def test_manifest_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "version": "v1.2.3",
            "artifacts": {"model.pt": "abc123"},
            "model_type": "policy",
            "description": "Test model",
            "dependencies": ["torch"],
            "metadata": {"author": "test"},
        }

        manifest = ArtifactManifest.from_dict(data)

        assert manifest.version == "v1.2.3"
        assert manifest.artifacts == {"model.pt": "abc123"}
        assert manifest.model_type == "policy"
        assert manifest.description == "Test model"
        assert manifest.dependencies == ["torch"]
        assert manifest.metadata == {"author": "test"}


class TestArtifactManager:
    """Test ArtifactManager functionality."""

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

    def test_artifact_manager_initialization(self, temp_dir):
        """Test ArtifactManager initialization."""
        manager = ArtifactManager(temp_dir / "artifacts")

        assert manager.artifacts_dir.exists()
        assert manager.manifests_dir.exists()
        assert manager.versions_dir.exists()

    def test_calculate_file_hash(self, artifact_manager, sample_artifacts):
        """Test file hash calculation."""
        model_file = sample_artifacts / "model.pt"
        hash_value = artifact_manager.calculate_file_hash(model_file)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length
        assert hash_value.isalnum()

    def test_calculate_file_hash_nonexistent(self, artifact_manager):
        """Test hash calculation for nonexistent file."""
        nonexistent_file = Path("nonexistent.txt")

        with pytest.raises(ArtifactIntegrityError) as exc_info:
            artifact_manager.calculate_file_hash(nonexistent_file)

        assert "Cannot calculate hash" in str(exc_info.value)
        assert exc_info.value.artifact_path == nonexistent_file

    def test_pin_artifacts(self, artifact_manager, sample_artifacts):
        """Test pinning artifacts."""
        manifest = artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        assert manifest.version == "v1.2.3"
        assert len(manifest.artifacts) == 3  # model.pt, config.yaml, subdir/preprocessor.pkl
        assert "model.pt" in manifest.artifacts
        assert "config.yaml" in manifest.artifacts
        assert "subdir/preprocessor.pkl" in manifest.artifacts

        # Check that hashes are valid
        for artifact_path, hash_value in manifest.artifacts.items():
            assert isinstance(hash_value, str)
            assert len(hash_value) == 64

    def test_pin_artifacts_nonexistent_dir(self, artifact_manager):
        """Test pinning artifacts from nonexistent directory."""
        nonexistent_dir = Path("nonexistent")

        with pytest.raises(ArtifactIntegrityError) as exc_info:
            artifact_manager.pin_artifacts("v1.2.3", nonexistent_dir)

        assert "does not exist" in str(exc_info.value)

    def test_pin_artifacts_empty_dir(self, artifact_manager, temp_dir):
        """Test pinning artifacts from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        with pytest.raises(ArtifactIntegrityError) as exc_info:
            artifact_manager.pin_artifacts("v1.2.3", empty_dir)

        assert "No artifact files found" in str(exc_info.value)

    def test_validate_artifacts(self, artifact_manager, sample_artifacts):
        """Test artifact validation."""
        # First pin the artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Then validate them
        result = artifact_manager.validate_artifacts("v1.2.3", sample_artifacts)

        assert result is True

    def test_validate_artifacts_missing_file(self, artifact_manager, sample_artifacts):
        """Test validation with missing file."""
        # Pin artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Remove a file
        (sample_artifacts / "model.pt").unlink()

        with pytest.raises(ArtifactIntegrityError) as exc_info:
            artifact_manager.validate_artifacts("v1.2.3", sample_artifacts)

        assert "Artifact file missing" in str(exc_info.value)
        assert exc_info.value.artifact_path == sample_artifacts / "model.pt"

    def test_validate_artifacts_corrupted_file(self, artifact_manager, sample_artifacts):
        """Test validation with corrupted file."""
        # Pin artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Corrupt a file
        (sample_artifacts / "model.pt").write_text("corrupted data")

        with pytest.raises(ArtifactIntegrityError) as exc_info:
            artifact_manager.validate_artifacts("v1.2.3", sample_artifacts)

        assert "Artifact integrity check failed" in str(exc_info.value)
        assert exc_info.value.artifact_path == sample_artifacts / "model.pt"

    def test_validate_artifacts_nonexistent_version(self, artifact_manager, sample_artifacts):
        """Test validation for nonexistent version."""
        with pytest.raises(ArtifactIntegrityError) as exc_info:
            artifact_manager.validate_artifacts("v999.999.999", sample_artifacts)

        assert "No manifest found" in str(exc_info.value)

    def test_get_artifact_hash(self, artifact_manager, sample_artifacts):
        """Test getting artifact hash."""
        # Pin artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Get hash for specific artifact
        hash_value = artifact_manager.get_artifact_hash("v1.2.3", "model.pt")

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

        # Test nonexistent artifact
        hash_value = artifact_manager.get_artifact_hash("v1.2.3", "nonexistent.pt")
        assert hash_value is None

    def test_list_artifacts(self, artifact_manager, sample_artifacts):
        """Test listing artifacts."""
        # Pin artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # List artifacts
        artifacts = artifact_manager.list_artifacts("v1.2.3")

        assert len(artifacts) == 3
        assert "model.pt" in artifacts
        assert "config.yaml" in artifacts
        assert "subdir/preprocessor.pkl" in artifacts

    def test_get_manifest(self, artifact_manager, sample_artifacts):
        """Test getting manifest."""
        # Pin artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Get manifest
        manifest = artifact_manager.get_manifest("v1.2.3")

        assert manifest is not None
        assert manifest.version == "v1.2.3"
        assert len(manifest.artifacts) == 3

    def test_list_versions(self, artifact_manager, sample_artifacts):
        """Test listing versions."""
        # Pin multiple versions
        artifact_manager.pin_artifacts("v1.0.0", sample_artifacts)
        artifact_manager.pin_artifacts("v1.1.0", sample_artifacts)
        artifact_manager.pin_artifacts("v2.0.0", sample_artifacts)

        # List versions
        versions = artifact_manager.list_versions()

        assert len(versions) == 3
        assert all(isinstance(v, SemanticVersion) for v in versions)
        assert versions[0] == SemanticVersion(1, 0, 0)
        assert versions[1] == SemanticVersion(1, 1, 0)
        assert versions[2] == SemanticVersion(2, 0, 0)

    def test_delete_version(self, artifact_manager, sample_artifacts):
        """Test deleting version."""
        # Pin artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Verify version exists
        assert artifact_manager.get_manifest("v1.2.3") is not None

        # Delete version
        result = artifact_manager.delete_version("v1.2.3")

        assert result is True
        assert artifact_manager.get_manifest("v1.2.3") is None

    def test_copy_artifacts(self, artifact_manager, sample_artifacts):
        """Test copying artifacts to versioned storage."""
        # Copy artifacts
        version_dir = artifact_manager.copy_artifacts("v1.2.3", sample_artifacts)

        assert version_dir.exists()
        assert (version_dir / "model.pt").exists()
        assert (version_dir / "config.yaml").exists()
        assert (version_dir / "subdir" / "preprocessor.pkl").exists()

    def test_copy_artifacts_with_validation(self, artifact_manager, sample_artifacts):
        """Test copying artifacts with validation."""
        # First pin the artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Then copy artifacts with validation
        version_dir = artifact_manager.copy_artifacts("v1.2.3", sample_artifacts, validate=True)

        assert version_dir.exists()

        # Verify artifacts were copied
        assert (version_dir / "model.pt").exists()
        assert (version_dir / "config.yaml").exists()
        assert (version_dir / "subdir" / "preprocessor.pkl").exists()

    def test_verify_integrity(self, artifact_manager, sample_artifacts):
        """Test integrity verification."""
        # Pin and copy artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)
        artifact_manager.copy_artifacts("v1.2.3", sample_artifacts)

        # Verify integrity
        results = artifact_manager.verify_integrity("v1.2.3")

        assert len(results) == 3
        assert all(results.values())  # All should be valid

    def test_get_integrity_report(self, artifact_manager, sample_artifacts):
        """Test integrity report generation."""
        # Pin and copy artifacts
        artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)
        artifact_manager.copy_artifacts("v1.2.3", sample_artifacts)

        # Get integrity report
        report = artifact_manager.get_integrity_report("v1.2.3")

        assert report["version"] == "v1.2.3"
        assert report["status"] == "valid"
        assert report["total_artifacts"] == 3
        assert report["valid_artifacts"] == 3
        assert report["invalid_artifacts"] == 0
        assert "artifacts" in report
        assert "manifest" in report

    def test_get_integrity_report_nonexistent_version(self, artifact_manager):
        """Test integrity report for nonexistent version."""
        report = artifact_manager.get_integrity_report("v999.999.999")

        assert report["version"] == "v999.999.999"
        assert report["status"] == "not_found"
        assert "No manifest found" in report["message"]

    def test_find_artifact_files(self, artifact_manager, sample_artifacts):
        """Test finding artifact files."""
        files = artifact_manager._find_artifact_files(sample_artifacts)

        assert len(files) == 3
        assert all(f.suffix.lower() in {".pt", ".yaml", ".pkl"} for f in files)

    def test_manifest_persistence(self, artifact_manager, sample_artifacts):
        """Test that manifests are persisted correctly."""
        # Pin artifacts
        manifest = artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)

        # Create new manager instance
        new_manager = ArtifactManager(artifact_manager.artifacts_dir)

        # Load manifest
        loaded_manifest = new_manager.get_manifest("v1.2.3")

        assert loaded_manifest is not None
        assert loaded_manifest.version == manifest.version
        assert loaded_manifest.artifacts == manifest.artifacts
