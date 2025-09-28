"""
Unit tests for artifact validation CLI tool.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.versioning.artifact_manager import ArtifactManager
from src.versioning.integrity_validator import IntegrityValidator


class TestValidateArtifactsCLI:
    """Test artifact validation CLI functionality."""

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

        (artifacts_dir / "model.pt").write_text("model data")
        (artifacts_dir / "config.yaml").write_text("config: test")

        return artifacts_dir

    @pytest.fixture
    def pinned_artifacts(self, artifact_manager, sample_artifacts):
        """Create pinned artifacts for testing."""
        manifest = artifact_manager.pin_artifacts("v1.2.3", sample_artifacts)
        artifact_manager.copy_artifacts("v1.2.3", sample_artifacts)
        return manifest

    def test_validate_version_artifacts_success(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test successful version validation."""
        from scripts.validate_artifacts import validate_version_artifacts

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = True
        args.verbose = False
        args.output = None

        # This should not raise an exception
        validate_version_artifacts(
            artifact_manager, integrity_validator, "v1.2.3", sample_artifacts, args
        )

    def test_validate_version_artifacts_failure(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test version validation failure."""
        from scripts.validate_artifacts import validate_version_artifacts

        # Corrupt a file
        (sample_artifacts / "model.pt").write_text("corrupted data")

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = True
        args.verbose = False
        args.output = None

        # This should raise SystemExit due to validation failure
        with pytest.raises(SystemExit) as exc_info:
            validate_version_artifacts(
                artifact_manager, integrity_validator, "v1.2.3", sample_artifacts, args
            )

        assert exc_info.value.code == 1

    def test_validate_version_artifacts_with_output(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts, temp_dir
    ):
        """Test version validation with output file."""
        from scripts.validate_artifacts import validate_version_artifacts

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = False
        args.verbose = False
        args.output = str(temp_dir / "validation_report.json")

        # Run validation
        validate_version_artifacts(
            artifact_manager, integrity_validator, "v1.2.3", sample_artifacts, args
        )

        # Check that output file was created
        output_file = Path(args.output)
        assert output_file.exists()

        # Check that it contains valid JSON
        with open(output_file) as f:
            report = json.load(f)

        assert "version" in report
        assert "total_artifacts" in report
        assert "valid_artifacts" in report

    def test_validate_all_versions(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validating all versions."""
        from scripts.validate_artifacts import validate_all_versions

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = False
        args.verbose = False
        args.output = None

        # This should not raise an exception
        validate_all_versions(artifact_manager, integrity_validator, args)

    def test_validate_all_versions_with_output(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts, temp_dir
    ):
        """Test validating all versions with output file."""
        from scripts.validate_artifacts import validate_all_versions

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = False
        args.verbose = False
        args.output = str(temp_dir / "all_versions_report.json")

        # Run validation
        validate_all_versions(artifact_manager, integrity_validator, args)

        # Check that output file was created
        output_file = Path(args.output)
        assert output_file.exists()

        # Check that it contains valid JSON
        with open(output_file) as f:
            report = json.load(f)

        assert "summary" in report
        assert "results" in report
        assert "total_versions" in report["summary"]

    def test_generate_integrity_report(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts, temp_dir
    ):
        """Test generating integrity report."""
        from scripts.validate_artifacts import generate_integrity_report

        # Mock args
        args = Mock()
        args.output = str(temp_dir / "integrity_report.json")
        args.verbose = False

        # Run report generation
        generate_integrity_report(artifact_manager, integrity_validator, "v1.2.3", args)

        # Check that output file was created
        output_file = Path(args.output)
        assert output_file.exists()

        # Check that it contains valid JSON
        with open(output_file) as f:
            report = json.load(f)

        assert "version" in report
        assert "integrity_status" in report
        assert "artifacts_summary" in report
        assert "artifacts" in report

    def test_generate_integrity_report_nonexistent_version(
        self, artifact_manager, integrity_validator
    ):
        """Test generating integrity report for nonexistent version."""
        from scripts.validate_artifacts import generate_integrity_report

        # Mock args
        args = Mock()
        args.output = None
        args.verbose = False

        # This should raise SystemExit
        with pytest.raises(SystemExit) as exc_info:
            generate_integrity_report(artifact_manager, integrity_validator, "v999.999.999", args)

        assert exc_info.value.code == 1

    def test_compare_versions(self, artifact_manager, integrity_validator, temp_dir):
        """Test comparing two versions."""
        from scripts.validate_artifacts import compare_versions

        # Create two different versions
        artifacts_v1 = temp_dir / "artifacts_v1"
        artifacts_v1.mkdir()
        (artifacts_v1 / "model.pt").write_text("model v1")
        (artifacts_v1 / "config.yaml").write_text("config v1")

        artifacts_v2 = temp_dir / "artifacts_v2"
        artifacts_v2.mkdir()
        (artifacts_v2 / "model.pt").write_text("model v2")
        (artifacts_v2 / "config.yaml").write_text("config v2")
        (artifacts_v2 / "preprocessor.pkl").write_text("preprocessor v2")

        # Pin both versions
        artifact_manager.pin_artifacts("v1.1.0", artifacts_v1)
        artifact_manager.copy_artifacts("v1.1.0", artifacts_v1)

        artifact_manager.pin_artifacts("v1.2.0", artifacts_v2)
        artifact_manager.copy_artifacts("v1.2.0", artifacts_v2)

        # Mock args
        args = Mock()
        args.output = str(temp_dir / "comparison.json")
        args.verbose = False

        # Run comparison
        compare_versions(artifact_manager, integrity_validator, "v1.1.0", "v1.2.0", args)

        # Check that output file was created
        output_file = Path(args.output)
        assert output_file.exists()

        # Check that it contains valid JSON
        with open(output_file) as f:
            comparison = json.load(f)

        assert "version1" in comparison
        assert "version2" in comparison
        assert "summary" in comparison
        assert "common_artifacts" in comparison
        assert "changed_artifacts" in comparison

    def test_compare_versions_nonexistent(self, artifact_manager, integrity_validator):
        """Test comparing versions where one doesn't exist."""
        from scripts.validate_artifacts import compare_versions

        # Mock args
        args = Mock()
        args.output = None
        args.verbose = False

        # This should raise SystemExit
        with pytest.raises(SystemExit) as exc_info:
            compare_versions(artifact_manager, integrity_validator, "v1.1.0", "v999.999.999", args)

        assert exc_info.value.code == 1

    def test_validate_with_required_artifacts(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validation with specific required artifacts."""
        from scripts.validate_artifacts import validate_version_artifacts

        # Mock args
        args = Mock()
        args.required_artifacts = ["model.pt"]
        args.strict = False
        args.verbose = False
        args.output = None

        # This should not raise an exception
        validate_version_artifacts(
            artifact_manager, integrity_validator, "v1.2.3", sample_artifacts, args
        )

    def test_validate_with_missing_required_artifacts(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validation with missing required artifacts."""
        from scripts.validate_artifacts import validate_version_artifacts

        # Mock args
        args = Mock()
        args.required_artifacts = ["model.pt", "nonexistent.pt"]
        args.strict = True
        args.verbose = False
        args.output = None

        # This should raise SystemExit due to missing required artifact
        with pytest.raises(SystemExit) as exc_info:
            validate_version_artifacts(
                artifact_manager, integrity_validator, "v1.2.3", sample_artifacts, args
            )

        assert exc_info.value.code == 1

    def test_validate_version_artifacts_invalid_version_format(
        self, artifact_manager, integrity_validator, sample_artifacts
    ):
        """Test validation with invalid version format."""
        from scripts.validate_artifacts import validate_version_artifacts

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = False
        args.verbose = False
        args.output = None

        # This should raise SystemExit due to invalid version format
        with pytest.raises(SystemExit) as exc_info:
            validate_version_artifacts(
                artifact_manager, integrity_validator, "invalid-version", sample_artifacts, args
            )

        assert exc_info.value.code == 1

    def test_validate_all_versions_strict_mode(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts
    ):
        """Test validating all versions in strict mode."""
        from scripts.validate_artifacts import validate_all_versions

        # Corrupt a file in the versioned artifacts directory
        versioned_artifacts_dir = artifact_manager.versions_dir / "v1.2.3"
        (versioned_artifacts_dir / "model.pt").write_text("corrupted data")

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = True
        args.verbose = False
        args.output = None

        # This should raise SystemExit due to validation failure in strict mode
        with pytest.raises(SystemExit) as exc_info:
            validate_all_versions(artifact_manager, integrity_validator, args)

        assert exc_info.value.code == 1

    def test_validate_version_artifacts_verbose_output(
        self, artifact_manager, integrity_validator, sample_artifacts, pinned_artifacts, capsys
    ):
        """Test validation with verbose output."""
        from scripts.validate_artifacts import validate_version_artifacts

        # Mock args
        args = Mock()
        args.required_artifacts = None
        args.strict = False
        args.verbose = True
        args.output = None

        # Run validation
        validate_version_artifacts(
            artifact_manager, integrity_validator, "v1.2.3", sample_artifacts, args
        )

        # Check that verbose output was produced
        captured = capsys.readouterr()
        assert "Artifact Details:" in captured.out
        assert "model.pt" in captured.out
        assert "config.yaml" in captured.out

    def test_compare_versions_verbose_output(
        self, artifact_manager, integrity_validator, temp_dir, capsys
    ):
        """Test comparison with verbose output."""
        from scripts.validate_artifacts import compare_versions

        # Create two different versions
        artifacts_v1 = temp_dir / "artifacts_v1"
        artifacts_v1.mkdir()
        (artifacts_v1 / "model.pt").write_text("model v1")
        (artifacts_v1 / "config.yaml").write_text("config v1")

        artifacts_v2 = temp_dir / "artifacts_v2"
        artifacts_v2.mkdir()
        (artifacts_v2 / "model.pt").write_text("model v2")
        (artifacts_v2 / "config.yaml").write_text("config v2")
        (artifacts_v2 / "preprocessor.pkl").write_text("preprocessor v2")

        # Pin both versions
        artifact_manager.pin_artifacts("v1.1.0", artifacts_v1)
        artifact_manager.copy_artifacts("v1.1.0", artifacts_v1)

        artifact_manager.pin_artifacts("v1.2.0", artifacts_v2)
        artifact_manager.copy_artifacts("v1.2.0", artifacts_v2)

        # Mock args
        args = Mock()
        args.output = None
        args.verbose = True

        # Run comparison
        compare_versions(artifact_manager, integrity_validator, "v1.1.0", "v1.2.0", args)

        # Check that verbose output was produced
        captured = capsys.readouterr()
        assert "Changed Artifacts:" in captured.out
        assert "Only in v1.2.0:" in captured.out
