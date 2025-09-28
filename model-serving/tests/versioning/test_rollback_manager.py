"""
Unit tests for artifact rollback functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.versioning.rollback_manager import (
    RollbackError,
    RollbackManager,
    RollbackOperation,
)
from src.versioning.artifact_manager import ArtifactManager
from src.versioning.integrity_validator import IntegrityValidator
from src.versioning.semantic_version import SemanticVersion


class TestRollbackOperation:
    """Test RollbackOperation functionality."""

    def test_rollback_operation_creation(self):
        """Test creating a rollback operation."""
        operation = RollbackOperation(
            operation_id="test123",
            from_version="v1.2.3",
            to_version="v1.1.0",
            timestamp="2024-01-01T00:00:00Z",
            status="pending",
            reason="Test rollback",
            artifacts_affected=["model.pt", "config.yaml"],
        )

        assert operation.operation_id == "test123"
        assert operation.from_version == "v1.2.3"
        assert operation.to_version == "v1.1.0"
        assert operation.status == "pending"
        assert operation.reason == "Test rollback"
        assert operation.artifacts_affected == ["model.pt", "config.yaml"]

    def test_rollback_operation_to_dict(self):
        """Test converting rollback operation to dictionary."""
        operation = RollbackOperation(
            operation_id="test123",
            from_version="v1.2.3",
            to_version="v1.1.0",
            timestamp="2024-01-01T00:00:00Z",
            status="completed",
            reason="Test rollback",
            artifacts_affected=["model.pt"],
        )

        data = operation.to_dict()

        assert data["operation_id"] == "test123"
        assert data["from_version"] == "v1.2.3"
        assert data["to_version"] == "v1.1.0"
        assert data["status"] == "completed"
        assert data["reason"] == "Test rollback"
        assert data["artifacts_affected"] == ["model.pt"]

    def test_rollback_operation_from_dict(self):
        """Test creating rollback operation from dictionary."""
        data = {
            "operation_id": "test123",
            "from_version": "v1.2.3",
            "to_version": "v1.1.0",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "completed",
            "reason": "Test rollback",
            "artifacts_affected": ["model.pt"],
            "backup_location": "/backup/path",
            "error_message": None,
        }

        operation = RollbackOperation.from_dict(data)

        assert operation.operation_id == "test123"
        assert operation.from_version == "v1.2.3"
        assert operation.to_version == "v1.1.0"
        assert operation.status == "completed"
        assert operation.reason == "Test rollback"
        assert operation.artifacts_affected == ["model.pt"]
        assert operation.backup_location == "/backup/path"
        assert operation.error_message is None


class TestRollbackManager:
    """Test RollbackManager functionality."""

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
    def rollback_manager(self, artifact_manager, integrity_validator):
        """Create RollbackManager instance for testing."""
        return RollbackManager(artifact_manager, integrity_validator)

    @pytest.fixture
    def sample_artifacts_v1(self, temp_dir):
        """Create sample artifacts for version 1."""
        artifacts_dir = temp_dir / "artifacts_v1"
        artifacts_dir.mkdir()

        (artifacts_dir / "model.pt").write_text("model v1")
        (artifacts_dir / "config.yaml").write_text("config v1")

        return artifacts_dir

    @pytest.fixture
    def sample_artifacts_v2(self, temp_dir):
        """Create sample artifacts for version 2."""
        artifacts_dir = temp_dir / "artifacts_v2"
        artifacts_dir.mkdir()

        (artifacts_dir / "model.pt").write_text("model v2")
        (artifacts_dir / "config.yaml").write_text("config v2")
        (artifacts_dir / "preprocessor.pkl").write_text("preprocessor v2")

        return artifacts_dir

    @pytest.fixture
    def pinned_versions(self, artifact_manager, sample_artifacts_v1, sample_artifacts_v2):
        """Create pinned versions for testing."""
        # Pin version 1
        manifest1 = artifact_manager.pin_artifacts("v1.1.0", sample_artifacts_v1)
        artifact_manager.copy_artifacts("v1.1.0", sample_artifacts_v1)

        # Pin version 2
        manifest2 = artifact_manager.pin_artifacts("v1.2.0", sample_artifacts_v2)
        artifact_manager.copy_artifacts("v1.2.0", sample_artifacts_v2)

        return manifest1, manifest2

    def test_rollback_manager_initialization(self, artifact_manager, integrity_validator):
        """Test RollbackManager initialization."""
        manager = RollbackManager(artifact_manager, integrity_validator)

        assert manager.artifact_manager == artifact_manager
        assert manager.integrity_validator == integrity_validator
        assert manager.rollback_history == []

    def test_rollback_to_version_success(self, rollback_manager, pinned_versions):
        """Test successful rollback to version."""
        # Mock the _detect_current_version method
        with patch.object(
            rollback_manager, "_detect_current_version", return_value=SemanticVersion(1, 2, 0)
        ):
            operation = rollback_manager.rollback_to_version("v1.1.0", reason="Test rollback")

        assert operation.from_version == "v1.2.0"
        assert operation.to_version == "v1.1.0"
        assert operation.status == "completed"
        assert operation.reason == "Test rollback"

    def test_rollback_to_version_nonexistent_target(self, rollback_manager, pinned_versions):
        """Test rollback to nonexistent target version."""
        with patch.object(
            rollback_manager, "_detect_current_version", return_value=SemanticVersion(1, 2, 0)
        ):
            with pytest.raises(RollbackError) as exc_info:
                rollback_manager.rollback_to_version("v999.999.999")

        assert "Target version v999.999.999 does not exist" in str(exc_info.value)
        assert exc_info.value.current_version == "v1.2.0"
        assert exc_info.value.target_version == "v999.999.999"

    def test_rollback_to_version_integrity_validation_failure(
        self, rollback_manager, pinned_versions
    ):
        """Test rollback with integrity validation failure."""
        with patch.object(
            rollback_manager, "_detect_current_version", return_value=SemanticVersion(1, 2, 0)
        ):
            with patch.object(rollback_manager, "_validate_version_integrity", return_value=False):
                with pytest.raises(RollbackError) as exc_info:
                    rollback_manager.rollback_to_version("v1.1.0", validate_target=True)

        assert "failed integrity validation" in str(exc_info.value)

    def test_rollback_to_version_with_backup(self, rollback_manager, pinned_versions):
        """Test rollback with backup creation."""
        with patch.object(
            rollback_manager, "_detect_current_version", return_value=SemanticVersion(1, 2, 0)
        ):
            with patch.object(rollback_manager, "_create_backup", return_value="/backup/path"):
                operation = rollback_manager.rollback_to_version("v1.1.0", create_backup=True)

        assert operation.backup_location == "/backup/path"

    def test_rollback_to_version_without_backup(self, rollback_manager, pinned_versions):
        """Test rollback without backup creation."""
        with patch.object(
            rollback_manager, "_detect_current_version", return_value=SemanticVersion(1, 2, 0)
        ):
            operation = rollback_manager.rollback_to_version("v1.1.0", create_backup=False)

        assert operation.backup_location is None

    def test_rollback_operation_success(self, rollback_manager, pinned_versions):
        """Test successful rollback operation reversal."""
        # Create a completed rollback operation
        operation = RollbackOperation(
            operation_id="test123",
            from_version="v1.1.0",
            to_version="v1.2.0",
            timestamp="2024-01-01T00:00:00Z",
            status="completed",
            reason="Test rollback",
            artifacts_affected=["model.pt"],
        )
        rollback_manager.rollback_history.append(operation)

        # Rollback the operation
        reversal_operation = rollback_manager.rollback_operation("test123", "Reversal test")

        assert reversal_operation.from_version == "v1.2.0"
        assert reversal_operation.to_version == "v1.1.0"
        assert reversal_operation.status == "completed"
        assert reversal_operation.reason == "Reversal test"

        # Check that original operation is marked as rolled back
        assert operation.status == "rolled_back"

    def test_rollback_operation_not_found(self, rollback_manager):
        """Test rollback operation with non-existent operation ID."""
        with pytest.raises(RollbackError) as exc_info:
            rollback_manager.rollback_operation("nonexistent", "Test")

        assert "Rollback operation nonexistent not found" in str(exc_info.value)

    def test_rollback_operation_invalid_status(self, rollback_manager):
        """Test rollback operation with invalid status."""
        # Create a failed rollback operation
        operation = RollbackOperation(
            operation_id="test123",
            from_version="v1.1.0",
            to_version="v1.2.0",
            timestamp="2024-01-01T00:00:00Z",
            status="failed",
            reason="Test rollback",
            artifacts_affected=[],
        )
        rollback_manager.rollback_history.append(operation)

        with pytest.raises(RollbackError) as exc_info:
            rollback_manager.rollback_operation("test123", "Test")

        assert "Cannot rollback operation test123 with status failed" in str(exc_info.value)

    def test_list_rollback_operations(self, rollback_manager):
        """Test listing rollback operations."""
        # Add some test operations
        operation1 = RollbackOperation(
            operation_id="test1",
            from_version="v1.1.0",
            to_version="v1.2.0",
            timestamp="2024-01-01T00:00:00Z",
            status="completed",
            reason="Test 1",
            artifacts_affected=[],
        )
        operation2 = RollbackOperation(
            operation_id="test2",
            from_version="v1.2.0",
            to_version="v1.1.0",
            timestamp="2024-01-02T00:00:00Z",
            status="failed",
            reason="Test 2",
            artifacts_affected=[],
        )

        rollback_manager.rollback_history.extend([operation1, operation2])

        # Test listing all operations
        all_operations = rollback_manager.list_rollback_operations()
        assert len(all_operations) == 2

        # Test filtering by status
        completed_operations = rollback_manager.list_rollback_operations("completed")
        assert len(completed_operations) == 1
        assert completed_operations[0].operation_id == "test1"

        failed_operations = rollback_manager.list_rollback_operations("failed")
        assert len(failed_operations) == 1
        assert failed_operations[0].operation_id == "test2"

    def test_get_rollback_operation(self, rollback_manager):
        """Test getting specific rollback operation."""
        operation = RollbackOperation(
            operation_id="test123",
            from_version="v1.1.0",
            to_version="v1.2.0",
            timestamp="2024-01-01T00:00:00Z",
            status="completed",
            reason="Test",
            artifacts_affected=[],
        )
        rollback_manager.rollback_history.append(operation)

        # Test getting existing operation
        retrieved = rollback_manager.get_rollback_operation("test123")
        assert retrieved == operation

        # Test getting non-existent operation
        retrieved = rollback_manager.get_rollback_operation("nonexistent")
        assert retrieved is None

    def test_get_available_versions(self, rollback_manager, pinned_versions):
        """Test getting available versions."""
        versions = rollback_manager.get_available_versions()

        assert len(versions) == 2
        assert SemanticVersion(1, 1, 0) in versions
        assert SemanticVersion(1, 2, 0) in versions

    def test_get_version_info_existing(self, rollback_manager, pinned_versions):
        """Test getting version info for existing version."""
        info = rollback_manager.get_version_info("v1.1.0")

        assert info["version"] == "v1.1.0"
        assert info["exists"] is True
        assert "manifest" in info
        assert "integrity_status" in info
        assert "artifacts_count" in info

    def test_get_version_info_nonexistent(self, rollback_manager):
        """Test getting version info for nonexistent version."""
        info = rollback_manager.get_version_info("v999.999.999")

        assert info["version"] == "v999.999.999"
        assert info["exists"] is False

    def test_cleanup_old_rollbacks(self, rollback_manager):
        """Test cleaning up old rollback operations."""
        # Add more than 10 operations
        for i in range(15):
            operation = RollbackOperation(
                operation_id=f"test{i}",
                from_version="v1.1.0",
                to_version="v1.2.0",
                timestamp=f"2024-01-{i+1:02d}T00:00:00Z",
                status="completed",
                reason=f"Test {i}",
                artifacts_affected=[],
            )
            rollback_manager.rollback_history.append(operation)

        # Clean up old operations
        cleaned_count = rollback_manager.cleanup_old_rollbacks(keep_last_n=10)

        assert cleaned_count == 5
        assert len(rollback_manager.rollback_history) == 10

    def test_cleanup_old_rollbacks_no_cleanup_needed(self, rollback_manager):
        """Test cleanup when no cleanup is needed."""
        # Add only 5 operations
        for i in range(5):
            operation = RollbackOperation(
                operation_id=f"test{i}",
                from_version="v1.1.0",
                to_version="v1.2.0",
                timestamp=f"2024-01-{i+1:02d}T00:00:00Z",
                status="completed",
                reason=f"Test {i}",
                artifacts_affected=[],
            )
            rollback_manager.rollback_history.append(operation)

        # Clean up old operations
        cleaned_count = rollback_manager.cleanup_old_rollbacks(keep_last_n=10)

        assert cleaned_count == 0
        assert len(rollback_manager.rollback_history) == 5

    def test_get_affected_artifacts(self, rollback_manager, pinned_versions):
        """Test getting affected artifacts between versions."""
        affected = rollback_manager._get_affected_artifacts("v1.1.0", "v1.2.0")

        # Should include artifacts that are different between versions
        assert len(affected) > 0
        # Should include the preprocessor.pkl that was added in v1.2.0
        assert any("preprocessor.pkl" in artifact for artifact in affected)

    def test_rollback_execution_error(self, rollback_manager, pinned_versions):
        """Test rollback execution error handling."""
        with patch.object(
            rollback_manager, "_detect_current_version", return_value=SemanticVersion(1, 2, 0)
        ):
            with patch.object(
                rollback_manager, "_execute_rollback", side_effect=Exception("Execution error")
            ):
                with pytest.raises(RollbackError) as exc_info:
                    rollback_manager.rollback_to_version("v1.1.0")

        assert "Rollback failed: Execution error" in str(exc_info.value)

        # Check that operation was marked as failed
        operations = rollback_manager.list_rollback_operations()
        assert len(operations) == 1
        assert operations[0].status == "failed"
        assert operations[0].error_message == "Execution error"
