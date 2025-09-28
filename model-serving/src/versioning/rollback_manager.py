"""
Artifact rollback functionality for version management.

This module provides comprehensive rollback capabilities for reverting
to previous model versions with proper validation and safety checks.
"""

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from .artifact_manager import ArtifactManager
from .integrity_validator import IntegrityValidator
from .semantic_version import SemanticVersion, parse_version


logger = logging.getLogger(__name__)


class RollbackError(Exception):
    """Exception raised for rollback operation errors."""

    def __init__(
        self,
        message: str,
        current_version: Optional[str] = None,
        target_version: Optional[str] = None,
    ):
        super().__init__(message)
        self.current_version = current_version
        self.target_version = target_version


@dataclass
class RollbackOperation:
    """Represents a rollback operation with metadata."""

    operation_id: str
    from_version: str
    to_version: str
    timestamp: str
    status: str  # pending, in_progress, completed, failed, rolled_back
    reason: str
    artifacts_affected: List[str]
    backup_location: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "timestamp": self.timestamp,
            "status": self.status,
            "reason": self.reason,
            "artifacts_affected": self.artifacts_affected,
            "backup_location": self.backup_location,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RollbackOperation":
        """Create from dictionary."""
        return cls(
            operation_id=data.get("operation_id", ""),
            from_version=data.get("from_version", ""),
            to_version=data.get("to_version", ""),
            timestamp=data.get("timestamp", ""),
            status=data.get("status", "pending"),
            reason=data.get("reason", ""),
            artifacts_affected=data.get("artifacts_affected", []),
            backup_location=data.get("backup_location"),
            error_message=data.get("error_message"),
        )


class RollbackManager:
    """
    Manages artifact rollback operations with safety checks and validation.

    Provides comprehensive rollback capabilities including version validation,
    backup creation, rollback execution, and rollback reversal.
    """

    def __init__(self, artifact_manager: ArtifactManager, integrity_validator: IntegrityValidator):
        """
        Initialize rollback manager.

        Args:
            artifact_manager: ArtifactManager instance for artifact operations
            integrity_validator: IntegrityValidator instance for validation
        """
        self.artifact_manager = artifact_manager
        self.integrity_validator = integrity_validator
        self.rollback_history: List[RollbackOperation] = []
        self._load_rollback_history()

        logger.info("Initialized RollbackManager")

    def rollback_to_version(
        self,
        target_version: Union[str, SemanticVersion],
        current_version: Optional[Union[str, SemanticVersion]] = None,
        reason: str = "Manual rollback",
        create_backup: bool = True,
        validate_target: bool = True,
    ) -> RollbackOperation:
        """
        Rollback to a specific version.

        Args:
            target_version: Version to rollback to
            current_version: Current version (if None, will be detected)
            reason: Reason for rollback
            create_backup: Whether to create backup of current state
            validate_target: Whether to validate target version integrity

        Returns:
            RollbackOperation object with operation details

        Raises:
            RollbackError: If rollback fails
        """
        target_version = parse_version(target_version)
        target_version_str = str(target_version)

        logger.info(f"Starting rollback to version {target_version_str}")

        # Detect current version if not provided
        if current_version is None:
            current_version = self._detect_current_version()
            if current_version is None:
                raise RollbackError("Cannot detect current version and none provided")

        current_version = parse_version(current_version)
        current_version_str = str(current_version)

        # Validate target version exists
        if not self._version_exists(target_version):
            raise RollbackError(
                f"Target version {target_version_str} does not exist",
                current_version_str,
                target_version_str,
            )

        # Validate target version integrity if requested
        if validate_target:
            if not self._validate_version_integrity(target_version):
                raise RollbackError(
                    f"Target version {target_version_str} failed integrity validation",
                    current_version_str,
                    target_version_str,
                )

        # Create rollback operation
        operation = self._create_rollback_operation(current_version_str, target_version_str, reason)

        try:
            # Update operation status
            operation.status = "in_progress"
            self._save_rollback_operation(operation)

            # Create backup if requested
            if create_backup:
                backup_location = self._create_backup(current_version_str)
                operation.backup_location = backup_location
                logger.info(f"Created backup at {backup_location}")

            # Get artifacts to be affected
            operation.artifacts_affected = self._get_affected_artifacts(
                current_version_str, target_version_str
            )

            # Perform rollback
            self._execute_rollback(operation)

            # Update operation status
            operation.status = "completed"
            self._save_rollback_operation(operation)

            logger.info(f"Successfully rolled back to version {target_version_str}")
            return operation

        except Exception as e:
            # Update operation status
            operation.status = "failed"
            operation.error_message = str(e)
            self._save_rollback_operation(operation)

            logger.error(f"Rollback failed: {e}")
            raise RollbackError(f"Rollback failed: {e}", current_version_str, target_version_str)

    def rollback_operation(
        self, operation_id: str, reason: str = "Rollback reversal"
    ) -> RollbackOperation:
        """
        Rollback a previous rollback operation.

        Args:
            operation_id: ID of the operation to rollback
            reason: Reason for rollback reversal

        Returns:
            New RollbackOperation object for the reversal

        Raises:
            RollbackError: If rollback reversal fails
        """
        # Find the original operation
        original_operation = self._get_rollback_operation(operation_id)
        if not original_operation:
            raise RollbackError(f"Rollback operation {operation_id} not found")

        if original_operation.status != "completed":
            raise RollbackError(
                f"Cannot rollback operation {operation_id} with status {original_operation.status}"
            )

        logger.info(f"Rolling back operation {operation_id}")

        # Create reversal operation
        reversal_operation = self._create_rollback_operation(
            original_operation.to_version, original_operation.from_version, reason
        )
        reversal_operation.backup_location = original_operation.backup_location

        try:
            # Update operation status
            reversal_operation.status = "in_progress"
            self._save_rollback_operation(reversal_operation)

            # Execute rollback reversal
            self._execute_rollback(reversal_operation)

            # Update operation status
            reversal_operation.status = "completed"
            self._save_rollback_operation(reversal_operation)

            # Mark original operation as rolled back
            original_operation.status = "rolled_back"
            self._save_rollback_operation(original_operation)

            logger.info(f"Successfully rolled back operation {operation_id}")
            return reversal_operation

        except Exception as e:
            # Update operation status
            reversal_operation.status = "failed"
            reversal_operation.error_message = str(e)
            self._save_rollback_operation(reversal_operation)

            logger.error(f"Rollback reversal failed: {e}")
            raise RollbackError(f"Rollback reversal failed: {e}")

    def list_rollback_operations(self, status: Optional[str] = None) -> List[RollbackOperation]:
        """
        List rollback operations.

        Args:
            status: Filter by status (if None, returns all)

        Returns:
            List of RollbackOperation objects
        """
        if status:
            return [op for op in self.rollback_history if op.status == status]
        return self.rollback_history.copy()

    def get_rollback_operation(self, operation_id: str) -> Optional[RollbackOperation]:
        """
        Get a specific rollback operation.

        Args:
            operation_id: Operation ID to retrieve

        Returns:
            RollbackOperation object or None if not found
        """
        return self._get_rollback_operation(operation_id)

    def get_available_versions(self) -> List[SemanticVersion]:
        """
        Get list of available versions for rollback.

        Returns:
            List of available versions sorted by version number
        """
        return self.artifact_manager.list_versions()

    def get_version_info(self, version: Union[str, SemanticVersion]) -> Dict[str, any]:
        """
        Get information about a version.

        Args:
            version: Version to get info for

        Returns:
            Dictionary with version information
        """
        version = parse_version(version)
        version_str = str(version)

        manifest = self.artifact_manager.get_manifest(version)
        if not manifest:
            return {"version": version_str, "exists": False}

        # Get integrity status
        version_dir = self.artifact_manager.versions_dir / version_str
        integrity_results = self.artifact_manager.verify_integrity(version)

        return {
            "version": version_str,
            "exists": True,
            "manifest": manifest.to_dict(),
            "integrity_status": "valid" if all(integrity_results.values()) else "invalid",
            "integrity_results": integrity_results,
            "artifacts_count": len(manifest.artifacts),
            "version_dir": str(version_dir),
        }

    def cleanup_old_rollbacks(self, keep_last_n: int = 10) -> int:
        """
        Clean up old rollback operations, keeping only the last N.

        Args:
            keep_last_n: Number of recent operations to keep

        Returns:
            Number of operations cleaned up
        """
        if len(self.rollback_history) <= keep_last_n:
            return 0

        # Sort by timestamp (newest first)
        sorted_operations = sorted(self.rollback_history, key=lambda op: op.timestamp, reverse=True)

        # Keep the most recent N operations
        operations_to_remove = sorted_operations[keep_last_n:]

        # Remove old operations
        for operation in operations_to_remove:
            self.rollback_history.remove(operation)
            self._remove_rollback_operation(operation.operation_id)

        logger.info(f"Cleaned up {len(operations_to_remove)} old rollback operations")
        return len(operations_to_remove)

    def _detect_current_version(self) -> Optional[SemanticVersion]:
        """Detect the current active version."""
        # This is a placeholder implementation
        # In a real system, this would check the current active version
        # For now, we'll return the latest version
        versions = self.artifact_manager.list_versions()
        if versions:
            return versions[-1]  # Return the latest version
        return None

    def _version_exists(self, version: SemanticVersion) -> bool:
        """Check if a version exists."""
        manifest = self.artifact_manager.get_manifest(version)
        return manifest is not None

    def _validate_version_integrity(self, version: SemanticVersion) -> bool:
        """Validate version integrity."""
        try:
            version_dir = self.artifact_manager.versions_dir / str(version)
            is_valid, _ = self.integrity_validator.validate_model_artifacts(
                version, version_dir, strict_mode=False
            )
            return is_valid
        except Exception as e:
            logger.warning(f"Version integrity validation failed: {e}")
            return False

    def _create_rollback_operation(
        self, from_version: str, to_version: str, reason: str
    ) -> RollbackOperation:
        """Create a new rollback operation."""
        operation_id = self._generate_operation_id()
        timestamp = self._get_current_timestamp()

        return RollbackOperation(
            operation_id=operation_id,
            from_version=from_version,
            to_version=to_version,
            timestamp=timestamp,
            status="pending",
            reason=reason,
            artifacts_affected=[],
        )

    def _execute_rollback(self, operation: RollbackOperation) -> None:
        """Execute the rollback operation."""
        # This is a placeholder implementation
        # In a real system, this would:
        # 1. Update the current version pointer
        # 2. Update any configuration files
        # 3. Restart services if needed
        # 4. Update any databases or registries

        logger.info(f"Executing rollback from {operation.from_version} to {operation.to_version}")

        # For now, we'll just log the operation
        # In a real implementation, this would perform the actual rollback
        pass

    def _create_backup(self, version: str) -> str:
        """Create backup of current state."""
        backup_dir = (
            self.artifact_manager.artifacts_dir
            / "backups"
            / f"backup_{version}_{self._get_timestamp()}"
        )
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy current version to backup
        current_version_dir = self.artifact_manager.versions_dir / version
        if current_version_dir.exists():
            shutil.copytree(current_version_dir, backup_dir / "artifacts")

        # Copy manifest
        manifest_file = self.artifact_manager.manifests_dir / f"{version}.json"
        if manifest_file.exists():
            shutil.copy2(manifest_file, backup_dir / "manifest.json")

        return str(backup_dir)

    def _get_affected_artifacts(self, from_version: str, to_version: str) -> List[str]:
        """Get list of artifacts that will be affected by rollback."""
        from_manifest = self.artifact_manager.get_manifest(parse_version(from_version))
        to_manifest = self.artifact_manager.get_manifest(parse_version(to_version))

        if not from_manifest or not to_manifest:
            return []

        # Find artifacts that differ between versions
        affected_artifacts = []

        # Check artifacts in from_version
        for artifact in from_manifest.artifacts:
            if artifact not in to_manifest.artifacts:
                affected_artifacts.append(f"{artifact} (removed)")
            elif from_manifest.artifacts[artifact] != to_manifest.artifacts[artifact]:
                affected_artifacts.append(f"{artifact} (changed)")

        # Check artifacts in to_version
        for artifact in to_manifest.artifacts:
            if artifact not in from_manifest.artifacts:
                affected_artifacts.append(f"{artifact} (added)")

        return affected_artifacts

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        import uuid

        return str(uuid.uuid4())[:8]

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _get_timestamp(self) -> str:
        """Get timestamp for file names."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_rollback_history(self) -> None:
        """Load rollback history from storage."""
        # This is a placeholder implementation
        # In a real system, this would load from persistent storage
        self.rollback_history = []

    def _save_rollback_operation(self, operation: RollbackOperation) -> None:
        """Save rollback operation to storage."""
        # This is a placeholder implementation
        # In a real system, this would save to persistent storage
        if operation not in self.rollback_history:
            self.rollback_history.append(operation)

    def _get_rollback_operation(self, operation_id: str) -> Optional[RollbackOperation]:
        """Get rollback operation by ID."""
        for operation in self.rollback_history:
            if operation.operation_id == operation_id:
                return operation
        return None

    def _remove_rollback_operation(self, operation_id: str) -> None:
        """Remove rollback operation from storage."""
        # This is a placeholder implementation
        # In a real system, this would remove from persistent storage
        pass
