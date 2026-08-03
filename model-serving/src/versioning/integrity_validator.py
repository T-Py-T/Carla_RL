"""
Artifact integrity validation for model loading.

This module provides comprehensive integrity validation that integrates
with the model loading process to ensure artifact integrity before
model inference.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .artifact_manager import ArtifactManager
from .semantic_version import SemanticVersion, parse_version


logger = logging.getLogger(__name__)


class IntegrityValidationError(Exception):
    """Exception raised for integrity validation errors during model loading."""

    def __init__(
        self,
        message: str,
        version: Optional[str] = None,
        failed_artifacts: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.version = version
        self.failed_artifacts = failed_artifacts or []


class IntegrityValidator:
    """
    Validates artifact integrity during model loading.

    Provides comprehensive validation that ensures all model artifacts
    are intact and match their expected hashes before model loading.
    """

    def __init__(self, artifact_manager: ArtifactManager):
        """
        Initialize integrity validator.

        Args:
            artifact_manager: ArtifactManager instance for hash validation
        """
        self.artifact_manager = artifact_manager
        self._validation_cache: Dict[str, bool] = {}

    def validate_model_artifacts(
        self,
        version: Union[str, SemanticVersion],
        artifacts_dir: Path,
        required_artifacts: Optional[List[str]] = None,
        strict_mode: bool = True,
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Validate all model artifacts for a version.

        Args:
            version: Model version to validate
            artifacts_dir: Directory containing artifacts to validate
            required_artifacts: List of required artifact files (if None, validates all)
            strict_mode: If True, fails on any validation error. If False, continues and reports issues.

        Returns:
            Tuple of (is_valid, validation_report)

        Raises:
            IntegrityValidationError: If validation fails in strict mode
        """
        version = parse_version(version)
        artifacts_dir = Path(artifacts_dir)

        logger.info(f"Starting integrity validation for version {version}")

        # Check if we've already validated this version recently
        cache_key = f"{version}_{artifacts_dir}"
        if cache_key in self._validation_cache:
            logger.debug(f"Using cached validation result for {version}")
            return self._validation_cache[cache_key], {"cached": True}

        validation_report = self._new_validation_report(version, artifacts_dir, strict_mode)

        try:
            artifacts_to_validate = self._artifacts_to_validate(
                version, required_artifacts, validation_report, strict_mode
            )
            validation_report["total_artifacts"] = len(artifacts_to_validate)
            for artifact_path, expected_hash in artifacts_to_validate.items():
                self._validate_artifact(
                    version,
                    artifacts_dir,
                    artifact_path,
                    expected_hash,
                    validation_report,
                    strict_mode,
                )

            is_valid = not (
                validation_report["invalid_artifacts"]
                or validation_report["missing_artifacts"]
                or validation_report["errors"]
            )
            validation_report["is_valid"] = is_valid
            if is_valid:
                self._validation_cache[cache_key] = True
            self._log_validation_result(version, validation_report, is_valid)
            return is_valid, validation_report
        except IntegrityValidationError:
            raise
        except Exception as exc:
            return self._handle_unexpected_error(
                exc, version, validation_report, strict_mode
            )

    def _new_validation_report(
        self, version: SemanticVersion, artifacts_dir: Path, strict_mode: bool
    ) -> Dict[str, Any]:
        """Create an empty validation report."""
        return {
            "version": str(version),
            "artifacts_dir": str(artifacts_dir),
            "validation_timestamp": self._get_current_timestamp(),
            "strict_mode": strict_mode,
            "total_artifacts": 0,
            "valid_artifacts": 0,
            "invalid_artifacts": 0,
            "missing_artifacts": 0,
            "artifacts": {},
            "errors": [],
            "warnings": [],
        }

    def _artifacts_to_validate(
        self,
        version: SemanticVersion,
        required_artifacts: Optional[List[str]],
        report: Dict[str, Any],
        strict_mode: bool,
    ) -> Dict[str, Optional[str]]:
        """Resolve the manifest entries requested by the caller."""
        manifest = self.artifact_manager.get_manifest(version)
        if manifest:
            if required_artifacts:
                return {
                    artifact: manifest.artifacts.get(artifact)
                    for artifact in required_artifacts
                }
            return manifest.artifacts

        message = f"No manifest found for version {version}"
        report["errors"].append(message)
        if strict_mode:
            raise IntegrityValidationError(message, str(version))
        report["warnings"].append(message)
        return {}

    def _validate_artifact(
        self,
        version: SemanticVersion,
        artifacts_dir: Path,
        artifact_path: str,
        expected_hash: Optional[str],
        report: Dict[str, Any],
        strict_mode: bool,
    ) -> None:
        """Validate one manifest entry and update the shared report."""
        if expected_hash is None:
            self._record_artifact_failure(
                version,
                artifact_path,
                f"Required artifact not found in manifest: {artifact_path}",
                "missing",
                report,
                strict_mode,
            )
            return

        full_path = artifacts_dir / artifact_path
        if not full_path.exists():
            self._record_artifact_failure(
                version,
                artifact_path,
                f"Artifact file missing: {artifact_path}",
                "missing",
                report,
                strict_mode,
            )
            return

        try:
            actual_hash = self.artifact_manager.calculate_file_hash(full_path)
        except Exception as exc:
            self._record_artifact_failure(
                version,
                artifact_path,
                f"Error validating {artifact_path}: {exc}",
                "error",
                report,
                strict_mode,
            )
            return

        if actual_hash == expected_hash:
            report["valid_artifacts"] += 1
            report["artifacts"][artifact_path] = {
                "status": "valid",
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
            }
            return

        self._record_artifact_failure(
            version,
            artifact_path,
            f"Hash mismatch for {artifact_path}: expected {expected_hash}, got {actual_hash}",
            "invalid",
            report,
            strict_mode,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
        )

    @staticmethod
    def _record_artifact_failure(
        version: SemanticVersion,
        artifact_path: str,
        message: str,
        status: str,
        report: Dict[str, Any],
        strict_mode: bool,
        **hashes: str,
    ) -> None:
        report["errors"].append(message)
        counter = "missing_artifacts" if status == "missing" else "invalid_artifacts"
        report[counter] += 1
        report["artifacts"][artifact_path] = {
            "status": status,
            **hashes,
            "error": message,
        }
        if strict_mode:
            raise IntegrityValidationError(message, str(version), [artifact_path])

    @staticmethod
    def _log_validation_result(
        version: SemanticVersion, report: Dict[str, Any], is_valid: bool
    ) -> None:
        if is_valid:
            logger.info(
                "Integrity validation passed for version %s: %s artifacts valid",
                version,
                report["valid_artifacts"],
            )
            return
        logger.warning(
            "Integrity validation failed for version %s: %s invalid, %s missing",
            version,
            report["invalid_artifacts"],
            report["missing_artifacts"],
        )

    @staticmethod
    def _handle_unexpected_error(
        error: Exception,
        version: SemanticVersion,
        report: Dict[str, Any],
        strict_mode: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        message = f"Unexpected error during integrity validation: {error}"
        logger.error(message)
        report["errors"].append(message)
        report["is_valid"] = False
        if strict_mode:
            raise IntegrityValidationError(message, str(version))
        return False, report

    def validate_required_artifacts(
        self,
        version: Union[str, SemanticVersion],
        artifacts_dir: Path,
        required_artifacts: List[str],
        strict_mode: bool = True,
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Validate only the required artifacts for model loading.

        Args:
            version: Model version to validate
            artifacts_dir: Directory containing artifacts to validate
            required_artifacts: List of required artifact files
            strict_mode: If True, fails on any validation error

        Returns:
            Tuple of (is_valid, validation_report)
        """
        return self.validate_model_artifacts(
            version, artifacts_dir, required_artifacts, strict_mode
        )

    def quick_validation(
        self,
        version: Union[str, SemanticVersion],
        artifacts_dir: Path,
        critical_artifacts: List[str],
    ) -> bool:
        """
        Perform quick validation of critical artifacts only.

        Args:
            version: Model version to validate
            artifacts_dir: Directory containing artifacts to validate
            critical_artifacts: List of critical artifact files (e.g., model.pt)

        Returns:
            True if all critical artifacts are valid, False otherwise
        """
        try:
            is_valid, _ = self.validate_required_artifacts(
                version, artifacts_dir, critical_artifacts, strict_mode=False
            )
            return is_valid
        except Exception as e:
            logger.error(f"Quick validation failed: {e}")
            return False

    def get_validation_summary(self, validation_report: Dict[str, any]) -> str:
        """
        Generate a human-readable validation summary.

        Args:
            validation_report: Validation report from validate_model_artifacts

        Returns:
            Human-readable summary string
        """
        version = validation_report.get("version", "unknown")
        total = validation_report.get("total_artifacts", 0)
        valid = validation_report.get("valid_artifacts", 0)
        invalid = validation_report.get("invalid_artifacts", 0)
        missing = validation_report.get("missing_artifacts", 0)
        errors = len(validation_report.get("errors", []))

        summary = f"Integrity validation for {version}: {valid}/{total} artifacts valid"

        if invalid > 0:
            summary += f", {invalid} invalid"
        if missing > 0:
            summary += f", {missing} missing"
        if errors > 0:
            summary += f", {errors} errors"

        return summary

    def clear_validation_cache(self) -> None:
        """Clear the validation cache."""
        self._validation_cache.clear()
        logger.debug("Validation cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get validation cache statistics."""
        return {
            "cached_validations": len(self._validation_cache),
            "cache_keys": list(self._validation_cache.keys()),
        }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelLoaderIntegrityMixin:
    """
    Mixin class to add integrity validation to model loaders.

    This mixin provides methods that can be integrated into existing
    model loading classes to add automatic integrity validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._integrity_validator: Optional[IntegrityValidator] = None
        self._validation_enabled = True

    def set_integrity_validator(self, validator: IntegrityValidator) -> None:
        """Set the integrity validator instance."""
        self._integrity_validator = validator

    def enable_integrity_validation(self, enabled: bool = True) -> None:
        """Enable or disable integrity validation."""
        self._validation_enabled = enabled
        logger.info(f"Integrity validation {'enabled' if enabled else 'disabled'}")

    def validate_before_loading(
        self,
        version: Union[str, SemanticVersion],
        artifacts_dir: Path,
        required_artifacts: List[str],
        strict_mode: bool = True,
    ) -> bool:
        """
        Validate artifacts before model loading.

        Args:
            version: Model version to validate
            artifacts_dir: Directory containing artifacts
            required_artifacts: List of required artifact files
            strict_mode: If True, raises exception on validation failure

        Returns:
            True if validation passed, False otherwise

        Raises:
            IntegrityValidationError: If validation fails in strict mode
        """
        if not self._validation_enabled or not self._integrity_validator:
            logger.debug("Integrity validation skipped (disabled or no validator)")
            return True

        try:
            is_valid, report = self._integrity_validator.validate_required_artifacts(
                version, artifacts_dir, required_artifacts, strict_mode
            )

            if is_valid:
                logger.info(f"Artifact validation passed for version {version}")
            else:
                logger.warning(f"Artifact validation failed for version {version}")
                logger.warning(self._integrity_validator.get_validation_summary(report))

            return is_valid

        except IntegrityValidationError as e:
            logger.error(f"Integrity validation failed: {e}")
            if strict_mode:
                raise
            return False
        except Exception as e:
            logger.error(f"Unexpected error during integrity validation: {e}")
            if strict_mode:
                raise IntegrityValidationError(f"Validation error: {e}", str(version))
            return False
