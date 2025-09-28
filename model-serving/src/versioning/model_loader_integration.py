"""
Integration module for artifact integrity validation with model loading.

This module provides integration between the integrity validation system
and the existing model loading infrastructure.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .artifact_manager import ArtifactManager
from .integrity_validator import IntegrityValidator, IntegrityValidationError
from .semantic_version import SemanticVersion, parse_version


logger = logging.getLogger(__name__)


class ModelLoaderWithIntegrity:
    """
    Model loader with integrated artifact integrity validation.

    This class provides a high-level interface for loading models
    with automatic integrity validation of all artifacts.
    """

    def __init__(self, artifacts_dir: Union[str, Path]):
        """
        Initialize model loader with integrity validation.

        Args:
            artifacts_dir: Base directory for artifact storage
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifact_manager = ArtifactManager(self.artifacts_dir)
        self.integrity_validator = IntegrityValidator(self.artifact_manager)
        self._loaded_models: Dict[str, any] = {}

        logger.info(
            f"Initialized ModelLoaderWithIntegrity with artifacts_dir: {self.artifacts_dir}"
        )

    def load_model(
        self,
        version: Union[str, SemanticVersion],
        required_artifacts: Optional[List[str]] = None,
        strict_validation: bool = True,
        validate_integrity: bool = True,
    ) -> Dict[str, any]:
        """
        Load a model with integrity validation.

        Args:
            version: Model version to load
            required_artifacts: List of required artifact files (if None, validates all)
            strict_validation: If True, fails on any validation error
            validate_integrity: If True, performs integrity validation before loading

        Returns:
            Dictionary containing loaded model and metadata

        Raises:
            IntegrityValidationError: If integrity validation fails
            FileNotFoundError: If model files are not found
        """
        version = parse_version(version)
        version_str = str(version)

        logger.info(f"Loading model version {version_str}")

        # Check if model is already loaded
        if version_str in self._loaded_models:
            logger.debug(f"Model {version_str} already loaded, returning cached version")
            return self._loaded_models[version_str]

        # Get version directory
        version_dir = self.artifacts_dir / "versions" / version_str
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version_str} not found at {version_dir}")

        # Perform integrity validation if requested
        if validate_integrity:
            logger.info(f"Validating integrity for model version {version_str}")

            try:
                is_valid, validation_report = self.integrity_validator.validate_model_artifacts(
                    version, version_dir, required_artifacts, strict_validation
                )

                if not is_valid:
                    error_msg = f"Integrity validation failed for model version {version_str}"
                    logger.error(error_msg)
                    logger.error(self.integrity_validator.get_validation_summary(validation_report))

                    if strict_validation:
                        raise IntegrityValidationError(
                            error_msg, version_str, validation_report.get("failed_artifacts", [])
                        )
                    else:
                        logger.warning(f"Integrity validation failed but continuing: {error_msg}")

                logger.info(f"Integrity validation passed for model version {version_str}")

            except IntegrityValidationError:
                # Re-raise integrity validation errors
                raise
            except Exception as e:
                error_msg = f"Unexpected error during integrity validation: {e}"
                logger.error(error_msg)
                if strict_validation:
                    raise IntegrityValidationError(error_msg, version_str)
                else:
                    logger.warning(f"Integrity validation error but continuing: {error_msg}")

        # Load the model (placeholder implementation)
        model_data = self._load_model_artifacts(version_dir, required_artifacts)

        # Store loaded model
        self._loaded_models[version_str] = model_data

        logger.info(f"Successfully loaded model version {version_str}")
        return model_data

    def unload_model(self, version: Union[str, SemanticVersion]) -> bool:
        """
        Unload a model from memory.

        Args:
            version: Model version to unload

        Returns:
            True if model was unloaded, False if not found
        """
        version_str = str(parse_version(version))

        if version_str in self._loaded_models:
            del self._loaded_models[version_str]
            logger.info(f"Unloaded model version {version_str}")
            return True

        logger.warning(f"Model version {version_str} not found in loaded models")
        return False

    def list_loaded_models(self) -> List[str]:
        """
        List all currently loaded model versions.

        Returns:
            List of loaded model version strings
        """
        return list(self._loaded_models.keys())

    def get_model_info(self, version: Union[str, SemanticVersion]) -> Optional[Dict[str, any]]:
        """
        Get information about a loaded model.

        Args:
            version: Model version to get info for

        Returns:
            Model information dictionary or None if not loaded
        """
        version_str = str(parse_version(version))
        return self._loaded_models.get(version_str)

    def validate_model_integrity(
        self,
        version: Union[str, SemanticVersion],
        required_artifacts: Optional[List[str]] = None,
        strict_mode: bool = True,
    ) -> Dict[str, any]:
        """
        Validate model integrity without loading.

        Args:
            version: Model version to validate
            required_artifacts: List of required artifact files
            strict_mode: If True, raises exception on validation failure

        Returns:
            Validation report dictionary
        """
        version = parse_version(version)
        version_dir = self.artifacts_dir / "versions" / str(version)

        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found at {version_dir}")

        is_valid, report = self.integrity_validator.validate_model_artifacts(
            version, version_dir, required_artifacts, strict_mode
        )

        return report

    def get_available_versions(self) -> List[SemanticVersion]:
        """
        Get list of available model versions.

        Returns:
            List of available versions sorted by version number
        """
        return self.artifact_manager.list_versions()

    def get_model_manifest(self, version: Union[str, SemanticVersion]) -> Optional[Dict[str, any]]:
        """
        Get model manifest for a version.

        Args:
            version: Model version to get manifest for

        Returns:
            Manifest dictionary or None if not found
        """
        version = parse_version(version)
        manifest = self.artifact_manager.get_manifest(version)

        if manifest:
            return manifest.to_dict()

        return None

    def _load_model_artifacts(
        self, version_dir: Path, required_artifacts: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Load model artifacts from directory.

        Args:
            version_dir: Directory containing model artifacts
            required_artifacts: List of required artifact files

        Returns:
            Dictionary containing loaded model data
        """
        # This is a placeholder implementation
        # In a real implementation, this would load the actual model files

        model_data = {
            "version": version_dir.name,
            "artifacts_dir": str(version_dir),
            "loaded_at": self._get_current_timestamp(),
            "artifacts": {},
        }

        # Find and load artifact files
        if required_artifacts:
            artifact_files = [version_dir / artifact for artifact in required_artifacts]
        else:
            # Find all artifact files
            artifact_extensions = {".pt", ".pkl", ".yaml", ".yml", ".json", ".txt", ".md"}
            artifact_files = []
            for file_path in version_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in artifact_extensions:
                    artifact_files.append(file_path)

        # Load each artifact file
        for artifact_file in artifact_files:
            if artifact_file.exists():
                relative_path = artifact_file.relative_to(version_dir)
                try:
                    # In a real implementation, this would load the actual file content
                    # For now, we'll just store metadata
                    model_data["artifacts"][str(relative_path)] = {
                        "path": str(artifact_file),
                        "size": artifact_file.stat().st_size,
                        "loaded": True,
                    }
                except Exception as e:
                    logger.warning(f"Failed to load artifact {artifact_file}: {e}")
                    model_data["artifacts"][str(relative_path)] = {
                        "path": str(artifact_file),
                        "error": str(e),
                        "loaded": False,
                    }

        return model_data

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class IntegrityValidationMiddleware:
    """
    Middleware for adding integrity validation to existing model loaders.

    This class can be used to wrap existing model loading functions
    with integrity validation capabilities.
    """

    def __init__(self, integrity_validator: IntegrityValidator):
        """
        Initialize middleware.

        Args:
            integrity_validator: IntegrityValidator instance
        """
        self.integrity_validator = integrity_validator
        self._validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
        }

    def validate_and_load(
        self,
        load_function,
        version: Union[str, SemanticVersion],
        artifacts_dir: Path,
        required_artifacts: List[str],
        *args,
        **kwargs,
    ) -> any:
        """
        Validate artifacts and then load model using provided function.

        Args:
            load_function: Function to call for actual model loading
            version: Model version to load
            artifacts_dir: Directory containing artifacts
            required_artifacts: List of required artifact files
            *args: Additional arguments for load_function
            **kwargs: Additional keyword arguments for load_function

        Returns:
            Result from load_function

        Raises:
            IntegrityValidationError: If integrity validation fails
        """
        version = parse_version(version)

        # Validate integrity
        is_valid, report = self.integrity_validator.validate_required_artifacts(
            version, artifacts_dir, required_artifacts, strict_mode=True
        )

        self._validation_stats["total_validations"] += 1

        if not is_valid:
            self._validation_stats["failed_validations"] += 1
            error_msg = f"Integrity validation failed for version {version}"
            logger.error(error_msg)
            logger.error(self.integrity_validator.get_validation_summary(report))
            raise IntegrityValidationError(error_msg, str(version))

        self._validation_stats["successful_validations"] += 1

        # Load model using provided function
        return load_function(version, artifacts_dir, *args, **kwargs)

    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self._validation_stats.copy()

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self._validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
        }
