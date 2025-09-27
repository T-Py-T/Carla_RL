"""
Semantic versioning and artifact management utilities.

This module provides comprehensive version management, artifact integrity
validation, and multi-version model support for the Policy-as-a-Service system.
"""

from .semantic_version import SemanticVersion, VersionError, parse_version, validate_version
from .artifact_manager import ArtifactManager, ArtifactIntegrityError, ArtifactManifest
from .integrity_validator import (
    IntegrityValidator,
    IntegrityValidationError,
    ModelLoaderIntegrityMixin,
)
from .model_loader_integration import ModelLoaderWithIntegrity, IntegrityValidationMiddleware
from .rollback_manager import RollbackManager, RollbackError, RollbackOperation
from .version_selector import (
    VersionSelector,
    VersionManager,
    VersionSelectionStrategy,
    VersionSelectionError,
    VersionSelectionResult,
    ModelVersionInfo,
)

__all__ = [
    "SemanticVersion",
    "VersionError",
    "parse_version",
    "validate_version",
    "ArtifactManager",
    "ArtifactIntegrityError",
    "ArtifactManifest",
    "IntegrityValidator",
    "IntegrityValidationError",
    "ModelLoaderIntegrityMixin",
    "ModelLoaderWithIntegrity",
    "IntegrityValidationMiddleware",
    "RollbackManager",
    "RollbackError",
    "RollbackOperation",
    "VersionSelector",
    "VersionManager",
    "VersionSelectionStrategy",
    "VersionSelectionError",
    "VersionSelectionResult",
    "ModelVersionInfo",
]
