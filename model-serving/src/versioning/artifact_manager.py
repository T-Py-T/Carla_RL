"""
Artifact management and integrity validation system.

This module provides comprehensive artifact management with SHA-256 hash pinning,
integrity validation, and content-addressable storage for model artifacts.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .semantic_version import SemanticVersion, parse_version


class ArtifactIntegrityError(Exception):
    """Exception raised for artifact integrity validation errors."""

    def __init__(
        self,
        message: str,
        artifact_path: Optional[Path] = None,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
    ):
        super().__init__(message)
        self.artifact_path = artifact_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


@dataclass
class ArtifactManifest:
    """Manifest containing artifact metadata and integrity information."""

    version: str
    artifacts: Dict[str, str] = field(default_factory=dict)  # filename -> SHA-256 hash
    created_at: str = ""
    model_type: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert manifest to dictionary."""
        return {
            "version": self.version,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "model_type": self.model_type,
            "description": self.description,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ArtifactManifest":
        """Create manifest from dictionary."""
        return cls(
            version=data.get("version", ""),
            artifacts=data.get("artifacts", {}),
            created_at=data.get("created_at", ""),
            model_type=data.get("model_type", ""),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
        )


class ArtifactManager:
    """
    Manages model artifacts with SHA-256 hash pinning and integrity validation.

    Provides content-addressable storage, integrity checking, and artifact
    management for the Policy-as-a-Service system.
    """

    def __init__(self, artifacts_dir: Union[str, Path]):
        """
        Initialize artifact manager.

        Args:
            artifacts_dir: Base directory for storing artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.manifests_dir = self.artifacts_dir / "manifests"
        self.versions_dir = self.artifacts_dir / "versions"

        # Create directories if they don't exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as hexadecimal string

        Raises:
            ArtifactIntegrityError: If file cannot be read
        """
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except (OSError, IOError) as e:
            raise ArtifactIntegrityError(f"Cannot calculate hash for {file_path}: {e}", file_path)

    def pin_artifacts(
        self, version: Union[str, SemanticVersion], artifacts_dir: Path
    ) -> ArtifactManifest:
        """
        Pin artifacts by calculating and storing their SHA-256 hashes.

        Args:
            version: Model version
            artifacts_dir: Directory containing artifacts to pin

        Returns:
            ArtifactManifest with pinned hashes

        Raises:
            ArtifactIntegrityError: If artifacts cannot be processed
        """
        version = parse_version(version)
        artifacts_dir = Path(artifacts_dir)

        if not artifacts_dir.exists():
            raise ArtifactIntegrityError(f"Artifacts directory does not exist: {artifacts_dir}")

        manifest = ArtifactManifest(version=str(version))
        manifest.created_at = self._get_current_timestamp()

        # Find all artifact files
        artifact_files = self._find_artifact_files(artifacts_dir)

        if not artifact_files:
            raise ArtifactIntegrityError(f"No artifact files found in {artifacts_dir}")

        # Calculate hashes for each artifact
        for artifact_file in artifact_files:
            relative_path = artifact_file.relative_to(artifacts_dir)
            file_hash = self.calculate_file_hash(artifact_file)
            manifest.artifacts[str(relative_path)] = file_hash

        # Save manifest
        self._save_manifest(manifest)

        return manifest

    def validate_artifacts(self, version: Union[str, SemanticVersion], artifacts_dir: Path) -> bool:
        """
        Validate artifact integrity against pinned hashes.

        Args:
            version: Model version to validate
            artifacts_dir: Directory containing artifacts to validate

        Returns:
            True if all artifacts are valid, False otherwise

        Raises:
            ArtifactIntegrityError: If validation fails
        """
        version = parse_version(version)
        artifacts_dir = Path(artifacts_dir)

        # Load manifest
        manifest = self._load_manifest(version)
        if not manifest:
            raise ArtifactIntegrityError(f"No manifest found for version {version}")

        # Validate each artifact
        for artifact_path, expected_hash in manifest.artifacts.items():
            full_path = artifacts_dir / artifact_path

            if not full_path.exists():
                raise ArtifactIntegrityError(
                    f"Artifact file missing: {artifact_path}", full_path, expected_hash
                )

            actual_hash = self.calculate_file_hash(full_path)
            if actual_hash != expected_hash:
                raise ArtifactIntegrityError(
                    f"Artifact integrity check failed for {artifact_path}",
                    full_path,
                    expected_hash,
                    actual_hash,
                )

        return True

    def get_artifact_hash(
        self, version: Union[str, SemanticVersion], artifact_path: str
    ) -> Optional[str]:
        """
        Get the pinned hash for a specific artifact.

        Args:
            version: Model version
            artifact_path: Relative path to the artifact

        Returns:
            SHA-256 hash if found, None otherwise
        """
        version = parse_version(version)
        manifest = self._load_manifest(version)

        if not manifest:
            return None

        return manifest.artifacts.get(artifact_path)

    def list_artifacts(self, version: Union[str, SemanticVersion]) -> List[str]:
        """
        List all artifacts for a version.

        Args:
            version: Model version

        Returns:
            List of artifact paths
        """
        version = parse_version(version)
        manifest = self._load_manifest(version)

        if not manifest:
            return []

        return list(manifest.artifacts.keys())

    def get_manifest(self, version: Union[str, SemanticVersion]) -> Optional[ArtifactManifest]:
        """
        Get the manifest for a version.

        Args:
            version: Model version

        Returns:
            ArtifactManifest if found, None otherwise
        """
        version = parse_version(version)
        return self._load_manifest(version)

    def list_versions(self) -> List[SemanticVersion]:
        """
        List all available versions.

        Returns:
            List of available versions sorted by version number
        """
        versions = []

        for manifest_file in self.manifests_dir.glob("*.json"):
            try:
                version_str = manifest_file.stem
                version = parse_version(version_str)
                versions.append(version)
            except Exception:
                continue  # Skip invalid manifest files

        return sorted(versions)

    def delete_version(self, version: Union[str, SemanticVersion]) -> bool:
        """
        Delete a version and its artifacts.

        Args:
            version: Model version to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        version = parse_version(version)

        # Delete manifest
        manifest_file = self.manifests_dir / f"{version}.json"
        if manifest_file.exists():
            manifest_file.unlink()

        # Delete version directory
        version_dir = self.versions_dir / str(version)
        if version_dir.exists():
            import shutil

            shutil.rmtree(version_dir)

        return True

    def copy_artifacts(
        self, version: Union[str, SemanticVersion], source_dir: Path, validate: bool = True
    ) -> Path:
        """
        Copy artifacts to versioned storage with integrity validation.

        Args:
            version: Model version
            source_dir: Source directory containing artifacts
            validate: Whether to validate integrity after copying

        Returns:
            Path to the copied artifacts directory
        """
        version = parse_version(version)
        source_dir = Path(source_dir)

        # Create version directory
        version_dir = self.versions_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts
        import shutil

        shutil.copytree(source_dir, version_dir, dirs_exist_ok=True)

        # Validate if requested and manifest exists
        if validate:
            manifest = self._load_manifest(version)
            if manifest:
                self.validate_artifacts(version, version_dir)

        return version_dir

    def _find_artifact_files(self, artifacts_dir: Path) -> List[Path]:
        """Find all artifact files in a directory."""
        artifact_extensions = {".pt", ".pkl", ".yaml", ".yml", ".json", ".txt", ".md"}
        artifact_files = []

        for file_path in artifacts_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in artifact_extensions:
                artifact_files.append(file_path)

        return sorted(artifact_files)

    def _save_manifest(self, manifest: ArtifactManifest) -> None:
        """Save manifest to file."""
        manifest_file = self.manifests_dir / f"{manifest.version}.json"

        with open(manifest_file, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def _load_manifest(self, version: SemanticVersion) -> Optional[ArtifactManifest]:
        """Load manifest from file."""
        manifest_file = self.manifests_dir / f"{version}.json"

        if not manifest_file.exists():
            return None

        try:
            with open(manifest_file, "r") as f:
                data = json.load(f)
            return ArtifactManifest.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def verify_integrity(self, version: Union[str, SemanticVersion]) -> Dict[str, bool]:
        """
        Verify integrity of all artifacts for a version.

        Args:
            version: Model version to verify

        Returns:
            Dictionary mapping artifact paths to validation results
        """
        version = parse_version(version)
        manifest = self._load_manifest(version)

        if not manifest:
            return {}

        results = {}
        version_dir = self.versions_dir / str(version)

        for artifact_path, expected_hash in manifest.artifacts.items():
            full_path = version_dir / artifact_path

            if not full_path.exists():
                results[artifact_path] = False
                continue

            try:
                actual_hash = self.calculate_file_hash(full_path)
                results[artifact_path] = actual_hash == expected_hash
            except ArtifactIntegrityError:
                results[artifact_path] = False

        return results

    def get_integrity_report(self, version: Union[str, SemanticVersion]) -> Dict:
        """
        Generate comprehensive integrity report for a version.

        Args:
            version: Model version to report on

        Returns:
            Detailed integrity report
        """
        version = parse_version(version)
        manifest = self._load_manifest(version)

        if not manifest:
            return {
                "version": str(version),
                "status": "not_found",
                "message": "No manifest found for this version",
            }

        integrity_results = self.verify_integrity(version)
        all_valid = all(integrity_results.values())

        return {
            "version": str(version),
            "status": "valid" if all_valid else "invalid",
            "total_artifacts": len(manifest.artifacts),
            "valid_artifacts": sum(integrity_results.values()),
            "invalid_artifacts": len(integrity_results) - sum(integrity_results.values()),
            "artifacts": integrity_results,
            "manifest": manifest.to_dict(),
        }
