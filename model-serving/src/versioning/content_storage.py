"""
Content-addressable storage system for artifact integrity.

This module provides a content-addressable storage system that uses
content hashes as addresses, ensuring data integrity and deduplication.
"""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import sqlite3
import threading

from .semantic_version import SemanticVersion, parse_version
from .artifact_manager import ArtifactManager, ArtifactManifest, ArtifactIntegrityError


logger = logging.getLogger(__name__)


class ContentStorageError(Exception):
    """Exception raised for content storage errors."""

    def __init__(
        self, message: str, content_hash: Optional[str] = None, operation: Optional[str] = None
    ):
        super().__init__(message)
        self.content_hash = content_hash
        self.operation = operation


@dataclass
class ContentReference:
    """Reference to content in the storage system."""

    content_hash: str
    size: int
    created_at: str
    last_accessed: str
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "content_hash": self.content_hash,
            "size": self.size,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ContentReference":
        """Create from dictionary."""
        return cls(
            content_hash=data["content_hash"],
            size=data["size"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StorageStats:
    """Storage system statistics."""

    total_objects: int
    total_size: int
    unique_hashes: int
    duplicate_objects: int
    storage_efficiency: float  # percentage of space saved through deduplication
    oldest_object: Optional[str] = None
    newest_object: Optional[str] = None
    most_accessed: Optional[str] = None
    least_accessed: Optional[str] = None


class ContentAddressableStorage:
    """
    Content-addressable storage system for artifact integrity.

    Provides a content-addressable storage system that uses content hashes
    as addresses, ensuring data integrity, deduplication, and efficient storage.
    """

    def __init__(
        self,
        storage_dir: Union[str, Path],
        hash_algorithm: str = "sha256",
        max_size: Optional[int] = None,
    ):
        """
        Initialize content-addressable storage.

        Args:
            storage_dir: Base directory for storage
            hash_algorithm: Hash algorithm to use (sha256, sha1, md5)
            max_size: Maximum storage size in bytes (None for unlimited)
        """
        self.storage_dir = Path(storage_dir)
        self.hash_algorithm = hash_algorithm
        self.max_size = max_size

        # Create storage directories
        self.objects_dir = self.storage_dir / "objects"
        self.index_dir = self.storage_dir / "index"
        self.temp_dir = self.storage_dir / "temp"

        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.index_dir / "content_index.db"
        self._init_database()

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"Initialized ContentAddressableStorage at {self.storage_dir}")

    def store_content(
        self, content: Union[bytes, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store content and return its hash.

        Args:
            content: Content to store (bytes or file path)
            metadata: Optional metadata

        Returns:
            Content hash

        Raises:
            ContentStorageError: If storage fails
        """
        with self._lock:
            try:
                # Calculate content hash
                if isinstance(content, Path):
                    content_hash = self._calculate_file_hash(content)
                    content_size = content.stat().st_size
                else:
                    content_hash = self._calculate_bytes_hash(content)
                    content_size = len(content)

                # Check if content already exists
                if self._content_exists(content_hash):
                    logger.debug(f"Content already exists: {content_hash}")
                    # Update access info for existing content
                    self._update_access_info(content_hash)
                    # Create a new reference for tracking purposes (duplicate)
                    now = self._get_current_timestamp()
                    reference = ContentReference(
                        content_hash=content_hash,
                        size=content_size,
                        created_at=now,
                        last_accessed=now,
                        access_count=1,
                        metadata=metadata or {},
                    )
                    self._store_reference(reference)
                    return content_hash

                # Check storage limits
                if self.max_size and self._get_total_size() + content_size > self.max_size:
                    raise ContentStorageError(
                        f"Storage limit exceeded: {self.max_size} bytes", content_hash, "store"
                    )

                # Store content
                content_path = self._get_content_path(content_hash)
                content_path.parent.mkdir(parents=True, exist_ok=True)

                if isinstance(content, Path):
                    shutil.copy2(content, content_path)
                else:
                    with open(content_path, "wb") as f:
                        f.write(content)

                # Create content reference
                now = self._get_current_timestamp()
                reference = ContentReference(
                    content_hash=content_hash,
                    size=content_size,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    metadata=metadata or {},
                )

                # Store reference in database
                self._store_reference(reference)

                logger.info(f"Stored content: {content_hash} ({content_size} bytes)")
                return content_hash

            except Exception as e:
                logger.error(f"Failed to store content: {e}")
                raise ContentStorageError(f"Failed to store content: {e}", operation="store")

    def retrieve_content(self, content_hash: str) -> bytes:
        """
        Retrieve content by hash.

        Args:
            content_hash: Content hash

        Returns:
            Content bytes

        Raises:
            ContentStorageError: If content not found
        """
        with self._lock:
            if not self._content_exists(content_hash):
                raise ContentStorageError(
                    f"Content not found: {content_hash}", content_hash, "retrieve"
                )

            content_path = self._get_content_path(content_hash)

            try:
                with open(content_path, "rb") as f:
                    content = f.read()

                # Update access info
                self._update_access_info(content_hash)

                logger.debug(f"Retrieved content: {content_hash}")
                return content

            except Exception as e:
                logger.error(f"Failed to retrieve content {content_hash}: {e}")
                raise ContentStorageError(
                    f"Failed to retrieve content: {e}", content_hash, "retrieve"
                )

    def copy_content_to(self, content_hash: str, target_path: Path) -> None:
        """
        Copy content to a file path.

        Args:
            content_hash: Content hash
            target_path: Target file path

        Raises:
            ContentStorageError: If content not found or copy fails
        """
        with self._lock:
            if not self._content_exists(content_hash):
                raise ContentStorageError(
                    f"Content not found: {content_hash}", content_hash, "copy"
                )

            source_path = self._get_content_path(content_hash)

            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)

                # Update access info
                self._update_access_info(content_hash)

                logger.debug(f"Copied content {content_hash} to {target_path}")

            except Exception as e:
                logger.error(f"Failed to copy content {content_hash}: {e}")
                raise ContentStorageError(f"Failed to copy content: {e}", content_hash, "copy")

    def content_exists(self, content_hash: str) -> bool:
        """
        Check if content exists.

        Args:
            content_hash: Content hash

        Returns:
            True if content exists
        """
        with self._lock:
            return self._content_exists(content_hash)

    def get_content_info(self, content_hash: str) -> Optional[ContentReference]:
        """
        Get content information.

        Args:
            content_hash: Content hash

        Returns:
            ContentReference if found, None otherwise
        """
        with self._lock:
            return self._get_content_reference(content_hash)

    def list_content(self, limit: Optional[int] = None) -> List[ContentReference]:
        """
        List all content references.

        Args:
            limit: Maximum number of references to return

        Returns:
            List of ContentReference objects
        """
        with self._lock:
            references = self._get_all_references()

            if limit:
                references = references[:limit]

            return references

    def delete_content(self, content_hash: str) -> bool:
        """
        Delete content.

        Args:
            content_hash: Content hash

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if not self._content_exists(content_hash):
                return False

            try:
                # Remove content file
                content_path = self._get_content_path(content_hash)
                if content_path.exists():
                    content_path.unlink()

                # Remove from database
                self._remove_reference(content_hash)

                logger.info(f"Deleted content: {content_hash}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete content {content_hash}: {e}")
                return False

    def get_storage_stats(self) -> StorageStats:
        """
        Get storage statistics.

        Returns:
            StorageStats object
        """
        with self._lock:
            references = self._get_all_references()

            if not references:
                return StorageStats(
                    total_objects=0,
                    total_size=0,
                    unique_hashes=0,
                    duplicate_objects=0,
                    storage_efficiency=0.0,
                )

            total_objects = len(references)
            total_size = sum(ref.size for ref in references)
            unique_hashes = len(set(ref.content_hash for ref in references))
            duplicate_objects = total_objects - unique_hashes

            # Calculate storage efficiency
            if total_objects > 0:
                storage_efficiency = (duplicate_objects / total_objects) * 100
            else:
                storage_efficiency = 0.0

            # Find oldest and newest objects
            sorted_by_created = sorted(references, key=lambda r: r.created_at)
            oldest_object = sorted_by_created[0].content_hash if sorted_by_created else None
            newest_object = sorted_by_created[-1].content_hash if sorted_by_created else None

            # Find most and least accessed objects
            sorted_by_access = sorted(references, key=lambda r: r.access_count)
            least_accessed = sorted_by_access[0].content_hash if sorted_by_access else None
            most_accessed = sorted_by_access[-1].content_hash if sorted_by_access else None

            return StorageStats(
                total_objects=total_objects,
                total_size=total_size,
                unique_hashes=unique_hashes,
                duplicate_objects=duplicate_objects,
                storage_efficiency=storage_efficiency,
                oldest_object=oldest_object,
                newest_object=newest_object,
                most_accessed=most_accessed,
                least_accessed=least_accessed,
            )

    def cleanup_orphaned_content(self) -> int:
        """
        Clean up orphaned content (content without references).

        Returns:
            Number of orphaned files cleaned up
        """
        with self._lock:
            orphaned_count = 0

            # Get all content hashes from database
            db_hashes = set(self._get_all_content_hashes())

            # Find orphaned files
            for content_file in self.objects_dir.rglob("*"):
                if content_file.is_file():
                    # Extract hash from path
                    relative_path = content_file.relative_to(self.objects_dir)
                    content_hash = str(relative_path).replace(os.sep, "")

                    if content_hash not in db_hashes:
                        try:
                            content_file.unlink()
                            orphaned_count += 1
                            logger.debug(f"Cleaned up orphaned file: {content_file}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up orphaned file {content_file}: {e}")

            logger.info(f"Cleaned up {orphaned_count} orphaned files")
            return orphaned_count

    def verify_integrity(self) -> Dict[str, bool]:
        """
        Verify integrity of all stored content.

        Returns:
            Dictionary mapping content hashes to integrity status
        """
        with self._lock:
            results = {}

            for reference in self._get_all_references():
                content_path = self._get_content_path(reference.content_hash)

                if not content_path.exists():
                    results[reference.content_hash] = False
                    continue

                try:
                    # Verify file hash
                    actual_hash = self._calculate_file_hash(content_path)
                    results[reference.content_hash] = actual_hash == reference.content_hash
                except Exception as e:
                    logger.warning(f"Failed to verify integrity for {reference.content_hash}: {e}")
                    results[reference.content_hash] = False

            return results

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file."""
        hash_obj = hashlib.new(self.hash_algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _calculate_bytes_hash(self, content: bytes) -> str:
        """Calculate hash of bytes content."""
        hash_obj = hashlib.new(self.hash_algorithm)
        hash_obj.update(content)
        return hash_obj.hexdigest()

    def _get_content_path(self, content_hash: str) -> Path:
        """Get file path for content hash."""
        # Use first 2 characters as directory for better distribution
        return self.objects_dir / content_hash[:2] / content_hash

    def _content_exists(self, content_hash: str) -> bool:
        """Check if content exists in storage."""
        content_path = self._get_content_path(content_hash)
        return content_path.exists() and self._get_content_reference(content_hash) is not None

    def _get_content_reference(self, content_hash: str) -> Optional[ContentReference]:
        """Get content reference from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM content_refs WHERE content_hash = ? ORDER BY id DESC LIMIT 1",
                (content_hash,),
            )
            row = cursor.fetchone()

            if row:
                data = json.loads(row[0])
                return ContentReference.from_dict(data)

            return None

    def _get_all_references(self) -> List[ContentReference]:
        """Get all content references from database."""
        references = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM content_refs")

            for row in cursor.fetchall():
                data = json.loads(row[0])
                references.append(ContentReference.from_dict(data))

        return references

    def _get_all_content_hashes(self) -> List[str]:
        """Get all content hashes from database."""
        hashes = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT content_hash FROM content_refs")

            for row in cursor.fetchall():
                hashes.append(row[0])

        return hashes

    def _store_reference(self, reference: ContentReference) -> None:
        """Store content reference in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO content_refs (content_hash, data) VALUES (?, ?)",
                (reference.content_hash, json.dumps(reference.to_dict())),
            )
            conn.commit()

    def _update_access_info(self, content_hash: str) -> None:
        """Update access information for content."""
        reference = self._get_content_reference(content_hash)
        if reference:
            reference.last_accessed = self._get_current_timestamp()
            reference.access_count += 1
            self._store_reference(reference)

    def _remove_reference(self, content_hash: str) -> None:
        """Remove content reference from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM content_refs WHERE content_hash = ?", (content_hash,))
            conn.commit()

    def _get_total_size(self) -> int:
        """Get total size of all stored content."""
        references = self._get_all_references()
        return sum(ref.size for ref in references)

    def _init_database(self) -> None:
        """Initialize the content index database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS content_refs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    data TEXT NOT NULL,
                    UNIQUE(content_hash, id)
                )
            """
            )
            conn.commit()

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ContentAddressableArtifactManager(ArtifactManager):
    """
    Enhanced ArtifactManager with content-addressable storage.

    Extends the base ArtifactManager to use content-addressable storage
    for improved integrity and deduplication.
    """

    def __init__(
        self,
        artifacts_dir: Union[str, Path],
        content_storage_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize content-addressable artifact manager.

        Args:
            artifacts_dir: Base directory for artifacts
            content_storage_dir: Directory for content-addressable storage
        """
        super().__init__(artifacts_dir)

        if content_storage_dir is None:
            content_storage_dir = self.artifacts_dir / "content_storage"

        self.content_storage = ContentAddressableStorage(content_storage_dir)
        logger.info("Initialized ContentAddressableArtifactManager")

    def pin_artifacts(
        self, version: Union[str, SemanticVersion], artifacts_dir: Path
    ) -> ArtifactManifest:
        """
        Pin artifacts using content-addressable storage.

        Args:
            version: Model version
            artifacts_dir: Directory containing artifacts to pin

        Returns:
            ArtifactManifest with content hashes
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

        # Store artifacts in content-addressable storage and get hashes
        for artifact_file in artifact_files:
            relative_path = artifact_file.relative_to(artifacts_dir)

            # Store in content-addressable storage
            content_hash = self.content_storage.store_content(
                artifact_file,
                metadata={
                    "version": str(version),
                    "artifact_path": str(relative_path),
                    "original_size": artifact_file.stat().st_size,
                },
            )

            manifest.artifacts[str(relative_path)] = content_hash

        # Save manifest
        self._save_manifest(manifest)

        return manifest

    def validate_artifacts(self, version: Union[str, SemanticVersion], artifacts_dir: Path) -> bool:
        """
        Validate artifact integrity using content-addressable storage.

        Args:
            version: Model version to validate
            artifacts_dir: Directory containing artifacts to validate

        Returns:
            True if all artifacts are valid
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

            # Verify content hash using content-addressable storage
            if not self.content_storage.content_exists(expected_hash):
                raise ArtifactIntegrityError(
                    f"Artifact content not found in storage: {artifact_path}",
                    full_path,
                    expected_hash,
                )

            # Verify file integrity
            actual_hash = self.content_storage._calculate_file_hash(full_path)
            if actual_hash != expected_hash:
                raise ArtifactIntegrityError(
                    f"Artifact integrity check failed for {artifact_path}",
                    full_path,
                    expected_hash,
                    actual_hash,
                )

        return True

    def get_content_storage_stats(self) -> StorageStats:
        """Get content storage statistics."""
        return self.content_storage.get_storage_stats()

    def cleanup_content_storage(self) -> int:
        """Clean up orphaned content in storage."""
        return self.content_storage.cleanup_orphaned_content()

    def verify_content_integrity(self) -> Dict[str, bool]:
        """Verify integrity of all content in storage."""
        return self.content_storage.verify_integrity()
