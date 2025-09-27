"""
Semantic versioning parser and validation for model artifacts.

This module implements semantic versioning (SemVer) parsing with validation
for model version management and comparison.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, order=True)
class SemanticVersion:
    """
    Semantic version representation with parsing and comparison support.
    
    Follows semantic versioning specification (https://semver.org/):
    - Version format: vMAJOR.MINOR.PATCH[-prerelease][+build]
    - Comparison follows semantic versioning rules
    - Immutable dataclass for safe use in collections
    """
    
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    # Regex pattern for semantic version parsing
    _SEMVER_PATTERN = re.compile(
        r'^v?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)'
        r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
        r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
        r'(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    )
    
    def __post_init__(self):
        """Validate semantic version components."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Version numbers must be non-negative")
        
        if self.prerelease is not None and not self.prerelease:
            raise ValueError("Prerelease identifier cannot be empty")
            
        if self.build is not None and not self.build:
            raise ValueError("Build metadata cannot be empty")
    
    @classmethod
    def parse(cls, version_str: str) -> 'SemanticVersion':
        """
        Parse a semantic version string into a SemanticVersion object.
        
        Args:
            version_str: Version string (e.g., "v1.2.3", "1.0.0-alpha", "2.1.0+build.123")
            
        Returns:
            SemanticVersion object
            
        Raises:
            ValueError: If version string is invalid
            
        Examples:
            >>> SemanticVersion.parse("v1.0.0")
            SemanticVersion(major=1, minor=0, patch=0)
            >>> SemanticVersion.parse("2.1.3-alpha+build.123")
            SemanticVersion(major=2, minor=1, patch=3, prerelease='alpha', build='build.123')
        """
        if not isinstance(version_str, str):
            raise ValueError(f"Version must be a string, got {type(version_str)}")
        
        match = cls._SEMVER_PATTERN.match(version_str.strip())
        if not match:
            raise ValueError(f"Invalid semantic version format: {version_str}")
        
        groups = match.groupdict()
        
        return cls(
            major=int(groups['major']),
            minor=int(groups['minor']),
            patch=int(groups['patch']),
            prerelease=groups.get('prerelease'),
            build=groups.get('build')
        )
    
    def __str__(self) -> str:
        """Return string representation of the version."""
        version = f"v{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"SemanticVersion(major={self.major}, minor={self.minor}, patch={self.patch}, prerelease={self.prerelease!r}, build={self.build!r})"
    
    def to_version_string(self, include_v_prefix: bool = True) -> str:
        """
        Convert to version string with optional 'v' prefix.
        
        Args:
            include_v_prefix: Whether to include 'v' prefix
            
        Returns:
            Version string
        """
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return f"v{version}" if include_v_prefix else version
    
    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return self.prerelease is not None
    
    def is_stable(self) -> bool:
        """Check if this is a stable release (not prerelease)."""
        return self.prerelease is None
    
    def bump_major(self) -> 'SemanticVersion':
        """Create new version with major version incremented."""
        return SemanticVersion(self.major + 1, 0, 0)
    
    def bump_minor(self) -> 'SemanticVersion':
        """Create new version with minor version incremented."""
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> 'SemanticVersion':
        """Create new version with patch version incremented."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)
    
    def is_compatible_with(self, other: 'SemanticVersion') -> bool:
        """
        Check if this version is backward compatible with another version.
        
        Follows semantic versioning compatibility rules:
        - Same major version for stable releases (major > 0)
        - Patch and minor version differences are compatible
        - Major version 0.x.x has no compatibility guarantees
        
        Args:
            other: Version to check compatibility against
            
        Returns:
            True if versions are compatible
        """
        if not isinstance(other, SemanticVersion):
            return False
        
        # No compatibility guarantees for major version 0
        if self.major == 0 or other.major == 0:
            return self == other
        
        # Compatible if same major version and this version is newer or equal
        return self.major == other.major and self >= other


def parse_version(version_str: str) -> SemanticVersion:
    """
    Convenience function to parse a version string.
    
    Args:
        version_str: Version string to parse
        
    Returns:
        SemanticVersion object
        
    Raises:
        ValueError: If version string is invalid
    """
    return SemanticVersion.parse(version_str)


def validate_version_format(version_str: str) -> bool:
    """
    Validate if a string follows semantic versioning format.
    
    Args:
        version_str: Version string to validate
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        SemanticVersion.parse(version_str)
        return True
    except ValueError:
        return False


def sort_versions(versions: list[str], descending: bool = True) -> list[str]:
    """
    Sort version strings according to semantic versioning rules.
    
    Args:
        versions: List of version strings to sort
        descending: If True, sort in descending order (newest first)
        
    Returns:
        Sorted list of version strings
        
    Raises:
        ValueError: If any version string is invalid
    """
    parsed_versions = [(SemanticVersion.parse(v), v) for v in versions]
    parsed_versions.sort(key=lambda x: x[0], reverse=descending)
    return [v[1] for v in parsed_versions]