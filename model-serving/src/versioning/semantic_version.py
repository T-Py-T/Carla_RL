"""
Semantic versioning parser and validation utilities.

This module provides comprehensive semantic version parsing, validation,
and comparison functionality following the vMAJOR.MINOR.PATCH format.
Semantic versioning parser and validation for model artifacts.

This module implements semantic versioning (SemVer) parsing with validation
for model version management and comparison.
"""

import re
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple


class VersionError(Exception):
    """Exception raised for version parsing and validation errors."""

    def __init__(self, message: str, version_string: Optional[str] = None):
        super().__init__(message)
        self.version_string = version_string


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
    build_metadata: Optional[str] = None

    def __post_init__(self):
        """Validate version components after initialization."""
        if self.major < 0:
            raise VersionError(f"Major version must be non-negative, got {self.major}")
        if self.minor < 0:
            raise VersionError(f"Minor version must be non-negative, got {self.minor}")
        if self.patch < 0:
            raise VersionError(f"Patch version must be non-negative, got {self.patch}")

    def __str__(self) -> str:
        """Return string representation of the version."""
        version_str = f"v{self.major}.{self.minor}.{self.patch}"

        if self.prerelease:
            version_str += f"-{self.prerelease}"

        if self.build_metadata:
            version_str += f"+{self.build_metadata}"

        return version_str

    def __repr__(self) -> str:
        """Return detailed representation of the version."""
        return f"SemanticVersion(major={self.major}, minor={self.minor}, patch={self.patch}, prerelease={self.prerelease}, build_metadata={self.build_metadata})"

    def __eq__(self, other) -> bool:
        """Check if two versions are equal."""
        if not isinstance(other, SemanticVersion):
            return False

        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other) -> bool:
        """Check if this version is less than another version."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented

        # Compare core version numbers
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch

        # Handle prerelease versions
        if self.prerelease is None and other.prerelease is not None:
            return False  # Stable version is greater than prerelease
        if self.prerelease is not None and other.prerelease is None:
            return True  # Prerelease is less than stable version
        if self.prerelease is not None and other.prerelease is not None:
            return self._compare_prerelease(self.prerelease, other.prerelease)

        return False

    def __le__(self, other) -> bool:
        """Check if this version is less than or equal to another version."""
        return self == other or self < other

    def __gt__(self, other) -> bool:
        """Check if this version is greater than another version."""
        return not self <= other

    def __ge__(self, other) -> bool:
        """Check if this version is greater than or equal to another version."""
        return not self < other

    def __hash__(self) -> int:
        """Return hash of the version."""
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def _compare_prerelease(self, prerelease1: str, prerelease2: str) -> bool:
        """
        Compare two prerelease strings.

        Returns True if prerelease1 < prerelease2, False otherwise.
        """
        # Split prerelease identifiers
        identifiers1 = prerelease1.split(".")
        identifiers2 = prerelease2.split(".")

        # Compare each identifier
        max_length = max(len(identifiers1), len(identifiers2))

        for i in range(max_length):
            id1 = identifiers1[i] if i < len(identifiers1) else None
            id2 = identifiers2[i] if i < len(identifiers2) else None

            # If one is shorter, it's less than the other
            if id1 is None:
                return True
            if id2 is None:
                return False

            # Compare identifiers
            if id1 == id2:
                continue

            # Check if both are numeric
            try:
                num1 = int(id1)
                num2 = int(id2)
                return num1 < num2
            except ValueError:
                # At least one is not numeric, compare as strings
                return id1 < id2

        return False

    def is_stable(self) -> bool:
        """Check if this is a stable release (no prerelease)."""
        return self.prerelease is None

    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return self.prerelease is not None

    def get_core_version(self) -> Tuple[int, int, int]:
        """Get the core version as a tuple (major, minor, patch)."""
        return (self.major, self.minor, self.patch)

    def get_next_major(self) -> "SemanticVersion":
        """Get the next major version."""
        return SemanticVersion(major=self.major + 1, minor=0, patch=0)

    def get_next_minor(self) -> "SemanticVersion":
        """Get the next minor version."""
        return SemanticVersion(major=self.major, minor=self.minor + 1, patch=0)

    def get_next_patch(self) -> "SemanticVersion":
        """Get the next patch version."""
        return SemanticVersion(major=self.major, minor=self.minor, patch=self.patch + 1)

    def satisfies(self, constraint: str) -> bool:
        """
        Check if this version satisfies a version constraint.

        Supports basic constraints like:
        - ">=1.0.0" - greater than or equal to
        - "<=2.0.0" - less than or equal to
        - "~1.0.0" - compatible with (same major, minor >= specified)
        - "^1.0.0" - compatible with (same major, minor and patch >= specified)
        - "1.0.0" - exact match
        """
        constraint = constraint.strip()

        if constraint.startswith(">="):
            version = parse_version(constraint[2:])
            return self >= version
        elif constraint.startswith("<="):
            version = parse_version(constraint[2:])
            return self <= version
        elif constraint.startswith(">"):
            version = parse_version(constraint[1:])
            return self > version
        elif constraint.startswith("<"):
            version = parse_version(constraint[1:])
            return self < version
        elif constraint.startswith("~"):
            version = parse_version(constraint[1:])
            return (
                self.major == version.major
                and self.minor == version.minor
                and self.patch >= version.patch
            )
        elif constraint.startswith("^"):
            version = parse_version(constraint[1:])
            return (
                self.major == version.major
                and self.minor >= version.minor
                and self.patch >= version.patch
            )
        else:
            # Exact match
            version = parse_version(constraint)
            return self == version


# Version parsing regex pattern - strict validation
VERSION_PATTERN = re.compile(
    r"^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*))?(?:\+([a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*))?$"
)


def parse_version(version_string: Union[str, SemanticVersion]) -> SemanticVersion:
    """
    Parse a version string into a SemanticVersion object.

    Args:
        version_string: Version string in vMAJOR.MINOR.PATCH format

    Returns:
        SemanticVersion object

    Raises:
        VersionError: If the version string is invalid

    Examples:
        >>> parse_version("v1.0.0")
        SemanticVersion(major=1, minor=0, patch=0, prerelease=None, build_metadata=None)
        >>> parse_version("1.2.3-alpha.1+build.123")
        SemanticVersion(major=1, minor=2, patch=3, prerelease='alpha.1', build_metadata='build.123')
    """
    if isinstance(version_string, SemanticVersion):
        return version_string

    if not isinstance(version_string, str):
        raise VersionError(f"Version must be a string, got {type(version_string)}")

    version_string = version_string.strip()

    if not version_string:
        raise VersionError("Version string cannot be empty")

    match = VERSION_PATTERN.match(version_string)
    if not match:
        raise VersionError(
            f"Invalid version format: '{version_string}'. Expected format: vMAJOR.MINOR.PATCH[-prerelease][+build]",
            version_string,
        )

    major, minor, patch, prerelease, build_metadata = match.groups()

    try:
        major_int = int(major)
        minor_int = int(minor)
        patch_int = int(patch)
    except ValueError as e:
        raise VersionError(f"Version numbers must be integers: {e}", version_string)

    return SemanticVersion(
        major=major_int,
        minor=minor_int,
        patch=patch_int,
        prerelease=prerelease,
        build_metadata=build_metadata,
    )


def validate_version(version_string: str) -> bool:
    """
    Validate if a version string is in the correct format.

    Args:
        version_string: Version string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_version(version_string)
        return True
    except VersionError:
        return False


def sort_versions(versions: List[Union[str, SemanticVersion]]) -> List[SemanticVersion]:
    """
    Sort a list of version strings or SemanticVersion objects.

    Args:
        versions: List of version strings or SemanticVersion objects

    Returns:
        Sorted list of SemanticVersion objects (ascending order)
    """
    parsed_versions = []

    for version in versions:
        try:
            parsed_versions.append(parse_version(version))
        except VersionError as e:
            raise VersionError(f"Invalid version in list: {e}")

    return sorted(parsed_versions)


def get_latest_version(versions: List[Union[str, SemanticVersion]]) -> SemanticVersion:
    """
    Get the latest (highest) version from a list.

    Args:
        versions: List of version strings or SemanticVersion objects

    Returns:
        Latest SemanticVersion object

    Raises:
        VersionError: If no valid versions found
    """
    if not versions:
        raise VersionError("Cannot get latest version from empty list")

    sorted_versions = sort_versions(versions)
    return sorted_versions[-1]


def get_stable_versions(versions: List[Union[str, SemanticVersion]]) -> List[SemanticVersion]:
    """
    Filter and return only stable versions (no prerelease).

    Args:
        versions: List of version strings or SemanticVersion objects

    Returns:
        List of stable SemanticVersion objects
    """
    parsed_versions = []

    for version in versions:
        try:
            parsed_version = parse_version(version)
            if parsed_version.is_stable():
                parsed_versions.append(parsed_version)
        except VersionError:
            continue  # Skip invalid versions

    return sorted(parsed_versions)
    
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


