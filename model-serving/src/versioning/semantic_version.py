"""
Semantic versioning parser and validation utilities.

This module provides semantic version parsing, validation, and comparison
functionality following the vMAJOR.MINOR.PATCH[-prerelease][+build] format
defined by https://semver.org/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering
from typing import List, Optional, Tuple, Union


class VersionError(ValueError):
    """Raised on version parsing and validation errors.

    Subclassing `ValueError` keeps callers that catch `ValueError` working,
    which matches the historical behaviour of `SemanticVersion.parse`.
    """

    def __init__(self, message: str, version_string: Optional[str] = None):
        super().__init__(message)
        self.version_string = version_string


# Strict semver pattern. Leading "v" is optional; prerelease identifiers are
# dot-separated alphanumerics; build metadata is dot-separated alphanumerics
# including hyphens.
VERSION_PATTERN = re.compile(
    r"^v?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


@total_ordering
@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version with parsing and full ordering support.

    `@total_ordering` fills in the remaining comparison operators from
    `__eq__` and `__lt__`. The dataclass is frozen so instances are hashable
    and safe to use in sets/dicts.
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __post_init__(self) -> None:
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise VersionError(
                f"Version numbers must be non-negative, got "
                f"{self.major}.{self.minor}.{self.patch}"
            )
        if self.prerelease is not None and not self.prerelease:
            raise VersionError("Prerelease identifier cannot be empty")
        if self.build is not None and not self.build:
            raise VersionError("Build metadata cannot be empty")

    @classmethod
    def parse(cls, version_str: Union[str, "SemanticVersion"]) -> "SemanticVersion":
        """Parse a version string into a SemanticVersion object."""
        if isinstance(version_str, SemanticVersion):
            return version_str
        if not isinstance(version_str, str):
            raise VersionError(
                f"Version must be a string, got {type(version_str).__name__}"
            )
        match = VERSION_PATTERN.match(version_str.strip())
        if not match:
            raise VersionError(
                f"Invalid semantic version format: '{version_str}'. "
                f"Expected vMAJOR.MINOR.PATCH[-prerelease][+build]",
                version_str,
            )
        groups = match.groupdict()
        return cls(
            major=int(groups["major"]),
            minor=int(groups["minor"]),
            patch=int(groups["patch"]),
            prerelease=groups.get("prerelease"),
            build=groups.get("build"),
        )

    def __str__(self) -> str:
        version = f"v{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __repr__(self) -> str:
        return (
            f"SemanticVersion(major={self.major}, minor={self.minor}, "
            f"patch={self.patch}, prerelease={self.prerelease!r}, "
            f"build={self.build!r})"
        )

    # Equality ignores build metadata per semver spec.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other: "SemanticVersion") -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        if self.get_core_version() != other.get_core_version():
            return self.get_core_version() < other.get_core_version()
        # Stable version ranks higher than prerelease version with same core.
        if self.prerelease is None and other.prerelease is not None:
            return False
        if self.prerelease is not None and other.prerelease is None:
            return True
        if self.prerelease is not None and other.prerelease is not None:
            return self._compare_prerelease(self.prerelease, other.prerelease)
        return False

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))

    @staticmethod
    def _compare_prerelease(a: str, b: str) -> bool:
        """Return True when prerelease `a` sorts strictly before `b`."""
        ids_a = a.split(".")
        ids_b = b.split(".")
        for i in range(max(len(ids_a), len(ids_b))):
            id_a = ids_a[i] if i < len(ids_a) else None
            id_b = ids_b[i] if i < len(ids_b) else None
            # A shorter identifier list has lower precedence per semver spec.
            if id_a is None:
                return True
            if id_b is None:
                return False
            if id_a == id_b:
                continue
            try:
                return int(id_a) < int(id_b)
            except ValueError:
                return id_a < id_b
        return False

    def is_stable(self) -> bool:
        return self.prerelease is None

    def is_prerelease(self) -> bool:
        return self.prerelease is not None

    def get_core_version(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def to_version_string(self, include_v_prefix: bool = True) -> str:
        """Return string form, optionally without the leading `v`."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return f"v{version}" if include_v_prefix else version

    def bump_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    # Kept for backwards compatibility with older callers.
    def get_next_major(self) -> "SemanticVersion":
        return self.bump_major()

    def get_next_minor(self) -> "SemanticVersion":
        return self.bump_minor()

    def get_next_patch(self) -> "SemanticVersion":
        return self.bump_patch()

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Return True if this version is backward compatible with `other`.

        For major > 0, compatibility requires same major and this >= other.
        For major == 0, any change is potentially breaking, so compatibility
        is defined as strict equality.
        """
        if not isinstance(other, SemanticVersion):
            return False
        if self.major == 0 or other.major == 0:
            return self == other
        return self.major == other.major and self >= other

    def satisfies(self, constraint: str) -> bool:
        """Check whether this version satisfies `constraint`.

        Supported operators: `>=`, `<=`, `>`, `<`, `~` (same major+minor,
        patch at or above), `^` (same major, minor+patch at or above),
        or an exact version.
        """
        constraint = constraint.strip()
        if constraint.startswith(">="):
            return self >= parse_version(constraint[2:])
        if constraint.startswith("<="):
            return self <= parse_version(constraint[2:])
        if constraint.startswith(">"):
            return self > parse_version(constraint[1:])
        if constraint.startswith("<"):
            return self < parse_version(constraint[1:])
        if constraint.startswith("~"):
            target = parse_version(constraint[1:])
            return (
                self.major == target.major
                and self.minor == target.minor
                and self.patch >= target.patch
            )
        if constraint.startswith("^"):
            target = parse_version(constraint[1:])
            return (
                self.major == target.major
                and (self.minor, self.patch) >= (target.minor, target.patch)
            )
        return self == parse_version(constraint)


def parse_version(version_string: Union[str, SemanticVersion]) -> SemanticVersion:
    """Parse a version string (module-level shorthand for SemanticVersion.parse)."""
    return SemanticVersion.parse(version_string)


def validate_version(version_string: str) -> bool:
    """Return True if `version_string` is a syntactically valid semver."""
    try:
        SemanticVersion.parse(version_string)
        return True
    except VersionError:
        return False


def validate_version_format(version_string: str) -> bool:
    """Alias for `validate_version` kept for backwards compatibility."""
    return validate_version(version_string)


def sort_versions(
    versions: List[Union[str, SemanticVersion]],
    descending: bool = True,
) -> List[SemanticVersion]:
    """Parse and return `versions` sorted in the requested direction.

    Defaults to descending order because that matches how callers typically
    want to enumerate versions ("newest first"). Pass `descending=False` to
    get ascending order instead.
    """
    parsed: List[SemanticVersion] = []
    for version in versions:
        try:
            parsed.append(SemanticVersion.parse(version))
        except VersionError as exc:
            raise VersionError(f"Invalid version in list: {exc}")
    return sorted(parsed, reverse=descending)


def get_latest_version(versions: List[Union[str, SemanticVersion]]) -> SemanticVersion:
    """Return the highest-ranking semver from `versions`."""
    if not versions:
        raise VersionError("Cannot get latest version from empty list")
    # sort_versions defaults to descending, so the newest is at index 0.
    return sort_versions(versions)[0]


def get_stable_versions(
    versions: List[Union[str, SemanticVersion]],
) -> List[SemanticVersion]:
    """Return only stable (non-prerelease) versions, sorted ascending.

    Invalid entries are silently skipped so noisy inputs don't block callers
    who only need stable releases.
    """
    stable: List[SemanticVersion] = []
    for version in versions:
        try:
            parsed = SemanticVersion.parse(version)
        except VersionError:
            continue
        if parsed.is_stable():
            stable.append(parsed)
    return sorted(stable)
