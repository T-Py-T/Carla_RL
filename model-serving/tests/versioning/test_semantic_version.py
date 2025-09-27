"""
Unit tests for semantic versioning functionality.
"""

import pytest

from src.versioning.semantic_version import (
    SemanticVersion,
    VersionError,
    parse_version,
    validate_version,
    sort_versions,
    get_latest_version,
    get_stable_versions,
)


class TestSemanticVersion:
    """Test SemanticVersion class."""

    def test_version_creation(self):
        """Test creating semantic versions."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build_metadata is None

    def test_version_with_prerelease(self):
        """Test creating version with prerelease."""
        version = SemanticVersion(major=1, minor=2, patch=3, prerelease="alpha.1")
        assert version.prerelease == "alpha.1"
        assert version.build_metadata is None

    def test_version_with_build_metadata(self):
        """Test creating version with build metadata."""
        version = SemanticVersion(
            major=1, minor=2, patch=3, prerelease="alpha.1", build_metadata="build.123"
        )
        assert version.prerelease == "alpha.1"
        assert version.build_metadata == "build.123"

    def test_version_validation_negative_numbers(self):
        """Test that negative version numbers raise errors."""
        with pytest.raises(VersionError):
            SemanticVersion(major=-1, minor=0, patch=0)

        with pytest.raises(VersionError):
            SemanticVersion(major=1, minor=-1, patch=0)

        with pytest.raises(VersionError):
            SemanticVersion(major=1, minor=0, patch=-1)

    def test_version_string_representation(self):
        """Test string representation of versions."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        assert str(version) == "v1.2.3"

        version_with_prerelease = SemanticVersion(major=1, minor=2, patch=3, prerelease="alpha.1")
        assert str(version_with_prerelease) == "v1.2.3-alpha.1"

        version_with_build = SemanticVersion(
            major=1, minor=2, patch=3, prerelease="alpha.1", build_metadata="build.123"
        )
        assert str(version_with_build) == "v1.2.3-alpha.1+build.123"

    def test_version_equality(self):
        """Test version equality comparison."""
        version1 = SemanticVersion(major=1, minor=2, patch=3)
        version2 = SemanticVersion(major=1, minor=2, patch=3)
        version3 = SemanticVersion(major=1, minor=2, patch=4)

        assert version1 == version2
        assert version1 != version3

    def test_version_comparison(self):
        """Test version comparison operators."""
        v1_0_0 = SemanticVersion(major=1, minor=0, patch=0)
        v1_1_0 = SemanticVersion(major=1, minor=1, patch=0)
        v2_0_0 = SemanticVersion(major=2, minor=0, patch=0)

        assert v1_0_0 < v1_1_0
        assert v1_1_0 < v2_0_0
        assert v1_0_0 <= v1_1_0
        assert v1_1_0 > v1_0_0
        assert v2_0_0 > v1_1_0
        assert v1_1_0 >= v1_0_0

    def test_prerelease_comparison(self):
        """Test prerelease version comparison."""
        stable = SemanticVersion(major=1, minor=0, patch=0)
        prerelease = SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.1")

        assert prerelease < stable
        assert stable > prerelease

    def test_prerelease_identifier_comparison(self):
        """Test prerelease identifier comparison."""
        alpha1 = SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.1")
        alpha2 = SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.2")
        beta1 = SemanticVersion(major=1, minor=0, patch=0, prerelease="beta.1")

        assert alpha1 < alpha2
        assert alpha2 < beta1

    def test_version_methods(self):
        """Test version utility methods."""
        version = SemanticVersion(major=1, minor=2, patch=3, prerelease="alpha.1")

        assert not version.is_stable()
        assert version.is_prerelease()
        assert version.get_core_version() == (1, 2, 3)

        stable_version = SemanticVersion(major=1, minor=2, patch=3)
        assert stable_version.is_stable()
        assert not stable_version.is_prerelease()

    def test_next_version_methods(self):
        """Test next version generation methods."""
        version = SemanticVersion(major=1, minor=2, patch=3)

        next_major = version.get_next_major()
        assert next_major == SemanticVersion(major=2, minor=0, patch=0)

        next_minor = version.get_next_minor()
        assert next_minor == SemanticVersion(major=1, minor=3, patch=0)

        next_patch = version.get_next_patch()
        assert next_patch == SemanticVersion(major=1, minor=2, patch=4)

    def test_version_constraints(self):
        """Test version constraint satisfaction."""
        version = SemanticVersion(major=1, minor=2, patch=3)

        assert version.satisfies(">=1.0.0")
        assert version.satisfies("<=2.0.0")
        assert version.satisfies(">1.1.0")
        assert version.satisfies("<1.3.0")
        assert version.satisfies("~1.2.0")
        assert version.satisfies("^1.0.0")
        assert version.satisfies("1.2.3")

        assert not version.satisfies(">=2.0.0")
        assert not version.satisfies("<=1.1.0")
        assert not version.satisfies("1.2.4")


class TestParseVersion:
    """Test version parsing functionality."""

    def test_parse_basic_version(self):
        """Test parsing basic version strings."""
        version = parse_version("v1.2.3")
        assert version == SemanticVersion(major=1, minor=2, patch=3)

        version = parse_version("1.2.3")
        assert version == SemanticVersion(major=1, minor=2, patch=3)

    def test_parse_version_with_prerelease(self):
        """Test parsing version with prerelease."""
        version = parse_version("v1.2.3-alpha.1")
        assert version == SemanticVersion(major=1, minor=2, patch=3, prerelease="alpha.1")

    def test_parse_version_with_build_metadata(self):
        """Test parsing version with build metadata."""
        version = parse_version("v1.2.3+build.123")
        assert version == SemanticVersion(major=1, minor=2, patch=3, build_metadata="build.123")

    def test_parse_version_with_prerelease_and_build(self):
        """Test parsing version with both prerelease and build metadata."""
        version = parse_version("v1.2.3-alpha.1+build.123")
        assert version == SemanticVersion(
            major=1, minor=2, patch=3, prerelease="alpha.1", build_metadata="build.123"
        )

    def test_parse_invalid_versions(self):
        """Test parsing invalid version strings."""
        invalid_versions = [
            "",
            "invalid",
            "1.2",
            "1.2.3.4",
            "v1.2.3-",
            "v1.2.3+",
            "v1.2.3-+",
            "v1.2.3--alpha",
            "v1.2.3++build",
        ]

        for invalid_version in invalid_versions:
            with pytest.raises(VersionError):
                parse_version(invalid_version)

    def test_parse_version_already_semantic_version(self):
        """Test parsing a SemanticVersion object."""
        original = SemanticVersion(major=1, minor=2, patch=3)
        parsed = parse_version(original)
        assert parsed == original
        # parse_version returns the same object if it's already a SemanticVersion
        assert parsed is original


class TestValidateVersion:
    """Test version validation functionality."""

    def test_validate_valid_versions(self):
        """Test validating valid version strings."""
        valid_versions = [
            "v1.0.0",
            "1.2.3",
            "v1.2.3-alpha.1",
            "v1.2.3+build.123",
            "v1.2.3-alpha.1+build.123",
        ]

        for version in valid_versions:
            assert validate_version(version)

    def test_validate_invalid_versions(self):
        """Test validating invalid version strings."""
        invalid_versions = [
            "",
            "invalid",
            "1.2",
            "1.2.3.4",
            "v1.2.3-",
            "v1.2.3+",
        ]

        for version in invalid_versions:
            assert not validate_version(version)


class TestSortVersions:
    """Test version sorting functionality."""

    def test_sort_versions(self):
        """Test sorting a list of versions."""
        versions = ["v2.0.0", "v1.0.0", "v1.1.0", "v1.0.1"]
        sorted_versions = sort_versions(versions)

        expected = [
            SemanticVersion(major=1, minor=0, patch=0),
            SemanticVersion(major=1, minor=0, patch=1),
            SemanticVersion(major=1, minor=1, patch=0),
            SemanticVersion(major=2, minor=0, patch=0),
        ]

        assert sorted_versions == expected

    def test_sort_versions_with_prerelease(self):
        """Test sorting versions with prerelease."""
        versions = ["v1.0.0", "v1.0.0-alpha.1", "v1.0.0-beta.1"]
        sorted_versions = sort_versions(versions)

        expected = [
            SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.1"),
            SemanticVersion(major=1, minor=0, patch=0, prerelease="beta.1"),
            SemanticVersion(major=1, minor=0, patch=0),
        ]

        assert sorted_versions == expected


class TestGetLatestVersion:
    """Test getting latest version functionality."""

    def test_get_latest_version(self):
        """Test getting the latest version from a list."""
        versions = ["v1.0.0", "v2.0.0", "v1.5.0"]
        latest = get_latest_version(versions)

        assert latest == SemanticVersion(major=2, minor=0, patch=0)

    def test_get_latest_version_empty_list(self):
        """Test getting latest version from empty list."""
        with pytest.raises(VersionError):
            get_latest_version([])


class TestGetStableVersions:
    """Test getting stable versions functionality."""

    def test_get_stable_versions(self):
        """Test filtering stable versions."""
        versions = [
            "v1.0.0",
            "v1.0.0-alpha.1",
            "v1.1.0",
            "v1.1.0-beta.1",
            "v2.0.0",
        ]
        stable_versions = get_stable_versions(versions)

        expected = [
            SemanticVersion(major=1, minor=0, patch=0),
            SemanticVersion(major=1, minor=1, patch=0),
            SemanticVersion(major=2, minor=0, patch=0),
        ]

        assert stable_versions == expected
