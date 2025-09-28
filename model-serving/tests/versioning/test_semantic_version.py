"""
<<<<<<< HEAD
Unit tests for semantic versioning functionality.
=======
Unit tests for semantic version parsing and validation.
>>>>>>> origin/dev
"""

import pytest

<<<<<<< HEAD
from src.versioning.semantic_version import (
    SemanticVersion,
    VersionError,
    parse_version,
    validate_version,
    sort_versions,
    get_latest_version,
    get_stable_versions,
=======
from model_serving.src.versioning.semantic_version import (
    SemanticVersion,
    parse_version,
    sort_versions,
    validate_version_format,
>>>>>>> origin/dev
)


class TestSemanticVersion:
<<<<<<< HEAD
    """Test SemanticVersion class."""

    def test_version_creation(self):
        """Test creating semantic versions."""
        version = SemanticVersion(major=1, minor=2, patch=3)
=======
    """Test cases for SemanticVersion class."""

    def test_basic_version_parsing(self):
        """Test parsing basic semantic versions."""
        version = SemanticVersion.parse("v1.2.3")
>>>>>>> origin/dev
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
<<<<<<< HEAD
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
=======
        assert version.build is None

    def test_version_without_v_prefix(self):
        """Test parsing versions without 'v' prefix."""
        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_prerelease_version(self):
        """Test parsing prerelease versions."""
        version = SemanticVersion.parse("v1.2.3-alpha")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease == "alpha"
        assert version.build is None

    def test_version_with_build_metadata(self):
        """Test parsing versions with build metadata."""
        version = SemanticVersion.parse("v1.2.3+build.123")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build == "build.123"

    def test_complete_version(self):
        """Test parsing version with prerelease and build metadata."""
        version = SemanticVersion.parse("v1.2.3-beta.1+build.456")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease == "beta.1"
        assert version.build == "build.456"

    def test_invalid_version_formats(self):
        """Test that invalid version formats raise ValueError."""
        invalid_versions = [
            "",
            "v1",
            "v1.2",
            "1.2.3.4",
            "v1.2.3-",
            "v1.2.3+",
            "not.a.version",
            "v-1.2.3",
            "v1.-2.3",
            "v1.2.-3",
        ]
        
        for invalid_version in invalid_versions:
            with pytest.raises(ValueError):
                SemanticVersion.parse(invalid_version)

    def test_version_comparison(self):
        """Test version comparison operators."""
        v1 = SemanticVersion.parse("v1.0.0")
        v2 = SemanticVersion.parse("v1.0.1")
        v3 = SemanticVersion.parse("v1.1.0")
        v4 = SemanticVersion.parse("v2.0.0")

        assert v1 < v2 < v3 < v4
        assert v4 > v3 > v2 > v1
        assert v1 == SemanticVersion.parse("v1.0.0")

    def test_prerelease_comparison(self):
        """Test comparison with prerelease versions."""
        stable = SemanticVersion.parse("v1.0.0")
        alpha = SemanticVersion.parse("v1.0.0-alpha")
        beta = SemanticVersion.parse("v1.0.0-beta")
        
        assert alpha < beta < stable
        assert alpha < stable
        assert beta < stable

    def test_version_string_representation(self):
        """Test string representation of versions."""
        version = SemanticVersion.parse("v1.2.3-alpha+build")
        assert str(version) == "v1.2.3-alpha+build"
        assert version.to_version_string(include_v_prefix=False) == "1.2.3-alpha+build"

    def test_version_bumping(self):
        """Test version bumping methods."""
        version = SemanticVersion.parse("v1.2.3")
        
        major_bump = version.bump_major()
        assert str(major_bump) == "v2.0.0"
        
        minor_bump = version.bump_minor()
        assert str(minor_bump) == "v1.3.0"
        
        patch_bump = version.bump_patch()
        assert str(patch_bump) == "v1.2.4"

    def test_compatibility_check(self):
        """Test version compatibility checking."""
        v1_0_0 = SemanticVersion.parse("v1.0.0")
        v1_1_0 = SemanticVersion.parse("v1.1.0")
        v1_2_3 = SemanticVersion.parse("v1.2.3")
        v2_0_0 = SemanticVersion.parse("v2.0.0")
        v0_1_0 = SemanticVersion.parse("v0.1.0")
        v0_2_0 = SemanticVersion.parse("v0.2.0")

        # Same major version should be compatible
        assert v1_2_3.is_compatible_with(v1_0_0)
        assert v1_2_3.is_compatible_with(v1_1_0)
        
        # Different major versions should not be compatible
        assert not v2_0_0.is_compatible_with(v1_0_0)
        assert not v1_0_0.is_compatible_with(v2_0_0)
        
        # Version 0.x.x has no compatibility guarantees
        assert not v0_2_0.is_compatible_with(v0_1_0)
        assert v0_1_0.is_compatible_with(v0_1_0)  # Only exact match

    def test_is_prerelease_and_stable(self):
        """Test prerelease and stable checks."""
        stable = SemanticVersion.parse("v1.0.0")
        prerelease = SemanticVersion.parse("v1.0.0-alpha")
        
        assert stable.is_stable()
        assert not stable.is_prerelease()
        
        assert not prerelease.is_stable()
        assert prerelease.is_prerelease()

    def test_validation_errors(self):
        """Test validation errors in constructor."""
        with pytest.raises(ValueError):
            SemanticVersion(-1, 0, 0)
        
        with pytest.raises(ValueError):
            SemanticVersion(0, -1, 0)
        
        with pytest.raises(ValueError):
            SemanticVersion(0, 0, -1)
        
        with pytest.raises(ValueError):
            SemanticVersion(1, 0, 0, prerelease="")
        
        with pytest.raises(ValueError):
            SemanticVersion(1, 0, 0, build="")


class TestVersionUtilities:
    """Test cases for version utility functions."""

    def test_parse_version_convenience_function(self):
        """Test parse_version convenience function."""
        version = parse_version("v1.2.3")
        assert isinstance(version, SemanticVersion)
        assert str(version) == "v1.2.3"

    def test_validate_version_format(self):
        """Test version format validation."""
        assert validate_version_format("v1.2.3")
        assert validate_version_format("1.0.0-alpha")
        assert validate_version_format("2.1.0+build.123")
        
        assert not validate_version_format("invalid")
        assert not validate_version_format("v1.2")
        assert not validate_version_format("")

    def test_sort_versions(self):
        """Test version sorting functionality."""
        versions = ["v1.0.0", "v2.1.0", "v1.2.3", "v1.0.1", "v2.0.0-alpha"]
        
        # Test descending sort (default)
        sorted_desc = sort_versions(versions)
        expected_desc = ["v2.1.0", "v2.0.0-alpha", "v1.2.3", "v1.0.1", "v1.0.0"]
        assert sorted_desc == expected_desc
        
        # Test ascending sort
        sorted_asc = sort_versions(versions, descending=False)
        expected_asc = ["v1.0.0", "v1.0.1", "v1.2.3", "v2.0.0-alpha", "v2.1.0"]
        assert sorted_asc == expected_asc

    def test_sort_versions_with_invalid_version(self):
        """Test that sort_versions raises error for invalid versions."""
        versions = ["v1.0.0", "invalid", "v1.2.3"]
        
        with pytest.raises(ValueError):
            sort_versions(versions)


class TestSemanticVersionEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_zero_versions(self):
        """Test versions with zero components."""
        version = SemanticVersion.parse("v0.0.0")
        assert version.major == 0
        assert version.minor == 0
        assert version.patch == 0

    def test_large_version_numbers(self):
        """Test versions with large numbers."""
        version = SemanticVersion.parse("v999.888.777")
        assert version.major == 999
        assert version.minor == 888
        assert version.patch == 777

    def test_complex_prerelease_identifiers(self):
        """Test complex prerelease identifiers."""
        versions = [
            "v1.0.0-alpha",
            "v1.0.0-alpha.1",
            "v1.0.0-alpha.beta",
            "v1.0.0-alpha.1.2",
            "v1.0.0-alpha0.valid",
            "v1.0.0-alpha-a.b-c-somethinglong",
        ]
        
        for version_str in versions:
            version = SemanticVersion.parse(version_str)
            assert version.is_prerelease()

    def test_complex_build_metadata(self):
        """Test complex build metadata."""
        versions = [
            "v1.0.0+20130313144700",
            "v1.0.0+exp.sha.5114f85",
            "v1.0.0+21AF26D3----117B344092BD",
            "v1.0.0-beta+exp.sha.5114f85",
        ]
        
        for version_str in versions:
            version = SemanticVersion.parse(version_str)
            assert version.build is not None

    def test_immutability(self):
        """Test that SemanticVersion objects are immutable."""
        version = SemanticVersion.parse("v1.0.0")
        
        # These should work (returning new objects)
        new_version = version.bump_major()
        assert version != new_version
        assert str(version) == "v1.0.0"  # Original unchanged

    def test_hashability(self):
        """Test that SemanticVersion objects are hashable."""
        v1 = SemanticVersion.parse("v1.0.0")
        v2 = SemanticVersion.parse("v1.0.0")
        v3 = SemanticVersion.parse("v1.0.1")
        
        # Can be used as dict keys
        version_dict = {v1: "first", v2: "second", v3: "third"}
        assert len(version_dict) == 2  # v1 and v2 are equal
        
        # Can be used in sets
        version_set = {v1, v2, v3}
        assert len(version_set) == 2  # v1 and v2 are equal
>>>>>>> origin/dev
