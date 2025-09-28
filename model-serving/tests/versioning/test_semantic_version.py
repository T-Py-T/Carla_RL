"""
Unit tests for semantic version parsing and validation.
"""

import pytest

from model_serving.src.versioning.semantic_version import (
    SemanticVersion,
    parse_version,
    sort_versions,
    validate_version_format,
)


class TestSemanticVersion:
    """Test cases for SemanticVersion class."""

    def test_basic_version_parsing(self):
        """Test parsing basic semantic versions."""
        version = SemanticVersion.parse("v1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
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