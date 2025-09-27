"""
Unit tests for version selector and multi-version model support.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest

from model_serving.src.versioning.semantic_version import SemanticVersion
from model_serving.src.versioning.version_selector import (
    VersionSelector,
    VersionSelectionStrategy,
    get_version_from_environment,
    select_best_version,
)
from model_serving.src.exceptions import ModelLoadingError


class TestVersionSelector:
    """Test cases for VersionSelector class."""

    def setup_method(self):
        """Set up test environment with temporary artifacts directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifacts_root = Path(self.temp_dir) / "artifacts"
        self.artifacts_root.mkdir(exist_ok=True)
        
        # Create test version directories and model cards
        self.test_versions = {
            "v1.0.0": {
                "model_name": "test-model",
                "version": "v1.0.0",
                "model_type": "pytorch",
                "performance_metrics": {
                    "latency_p50_ms": 8.5,
                    "throughput_rps": 120,
                    "memory_usage_mb": 256
                }
            },
            "v1.1.0": {
                "model_name": "test-model",
                "version": "v1.1.0",
                "model_type": "pytorch",
                "performance_metrics": {
                    "latency_p50_ms": 7.2,
                    "throughput_rps": 150,
                    "memory_usage_mb": 280
                }
            },
            "v1.2.0-beta": {
                "model_name": "test-model",
                "version": "v1.2.0-beta",
                "model_type": "pytorch",
                "performance_metrics": {
                    "latency_p50_ms": 6.8,
                    "throughput_rps": 180,
                    "memory_usage_mb": 290
                }
            },
            "v2.0.0": {
                "model_name": "test-model",
                "version": "v2.0.0",
                "model_type": "pytorch",
                "performance_metrics": {
                    "latency_p50_ms": 9.1,
                    "throughput_rps": 110,
                    "memory_usage_mb": 320
                }
            }
        }
        
        # Create version directories and model cards
        for version_str, model_card in self.test_versions.items():
            version_dir = self.artifacts_root / version_str
            version_dir.mkdir()
            
            # Write model card
            model_card_path = version_dir / "model_card.yaml"
            with open(model_card_path, 'w') as f:
                yaml.dump(model_card, f)
            
            # Create dummy model file
            model_path = version_dir / "model.pt"
            model_path.touch()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_discover_versions(self):
        """Test version discovery functionality."""
        selector = VersionSelector(self.artifacts_root)
        versions = selector.discover_versions()
        
        # Should return all versions sorted in descending order
        expected_versions = [
            SemanticVersion.parse("v2.0.0"),
            SemanticVersion.parse("v1.2.0-beta"),
            SemanticVersion.parse("v1.1.0"),
            SemanticVersion.parse("v1.0.0"),
        ]
        
        assert versions == expected_versions

    def test_discover_versions_with_cache(self):
        """Test that version discovery uses caching."""
        selector = VersionSelector(self.artifacts_root)
        
        # First call should scan filesystem
        versions1 = selector.discover_versions()
        
        # Second call should use cache
        versions2 = selector.discover_versions()
        
        assert versions1 == versions2
        assert selector._version_cache is not None

    def test_discover_versions_force_rescan(self):
        """Test forced rescanning of versions."""
        selector = VersionSelector(self.artifacts_root)
        
        # Initial scan
        versions1 = selector.discover_versions()
        
        # Add new version
        new_version_dir = self.artifacts_root / "v1.3.0"
        new_version_dir.mkdir()
        model_card = {
            "model_name": "test-model",
            "version": "v1.3.0",
            "model_type": "pytorch"
        }
        with open(new_version_dir / "model_card.yaml", 'w') as f:
            yaml.dump(model_card, f)
        
        # Scan without force should return cached results
        versions2 = selector.discover_versions(force_rescan=False)
        assert len(versions2) == len(versions1)
        
        # Scan with force should find new version
        versions3 = selector.discover_versions(force_rescan=True)
        assert len(versions3) == len(versions1) + 1

    def test_select_latest_stable_version(self):
        """Test selection of latest stable version."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(VersionSelectionStrategy.LATEST_STABLE)
        
        # Should select v2.0.0 (latest stable, excluding beta)
        assert str(version) == "v2.0.0"

    def test_select_latest_version(self):
        """Test selection of latest version including prereleases."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.LATEST,
            exclude_prereleases=False
        )
        
        # Should select v2.0.0 (latest overall)
        assert str(version) == "v2.0.0"

    def test_select_exact_version(self):
        """Test selection of exact version."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.EXACT,
            exact_version="v1.1.0"
        )
        
        assert str(version) == "v1.1.0"

    def test_select_exact_version_not_found(self):
        """Test exact version selection when version doesn't exist."""
        selector = VersionSelector(self.artifacts_root)
        
        with pytest.raises(ModelLoadingError, match="Exact version v3.0.0 not found"):
            selector.select_version(
                VersionSelectionStrategy.EXACT,
                exact_version="v3.0.0"
            )

    def test_select_exact_version_missing_parameter(self):
        """Test exact version selection without specifying version."""
        selector = VersionSelector(self.artifacts_root)
        
        with pytest.raises(ModelLoadingError, match="Exact version must be specified"):
            selector.select_version(VersionSelectionStrategy.EXACT)

    def test_select_compatible_version(self):
        """Test selection of compatible version."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.COMPATIBLE,
            minimum_version="v1.0.0"
        )
        
        # Should select v1.1.0 (latest compatible with v1.0.0)
        assert str(version) == "v1.1.0"

    def test_select_performance_optimized_version(self):
        """Test selection of performance-optimized version."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.PERFORMANCE_OPTIMIZED,
            exclude_prereleases=False
        )
        
        # Should select version with best performance score
        # v1.2.0-beta has best latency and throughput
        assert str(version) == "v1.2.0-beta"

    def test_select_performance_optimized_with_threshold(self):
        """Test performance optimization with threshold requirements."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.PERFORMANCE_OPTIMIZED,
            performance_threshold={"latency_p50_ms": 8.0},
            exclude_prereleases=False
        )
        
        # Should select v1.2.0-beta (latency 6.8ms < 8.0ms threshold)
        assert str(version) == "v1.2.0-beta"

    def test_select_performance_optimized_no_versions_meet_threshold(self):
        """Test performance optimization when no versions meet threshold."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.PERFORMANCE_OPTIMIZED,
            performance_threshold={"latency_p50_ms": 5.0},  # Too strict
            exclude_prereleases=False
        )
        
        # Should return None when no versions meet threshold
        assert version is None

    def test_select_version_with_minimum_version_filter(self):
        """Test version selection with minimum version filtering."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.LATEST_STABLE,
            minimum_version="v1.1.0"
        )
        
        # Should select v2.0.0 (latest stable above v1.1.0)
        assert str(version) == "v2.0.0"

    def test_select_version_exclude_prereleases(self):
        """Test version selection excluding prereleases."""
        selector = VersionSelector(self.artifacts_root)
        version = selector.select_version(
            VersionSelectionStrategy.LATEST,
            exclude_prereleases=True
        )
        
        # Should select v2.0.0 (latest excluding beta)
        assert str(version) == "v2.0.0"

    def test_select_version_no_suitable_versions(self):
        """Test version selection when no versions meet criteria."""
        selector = VersionSelector(self.artifacts_root)
        
        with pytest.raises(ModelLoadingError, match="No versions match the specified criteria"):
            selector.select_version(
                VersionSelectionStrategy.LATEST_STABLE,
                minimum_version="v3.0.0"  # Higher than any available version
            )

    def test_get_version_info(self):
        """Test getting detailed version information."""
        selector = VersionSelector(self.artifacts_root)
        version = SemanticVersion.parse("v1.1.0")
        info = selector.get_version_info(version)
        
        assert info['version'] == "v1.1.0"
        assert info['is_stable'] is True
        assert info['is_prerelease'] is False
        assert 'model_card' in info
        assert 'performance_metrics' in info
        assert info['performance_metrics']['latency_p50_ms'] == 7.2

    def test_get_version_info_nonexistent_version(self):
        """Test getting version info for nonexistent version."""
        selector = VersionSelector(self.artifacts_root)
        version = SemanticVersion.parse("v3.0.0")
        
        with pytest.raises(ModelLoadingError, match="Version directory not found"):
            selector.get_version_info(version)

    def test_validate_version_availability_valid(self):
        """Test validation of available version."""
        selector = VersionSelector(self.artifacts_root)
        is_available, error_msg = selector.validate_version_availability("v1.0.0")
        
        assert is_available is True
        assert error_msg is None

    def test_validate_version_availability_invalid_format(self):
        """Test validation of invalid version format."""
        selector = VersionSelector(self.artifacts_root)
        is_available, error_msg = selector.validate_version_availability("invalid")
        
        assert is_available is False
        assert "Invalid version format" in error_msg

    def test_validate_version_availability_not_found(self):
        """Test validation of nonexistent version."""
        selector = VersionSelector(self.artifacts_root)
        is_available, error_msg = selector.validate_version_availability("v3.0.0")
        
        assert is_available is False
        assert "Version directory not found" in error_msg

    def test_nonexistent_artifacts_directory(self):
        """Test behavior with nonexistent artifacts directory."""
        nonexistent_path = Path("/nonexistent/path")
        selector = VersionSelector(nonexistent_path)
        
        with pytest.raises(ModelLoadingError, match="Artifacts directory not found"):
            selector.discover_versions()


class TestVersionSelectorUtilities:
    """Test utility functions for version selection."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifacts_root = Path(self.temp_dir) / "artifacts"
        self.artifacts_root.mkdir(exist_ok=True)
        
        # Create a simple test version
        version_dir = self.artifacts_root / "v1.0.0"
        version_dir.mkdir()
        
        model_card = {
            "model_name": "test-model",
            "version": "v1.0.0",
            "model_type": "pytorch"
        }
        
        with open(version_dir / "model_card.yaml", 'w') as f:
            yaml.dump(model_card, f)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_select_best_version_convenience_function(self):
        """Test select_best_version convenience function."""
        version = select_best_version(
            self.artifacts_root,
            VersionSelectionStrategy.LATEST_STABLE
        )
        
        assert str(version) == "v1.0.0"

    @patch.dict(os.environ, {"MODEL_VERSION": "v1.0.0"})
    def test_get_version_from_environment_valid(self):
        """Test getting version from environment variable."""
        version = get_version_from_environment(
            self.artifacts_root,
            env_var="MODEL_VERSION"
        )
        
        assert str(version) == "v1.0.0"

    @patch.dict(os.environ, {"MODEL_VERSION": "v2.0.0"})
    def test_get_version_from_environment_invalid(self):
        """Test fallback when environment version is invalid."""
        with patch('builtins.print') as mock_print:
            version = get_version_from_environment(
                self.artifacts_root,
                env_var="MODEL_VERSION",
                fallback_strategy=VersionSelectionStrategy.LATEST_STABLE
            )
            
            # Should fall back to available version
            assert str(version) == "v1.0.0"
            
            # Should print warning
            mock_print.assert_called_once()
            assert "Warning: Environment version v2.0.0 not available" in str(mock_print.call_args)

    def test_get_version_from_environment_no_env_var(self):
        """Test fallback when environment variable is not set."""
        version = get_version_from_environment(
            self.artifacts_root,
            env_var="NONEXISTENT_VAR",
            fallback_strategy=VersionSelectionStrategy.LATEST_STABLE
        )
        
        assert str(version) == "v1.0.0"


class TestVersionSelectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_artifacts_directory(self):
        """Test behavior with empty artifacts directory."""
        temp_dir = tempfile.mkdtemp()
        artifacts_root = Path(temp_dir) / "artifacts"
        artifacts_root.mkdir()
        
        try:
            selector = VersionSelector(artifacts_root)
            versions = selector.discover_versions()
            assert versions == []
            
            version = selector.select_version(VersionSelectionStrategy.LATEST_STABLE)
            assert version is None
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_invalid_model_card_files(self):
        """Test handling of invalid model card files."""
        temp_dir = tempfile.mkdtemp()
        artifacts_root = Path(temp_dir) / "artifacts"
        artifacts_root.mkdir()
        
        try:
            # Create version directory with invalid model card
            version_dir = artifacts_root / "v1.0.0"
            version_dir.mkdir()
            
            # Write invalid YAML
            model_card_path = version_dir / "model_card.yaml"
            with open(model_card_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            selector = VersionSelector(artifacts_root)
            versions = selector.discover_versions()
            
            # Should still discover the version
            assert len(versions) == 1
            assert str(versions[0]) == "v1.0.0"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_directories_without_valid_version_names(self):
        """Test ignoring directories that don't follow semantic versioning."""
        temp_dir = tempfile.mkdtemp()
        artifacts_root = Path(temp_dir) / "artifacts"
        artifacts_root.mkdir()
        
        try:
            # Create directories with invalid version names
            (artifacts_root / "not-a-version").mkdir()
            (artifacts_root / "v1").mkdir()  # Incomplete version
            (artifacts_root / "random-directory").mkdir()
            
            # Create valid version directory
            version_dir = artifacts_root / "v1.0.0"
            version_dir.mkdir()
            model_card = {
                "model_name": "test-model",
                "version": "v1.0.0",
                "model_type": "pytorch"
            }
            with open(version_dir / "model_card.yaml", 'w') as f:
                yaml.dump(model_card, f)
            
            selector = VersionSelector(artifacts_root)
            versions = selector.discover_versions()
            
            # Should only discover the valid version
            assert len(versions) == 1
            assert str(versions[0]) == "v1.0.0"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)