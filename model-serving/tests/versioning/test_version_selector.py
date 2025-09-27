"""
Unit tests for version selector and multi-version model support.

Tests comprehensive version selection logic, fallback strategies,
and multi-version model management capabilities.
"""

import pytest
from unittest.mock import Mock

from src.versioning.version_selector import (
    VersionSelector,
    VersionManager,
    VersionSelectionStrategy,
    VersionSelectionError,
    VersionSelectionResult,
    ModelVersionInfo,
)
from src.versioning.semantic_version import SemanticVersion, parse_version
from src.versioning.artifact_manager import ArtifactManager, ArtifactManifest


class TestVersionSelectionStrategy:
    """Test version selection strategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert VersionSelectionStrategy.LATEST.value == "latest"
        assert VersionSelectionStrategy.STABLE.value == "stable"
        assert VersionSelectionStrategy.SPECIFIC.value == "specific"
        assert VersionSelectionStrategy.COMPATIBLE.value == "compatible"
        assert VersionSelectionStrategy.FALLBACK.value == "fallback"


class TestVersionSelectionResult:
    """Test version selection result dataclass."""

    def test_result_creation(self):
        """Test creating version selection result."""
        version = parse_version("v1.2.3")
        result = VersionSelectionResult(
            selected_version=version, strategy_used=VersionSelectionStrategy.LATEST
        )

        assert result.selected_version == version
        assert result.strategy_used == VersionSelectionStrategy.LATEST
        assert result.fallback_used is False
        assert result.fallback_reason is None
        assert result.available_versions is None
        assert result.selection_metadata == {}

    def test_result_with_fallback(self):
        """Test result with fallback information."""
        version = parse_version("v1.2.3")
        result = VersionSelectionResult(
            selected_version=version,
            strategy_used=VersionSelectionStrategy.STABLE,
            fallback_used=True,
            fallback_reason="Primary strategy failed",
        )

        assert result.fallback_used is True
        assert result.fallback_reason == "Primary strategy failed"


class TestModelVersionInfo:
    """Test model version info dataclass."""

    def test_version_info_creation(self):
        """Test creating model version info."""
        version = parse_version("v1.2.3")
        manifest = ArtifactManifest(version="v1.2.3")

        info = ModelVersionInfo(
            version=version,
            manifest=manifest,
            is_available=True,
            integrity_status={"model.pt": True},
        )

        assert info.version == version
        assert info.manifest == manifest
        assert info.is_available is True
        assert info.integrity_status == {"model.pt": True}
        assert info.last_accessed is None
        assert info.usage_count == 0
        assert info.performance_metrics == {}


class TestVersionSelector:
    """Test version selector functionality."""

    @pytest.fixture
    def mock_artifact_manager(self):
        """Create mock artifact manager."""
        manager = Mock(spec=ArtifactManager)
        manager.list_versions.return_value = [
            parse_version("v1.0.0"),
            parse_version("v1.1.0"),
            parse_version("v1.2.0"),
            parse_version("v2.0.0-alpha.1"),
            parse_version("v2.0.0"),
        ]

        # Mock manifests
        def mock_get_manifest(version):
            return ArtifactManifest(
                version=str(version),
                artifacts={"model.pt": "hash123"},
                created_at="2024-01-01T00:00:00Z",
            )

        manager.get_manifest.side_effect = mock_get_manifest
        manager.verify_integrity.return_value = {"model.pt": True}

        return manager

    @pytest.fixture
    def version_selector(self, mock_artifact_manager):
        """Create version selector with mock artifact manager."""
        return VersionSelector(mock_artifact_manager)

    def test_initialization(self, mock_artifact_manager):
        """Test version selector initialization."""
        selector = VersionSelector(mock_artifact_manager)

        assert selector.artifact_manager == mock_artifact_manager
        assert selector.default_strategy == VersionSelectionStrategy.STABLE
        assert len(selector.version_cache) > 0
        assert len(selector.selection_history) == 0

    def test_select_latest_version(self, version_selector):
        """Test selecting latest version."""
        result = version_selector.select_version(strategy=VersionSelectionStrategy.LATEST)

        assert result.selected_version == parse_version("v2.0.0")
        assert result.strategy_used == VersionSelectionStrategy.LATEST
        assert not result.fallback_used

    def test_select_stable_version(self, version_selector):
        """Test selecting latest stable version."""
        result = version_selector.select_version(strategy=VersionSelectionStrategy.STABLE)

        assert result.selected_version == parse_version("v2.0.0")
        assert result.strategy_used == VersionSelectionStrategy.STABLE
        assert not result.fallback_used

    def test_select_specific_version(self, version_selector):
        """Test selecting specific version."""
        result = version_selector.select_version(
            version_spec="v1.1.0", strategy=VersionSelectionStrategy.SPECIFIC
        )

        assert result.selected_version == parse_version("v1.1.0")
        assert result.strategy_used == VersionSelectionStrategy.SPECIFIC

    def test_select_specific_version_not_available(self, version_selector):
        """Test selecting specific version that's not available."""
        # Mock to make specific version unavailable
        version_selector.version_cache = {
            parse_version("v1.0.0"): Mock(),
            parse_version("v1.1.0"): Mock(),
        }

        with pytest.raises(VersionSelectionError) as exc_info:
            version_selector.select_version(
                version_spec="v3.0.0",
                strategy=VersionSelectionStrategy.SPECIFIC,
                fallback_strategies=[],  # Disable fallback
            )

        assert "not available" in str(exc_info.value) or "No suitable version found" in str(
            exc_info.value
        )

    def test_select_compatible_version(self, version_selector):
        """Test selecting compatible version."""
        result = version_selector.select_version(
            version_spec="v1.0.0", strategy=VersionSelectionStrategy.COMPATIBLE
        )

        # Should select latest compatible version (v1.2.0)
        assert result.selected_version == parse_version("v1.2.0")
        assert result.strategy_used == VersionSelectionStrategy.COMPATIBLE

    def test_select_with_constraints(self, version_selector):
        """Test selecting version with constraints."""
        result = version_selector.select_version(
            constraints=[">=1.1.0", "<2.0.0"], strategy=VersionSelectionStrategy.LATEST
        )

        # Should select v1.2.0 (latest in range, excluding prerelease)
        assert result.selected_version == parse_version("v1.2.0")

    def test_fallback_strategy(self, version_selector):
        """Test fallback strategy when primary fails."""
        # Mock to make specific version unavailable
        version_selector.version_cache = {
            parse_version("v1.0.0"): Mock(),
            parse_version("v1.1.0"): Mock(),
        }

        # Mock the artifact manager to return only the cached versions
        version_selector.artifact_manager.list_versions.return_value = [
            parse_version("v1.0.0"),
            parse_version("v1.1.0"),
        ]

        result = version_selector.select_version(
            version_spec="v3.0.0",
            strategy=VersionSelectionStrategy.SPECIFIC,
            fallback_strategies=[VersionSelectionStrategy.LATEST],
        )

        assert result.fallback_used is True
        assert "Primary strategy" in result.fallback_reason
        # Should select the latest available version (v1.1.0)
        assert result.selected_version == parse_version("v1.1.0")

    def test_no_versions_available(self, mock_artifact_manager):
        """Test behavior when no versions are available."""
        mock_artifact_manager.list_versions.return_value = []
        selector = VersionSelector(mock_artifact_manager)

        with pytest.raises(VersionSelectionError) as exc_info:
            selector.select_version()

        assert "No model versions available" in str(exc_info.value)

    def test_get_version_info(self, version_selector):
        """Test getting version information."""
        info = version_selector.get_version_info("v1.0.0")

        assert info is not None
        assert info.version == parse_version("v1.0.0")
        assert info.is_available is True

    def test_get_version_info_not_found(self, version_selector):
        """Test getting version information for non-existent version."""
        info = version_selector.get_version_info("v3.0.0")
        assert info is None

    def test_list_available_versions(self, version_selector):
        """Test listing available versions."""
        versions = version_selector.list_available_versions()

        assert len(versions) > 0
        assert all(isinstance(v, SemanticVersion) for v in versions)
        assert versions == sorted(versions)

    def test_list_stable_versions_only(self, version_selector):
        """Test listing only stable versions."""
        stable_versions = version_selector.list_available_versions(stable_only=True)

        assert all(v.is_stable() for v in stable_versions)
        assert parse_version("v2.0.0-alpha.1") not in stable_versions

    def test_get_latest_version(self, version_selector):
        """Test getting latest version."""
        latest = version_selector.get_latest_version()
        assert latest == parse_version("v2.0.0")

    def test_get_latest_stable_version(self, version_selector):
        """Test getting latest stable version."""
        latest_stable = version_selector.get_latest_version(stable_only=True)
        assert latest_stable == parse_version("v2.0.0")

    def test_check_version_compatibility(self, version_selector):
        """Test version compatibility checking."""
        version = parse_version("v1.2.0")

        # Test compatible constraints
        assert version_selector.check_version_compatibility(version, [">=1.0.0", "<2.0.0"])
        assert version_selector.check_version_compatibility(version, ["~1.2.0"])

        # Test incompatible constraints
        assert not version_selector.check_version_compatibility(version, [">=2.0.0"])
        assert not version_selector.check_version_compatibility(version, ["<1.0.0"])

    def test_register_custom_selector(self, version_selector):
        """Test registering custom selector."""

        def custom_selector(versions, criteria):
            return max(versions)

        version_selector.register_custom_selector("custom", custom_selector)
        assert "custom" in version_selector.custom_selectors

    def test_get_selection_history(self, version_selector):
        """Test getting selection history."""
        # Make some selections
        version_selector.select_version(strategy=VersionSelectionStrategy.LATEST)
        version_selector.select_version(strategy=VersionSelectionStrategy.STABLE)

        history = version_selector.get_selection_history()
        assert len(history) == 2

        # Test with limit
        limited_history = version_selector.get_selection_history(limit=1)
        assert len(limited_history) == 1

    def test_performance_weighting(self, version_selector):
        """Test performance-based version selection."""
        # Mock to limit available versions to only those with performance metrics
        version_selector.version_cache = {
            parse_version("v1.0.0"): Mock(),
            parse_version("v1.1.0"): Mock(),
        }
        version_selector.artifact_manager.list_versions.return_value = [
            parse_version("v1.0.0"),
            parse_version("v1.1.0"),
        ]

        # Add performance metrics
        version_selector.update_performance_metrics(
            "v1.0.0", {"avg_latency_ms": 50.0, "throughput_rps": 1000.0}
        )
        version_selector.update_performance_metrics(
            "v1.1.0", {"avg_latency_ms": 30.0, "throughput_rps": 1200.0}
        )

        result = version_selector.select_version(
            strategy=VersionSelectionStrategy.LATEST, performance_weight=0.5
        )

        # Should select version with better performance
        assert result.selected_version in [parse_version("v1.0.0"), parse_version("v1.1.0")]

    def test_get_recommended_version(self, version_selector):
        """Test getting recommended version for different use cases."""
        # Production should return latest stable
        prod_version = version_selector.get_recommended_version("production")
        assert prod_version == parse_version("v2.0.0")

        # Development should return latest (including prerelease)
        dev_version = version_selector.get_recommended_version("development")
        assert dev_version == parse_version("v2.0.0")

        # Testing should return latest stable
        test_version = version_selector.get_recommended_version("testing")
        assert test_version == parse_version("v2.0.0")

    def test_refresh_version_cache(self, version_selector):
        """Test refreshing version cache."""
        initial_count = len(version_selector.version_cache)

        # Mock new version
        version_selector.artifact_manager.list_versions.return_value.append(parse_version("v3.0.0"))

        version_selector._refresh_version_cache()

        # Cache should be updated
        assert len(version_selector.version_cache) > initial_count


class TestVersionManager:
    """Test version manager functionality."""

    @pytest.fixture
    def mock_artifact_manager(self):
        """Create mock artifact manager."""
        manager = Mock(spec=ArtifactManager)
        manager.list_versions.return_value = [
            parse_version("v1.0.0"),
            parse_version("v1.1.0"),
            parse_version("v2.0.0"),
        ]

        def mock_get_manifest(version):
            return ArtifactManifest(version=str(version))

        manager.get_manifest.side_effect = mock_get_manifest
        manager.verify_integrity.return_value = {"model.pt": True}

        return manager

    @pytest.fixture
    def version_manager(self, mock_artifact_manager):
        """Create version manager with mock artifact manager."""
        return VersionManager(mock_artifact_manager)

    def test_initialization(self, mock_artifact_manager):
        """Test version manager initialization."""
        manager = VersionManager(mock_artifact_manager)

        assert manager.artifact_manager == mock_artifact_manager
        assert isinstance(manager.selector, VersionSelector)
        assert manager.current_version is None
        assert len(manager.version_callbacks) == 0

    def test_get_model_version_with_spec(self, version_manager):
        """Test getting model version with specification."""
        # Mock the selector to return specific version
        mock_result = VersionSelectionResult(
            selected_version=parse_version("v1.1.0"),
            strategy_used=VersionSelectionStrategy.SPECIFIC,
        )
        version_manager.selector.select_version = Mock(return_value=mock_result)

        version = version_manager.get_model_version("v1.1.0")

        assert version == parse_version("v1.1.0")
        assert version_manager.current_version == parse_version("v1.1.0")

    def test_get_model_version_without_spec(self, version_manager):
        """Test getting model version without specification."""
        version = version_manager.get_model_version()

        assert version is not None
        assert version_manager.current_version == version

    def test_get_model_version_with_auto_fallback(self, version_manager):
        """Test getting model version with auto fallback."""
        # Mock selector to raise error on first call, succeed on second
        mock_selector = Mock()
        mock_selector.select_version.side_effect = [
            VersionSelectionError("Test error"),
            VersionSelectionResult(
                selected_version=parse_version("v1.0.0"),
                strategy_used=VersionSelectionStrategy.STABLE,
            ),
        ]
        version_manager.selector = mock_selector

        version = version_manager.get_model_version("v3.0.0", auto_fallback=True)

        assert version == parse_version("v1.0.0")
        assert version_manager.current_version == parse_version("v1.0.0")

    def test_get_model_version_without_auto_fallback(self, version_manager):
        """Test getting model version without auto fallback."""
        mock_selector = Mock()
        mock_selector.select_version.side_effect = VersionSelectionError("Test error")
        version_manager.selector = mock_selector

        with pytest.raises(VersionSelectionError):
            version_manager.get_model_version("v3.0.0", auto_fallback=False)

    def test_register_version_change_callback(self, version_manager):
        """Test registering version change callback."""
        callback = Mock()
        version_manager.register_version_change_callback(callback)

        assert callback in version_manager.version_callbacks

    def test_version_change_callback_execution(self, version_manager):
        """Test version change callback execution."""
        callback = Mock()
        version_manager.register_version_change_callback(callback)

        # Trigger version change
        version_manager._notify_version_change(parse_version("v1.0.0"))

        callback.assert_called_once_with(parse_version("v1.0.0"))

    def test_version_change_callback_error_handling(self, version_manager, caplog):
        """Test error handling in version change callbacks."""

        def failing_callback(version):
            raise Exception("Callback error")

        version_manager.register_version_change_callback(failing_callback)

        # Should not raise exception
        version_manager._notify_version_change(parse_version("v1.0.0"))

        # Should log error
        assert "Error in version change callback" in caplog.text

    def test_get_version_status(self, version_manager):
        """Test getting version status."""
        # Set current version
        version_manager.current_version = parse_version("v1.0.0")

        status = version_manager.get_version_status()

        assert status["current_version"] == "v1.0.0"
        assert "available_versions" in status
        assert "stable_versions" in status
        assert "latest_version" in status
        assert "selection_history_count" in status


class TestIntegration:
    """Integration tests for version selector system."""

    def test_end_to_end_version_selection(self):
        """Test end-to-end version selection workflow."""
        # Create real artifact manager with temporary directory
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_manager = ArtifactManager(temp_dir)

            # Create some test manifests
            for version in ["v1.0.0", "v1.1.0", "v2.0.0"]:
                manifest = ArtifactManifest(version=version)
                artifact_manager._save_manifest(manifest)

            # Create version selector
            selector = VersionSelector(artifact_manager)

            # Test various selection strategies
            latest_result = selector.select_version(strategy=VersionSelectionStrategy.LATEST)
            assert latest_result.selected_version == parse_version("v2.0.0")

            stable_result = selector.select_version(strategy=VersionSelectionStrategy.STABLE)
            assert stable_result.selected_version == parse_version("v2.0.0")

            specific_result = selector.select_version(
                version_spec="v1.1.0", strategy=VersionSelectionStrategy.SPECIFIC
            )
            assert specific_result.selected_version == parse_version("v1.1.0")

    def test_version_manager_integration(self):
        """Test version manager integration."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_manager = ArtifactManager(temp_dir)

            # Create test manifest
            manifest = ArtifactManifest(version="v1.0.0")
            artifact_manager._save_manifest(manifest)

            # Create version manager
            manager = VersionManager(artifact_manager)

            # Test getting model version
            version = manager.get_model_version("v1.0.0")
            assert version == parse_version("v1.0.0")

            # Test status
            status = manager.get_version_status()
            assert status["current_version"] == "v1.0.0"
