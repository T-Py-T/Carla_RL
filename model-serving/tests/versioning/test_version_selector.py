# Unit tests for version selector and multi-version model support.

import tempfile
import yaml
from pathlib import Path

from src.versioning.version_selector import (
    VersionSelector,
    VersionManager,
    VersionSelectionStrategy,
    VersionSelectionResult,
)
from src.versioning.semantic_version import parse_version
from src.versioning.artifact_manager import ArtifactManager


class TestVersionSelectionStrategy:
    # Test version selection strategy enum.

    def test_strategy_values(self):
        # Test strategy enum values.
        assert VersionSelectionStrategy.LATEST.value == "latest"
        assert VersionSelectionStrategy.STABLE.value == "stable"
        assert VersionSelectionStrategy.SPECIFIC.value == "specific"
        assert VersionSelectionStrategy.COMPATIBLE.value == "compatible"
        assert VersionSelectionStrategy.FALLBACK.value == "fallback"


class TestVersionSelectionResult:
    # Test version selection result dataclass.

    def test_result_creation(self):
        # Test creating version selection result.
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
        # Test result with fallback information.
        version = parse_version("v1.2.3")
        result = VersionSelectionResult(
            selected_version=version,
            strategy_used=VersionSelectionStrategy.STABLE,
            fallback_used=True,
            fallback_reason="Primary strategy failed",
        )

        assert result.fallback_used is True
        assert result.fallback_reason == "Primary strategy failed"


class TestVersionSelector:
    # Test version selector functionality.

    def setup_method(self):
        # Set up test environment with temporary artifacts directory.
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
                    "memory_usage_mb": 300
                }
            }
        }
        
        # Create version directories and model cards
        for version, metadata in self.test_versions.items():
            version_dir = self.artifacts_root / version
            version_dir.mkdir(exist_ok=True)
            
            # Create model card
            model_card_path = version_dir / "model_card.yaml"
            with open(model_card_path, 'w') as f:
                yaml.dump(metadata, f)
        
        # Create artifact manager
        self.artifact_manager = ArtifactManager(str(self.artifacts_root))
        
        # Create version selector
        self.selector = VersionSelector(self.artifact_manager)

    def teardown_method(self):
        # Clean up temporary directory.
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_latest_version_selection(self):
        # Test selecting the latest version.
        result = self.selector.select_version(VersionSelectionStrategy.LATEST)
        
        assert result.selected_version == parse_version("v2.0.0")
        assert result.strategy_used == VersionSelectionStrategy.LATEST
        assert result.fallback_used is False

    def test_stable_version_selection(self):
        # Test selecting the latest stable version.
        result = self.selector.select_version(VersionSelectionStrategy.STABLE)
        
        # Should select v1.1.0 (latest stable, excluding beta)
        assert result.selected_version == parse_version("v1.1.0")
        assert result.strategy_used == VersionSelectionStrategy.STABLE

    def test_specific_version_selection(self):
        # Test selecting a specific version.
        result = self.selector.select_version(
            VersionSelectionStrategy.SPECIFIC, 
            target_version="v1.0.0"
        )
        
        assert result.selected_version == parse_version("v1.0.0")
        assert result.strategy_used == VersionSelectionStrategy.SPECIFIC

    def test_compatible_version_selection(self):
        # Test selecting a compatible version.
        result = self.selector.select_version(
            VersionSelectionStrategy.COMPATIBLE,
            target_version="v1.0.0"
        )
        
        # Should select v1.1.0 (compatible with v1.0.0)
        assert result.selected_version == parse_version("v1.1.0")
        assert result.strategy_used == VersionSelectionStrategy.COMPATIBLE

    def test_fallback_version_selection(self):
        # Test fallback when primary strategy fails.
        result = self.selector.select_version(
            VersionSelectionStrategy.SPECIFIC,
            target_version="v999.0.0"  # Non-existent version
        )
        
        assert result.fallback_used is True
        assert result.fallback_reason is not None
        # Should fallback to latest available version
        assert result.selected_version in [parse_version(v) for v in self.test_versions.keys()]

    def test_version_manager_integration(self):
        # Test integration with version manager.
        manager = VersionManager(self.artifact_manager)
        
        # Test getting available versions
        versions = manager.get_available_versions()
        expected_versions = [parse_version(v) for v in self.test_versions.keys()]
        assert set(versions) == set(expected_versions)
        
        # Test getting model version
        version = manager.get_model_version("v1.0.0")
        assert version == parse_version("v1.0.0")
        
        # Test status
        status = manager.get_version_status()
        assert status["current_version"] == "v1.0.0"
