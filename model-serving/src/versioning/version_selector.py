"""
Version selection logic for multi-version model support.

This module provides intelligent version selection based on criteria like
compatibility, stability, performance requirements, and deployment preferences.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .semantic_version import SemanticVersion
from ..exceptions import ArtifactValidationError, ModelLoadingError


class VersionSelectionStrategy(Enum):
    """
    Strategies for selecting model versions.
    """
    LATEST_STABLE = "latest_stable"       # Latest stable (non-prerelease) version
    LATEST = "latest"                     # Latest version including prereleases
    EXACT = "exact"                       # Exact version match
    COMPATIBLE = "compatible"             # Latest compatible version
    PERFORMANCE_OPTIMIZED = "performance" # Version with best performance metrics


class VersionSelector:
    """
    Intelligent version selector for multi-version model support.
    
    Provides version selection logic based on various strategies and criteria
    including compatibility, stability, performance metrics, and deployment preferences.
    """
    
    def __init__(self, artifacts_root: Path):
        """
        Initialize version selector.
        
        Args:
            artifacts_root: Root directory containing versioned model artifacts
        """
        self.artifacts_root = Path(artifacts_root)
        self._version_cache: Optional[Dict[str, Any]] = None
        self._last_scan_time: Optional[float] = None
        
    def discover_versions(self, force_rescan: bool = False) -> List[SemanticVersion]:
        """
        Discover all available model versions in the artifacts directory.
        
        Args:
            force_rescan: Force rescanning even if cache exists
            
        Returns:
            List of discovered versions sorted in descending order
            
        Raises:
            ModelLoadingError: If artifacts directory is not accessible
        """
        import time
        
        current_time = time.time()
        
        # Use cache if available and recent (within 60 seconds)
        if (not force_rescan and 
            self._version_cache is not None and 
            self._last_scan_time is not None and 
            current_time - self._last_scan_time < 60):
            return self._version_cache.get('versions', [])
        
        if not self.artifacts_root.exists():
            raise ModelLoadingError(f"Artifacts directory not found: {self.artifacts_root}")
        
        versions = []
        version_metadata = {}
        
        for item in self.artifacts_root.iterdir():
            if not item.is_dir():
                continue
                
            # Try to parse as semantic version
            try:
                version = SemanticVersion.parse(item.name)
                versions.append(version)
                
                # Load model card if available for metadata
                model_card_path = item / "model_card.yaml"
                if model_card_path.exists():
                    try:
                        with open(model_card_path) as f:
                            model_card = yaml.safe_load(f)
                        version_metadata[str(version)] = model_card
                    except Exception:
                        # Continue if model card can't be loaded
                        pass
                        
            except ValueError:
                # Skip directories that don't follow semantic versioning
                continue
        
        # Sort versions in descending order (latest first)
        versions.sort(reverse=True)
        
        # Update cache
        self._version_cache = {
            'versions': versions,
            'metadata': version_metadata
        }
        self._last_scan_time = current_time
        
        return versions
    
    def select_version(
        self,
        strategy: VersionSelectionStrategy = VersionSelectionStrategy.LATEST_STABLE,
        exact_version: Optional[str] = None,
        minimum_version: Optional[str] = None,
        exclude_prereleases: bool = True,
        performance_threshold: Optional[Dict[str, float]] = None,
        force_rescan: bool = False
    ) -> Optional[SemanticVersion]:
        """
        Select the best version based on specified strategy and criteria.
        
        Args:
            strategy: Version selection strategy
            exact_version: Exact version to select (for EXACT strategy)
            minimum_version: Minimum acceptable version
            exclude_prereleases: Whether to exclude prerelease versions
            performance_threshold: Performance requirements (e.g., {"latency_p50_ms": 10.0})
            force_rescan: Force rescanning of available versions
            
        Returns:
            Selected version or None if no suitable version found
            
        Raises:
            ModelLoadingError: If selection criteria cannot be satisfied
        """
        versions = self.discover_versions(force_rescan=force_rescan)
        
        if not versions:
            return None
        
        # Parse minimum version if specified
        min_version = None
        if minimum_version:
            try:
                min_version = SemanticVersion.parse(minimum_version)
            except ValueError as e:
                raise ModelLoadingError(f"Invalid minimum version format: {minimum_version}") from e
        
        # Apply basic filtering
        candidate_versions = []
        
        for version in versions:
            # Skip if below minimum version
            if min_version and version < min_version:
                continue
            
            # Skip prereleases if requested
            if exclude_prereleases and version.is_prerelease():
                continue
            
            candidate_versions.append(version)
        
        if not candidate_versions:
            raise ModelLoadingError(
                "No versions match the specified criteria",
                details={
                    "available_versions": [str(v) for v in versions],
                    "minimum_version": minimum_version,
                    "exclude_prereleases": exclude_prereleases
                }
            )
        
        # Apply strategy-specific selection
        if strategy == VersionSelectionStrategy.EXACT:
            if not exact_version:
                raise ModelLoadingError("Exact version must be specified for EXACT strategy")
            
            try:
                target_version = SemanticVersion.parse(exact_version)
                for version in candidate_versions:
                    if version == target_version:
                        return version
                
                raise ModelLoadingError(f"Exact version {exact_version} not found")
                
            except ValueError as e:
                raise ModelLoadingError(f"Invalid exact version format: {exact_version}") from e
        
        elif strategy == VersionSelectionStrategy.LATEST:
            return candidate_versions[0] if candidate_versions else None
        
        elif strategy == VersionSelectionStrategy.LATEST_STABLE:
            stable_versions = [v for v in candidate_versions if v.is_stable()]
            return stable_versions[0] if stable_versions else None
        
        elif strategy == VersionSelectionStrategy.COMPATIBLE:
            # For compatibility, select the latest version that's compatible with minimum version
            if not min_version:
                return candidate_versions[0] if candidate_versions else None
            
            compatible_versions = [
                v for v in candidate_versions 
                if v.is_compatible_with(min_version)
            ]
            return compatible_versions[0] if compatible_versions else None
        
        elif strategy == VersionSelectionStrategy.PERFORMANCE_OPTIMIZED:
            return self._select_performance_optimized_version(
                candidate_versions, performance_threshold
            )
        
        else:
            raise ModelLoadingError(f"Unknown version selection strategy: {strategy}")
    
    def _select_performance_optimized_version(
        self,
        versions: List[SemanticVersion],
        performance_threshold: Optional[Dict[str, float]] = None
    ) -> Optional[SemanticVersion]:
        """
        Select version with best performance metrics.
        
        Args:
            versions: Candidate versions to choose from
            performance_threshold: Minimum performance requirements
            
        Returns:
            Version with best performance or None if none meet threshold
        """
        if not self._version_cache or 'metadata' not in self._version_cache:
            # Fallback to latest if no performance data available
            return versions[0] if versions else None
        
        metadata = self._version_cache['metadata']
        scored_versions = []
        
        for version in versions:
            version_str = str(version)
            if version_str not in metadata:
                continue
            
            model_card = metadata[version_str]
            performance = model_card.get('performance_metrics', {})
            
            # Check if version meets performance thresholds
            if performance_threshold:
                meets_threshold = True
                for metric, threshold in performance_threshold.items():
                    if metric not in performance:
                        meets_threshold = False
                        break
                    
                    # For latency metrics, lower is better
                    if 'latency' in metric.lower() or 'ms' in metric:
                        if performance[metric] > threshold:
                            meets_threshold = False
                            break
                    # For throughput metrics, higher is better
                    elif 'throughput' in metric.lower() or 'rps' in metric:
                        if performance[metric] < threshold:
                            meets_threshold = False
                            break
                
                if not meets_threshold:
                    continue
            
            # Calculate performance score (simple weighted sum)
            score = self._calculate_performance_score(performance)
            scored_versions.append((version, score))
        
        if not scored_versions:
            return None
        
        # Sort by score (higher is better) and return best version
        scored_versions.sort(key=lambda x: x[1], reverse=True)
        return scored_versions[0][0]
    
    def _calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """
        Calculate performance score from metrics.
        
        Args:
            performance: Performance metrics dictionary
            
        Returns:
            Performance score (higher is better)
        """
        score = 0.0
        
        # Latency score (lower latency = higher score)
        if 'latency_p50_ms' in performance:
            latency_p50 = performance['latency_p50_ms']
            score += max(0, 100 - latency_p50)  # Baseline: 100ms is 0 points
        
        # Throughput score (higher throughput = higher score)
        if 'throughput_rps' in performance:
            throughput = performance['throughput_rps']
            score += throughput / 10  # Scale factor
        
        # Memory efficiency score (lower memory = higher score)
        if 'memory_usage_mb' in performance:
            memory_mb = performance['memory_usage_mb']
            score += max(0, (2000 - memory_mb) / 100)  # Baseline: 2GB is 0 points
        
        return score
    
    def get_version_info(self, version: SemanticVersion) -> Dict[str, Any]:
        """
        Get detailed information about a specific version.
        
        Args:
            version: Version to get information for
            
        Returns:
            Dictionary containing version information and metadata
            
        Raises:
            ModelLoadingError: If version information cannot be retrieved
        """
        version_dir = self.artifacts_root / str(version)
        
        if not version_dir.exists():
            raise ModelLoadingError(f"Version directory not found: {version_dir}")
        
        info = {
            'version': str(version),
            'parsed_version': version,
            'directory': version_dir,
            'is_prerelease': version.is_prerelease(),
            'is_stable': version.is_stable(),
            'files': list(version_dir.iterdir()) if version_dir.exists() else []
        }
        
        # Load model card if available
        model_card_path = version_dir / "model_card.yaml"
        if model_card_path.exists():
            try:
                with open(model_card_path) as f:
                    model_card = yaml.safe_load(f)
                info['model_card'] = model_card
                info['performance_metrics'] = model_card.get('performance_metrics', {})
            except Exception as e:
                info['model_card_error'] = str(e)
        
        return info
    
    def validate_version_availability(self, version_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a version is available and properly formatted.
        
        Args:
            version_str: Version string to validate
            
        Returns:
            Tuple of (is_available, error_message)
        """
        try:
            version = SemanticVersion.parse(version_str)
        except ValueError as e:
            return False, f"Invalid version format: {str(e)}"
        
        version_dir = self.artifacts_root / str(version)
        if not version_dir.exists():
            return False, f"Version directory not found: {version_dir}"
        
        # Check for required files
        required_files = ["model_card.yaml"]
        missing_files = []
        
        for file_name in required_files:
            if not (version_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            return False, f"Missing required files: {missing_files}"
        
        return True, None


def select_best_version(
    artifacts_root: Path,
    strategy: VersionSelectionStrategy = VersionSelectionStrategy.LATEST_STABLE,
    **kwargs
) -> Optional[SemanticVersion]:
    """
    Convenience function to select the best version using specified strategy.
    
    Args:
        artifacts_root: Root directory containing versioned model artifacts
        strategy: Version selection strategy
        **kwargs: Additional arguments passed to select_version()
        
    Returns:
        Selected version or None if no suitable version found
    """
    selector = VersionSelector(artifacts_root)
    return selector.select_version(strategy=strategy, **kwargs)


def get_version_from_environment(
    artifacts_root: Path,
    env_var: str = "MODEL_VERSION",
    fallback_strategy: VersionSelectionStrategy = VersionSelectionStrategy.LATEST_STABLE
) -> Optional[SemanticVersion]:
    """
    Get version from environment variable with fallback to selection strategy.
    
    Args:
        artifacts_root: Root directory containing versioned model artifacts
        env_var: Environment variable name containing version
        fallback_strategy: Fallback strategy if env var not set or invalid
        
    Returns:
        Selected version or None if no suitable version found
    """
    selector = VersionSelector(artifacts_root)
    
    # Try to get version from environment
    env_version = os.getenv(env_var)
    if env_version:
        is_available, error_msg = selector.validate_version_availability(env_version)
        if is_available:
            return SemanticVersion.parse(env_version)
        else:
            # Log warning but continue with fallback
            print(f"Warning: Environment version {env_version} not available: {error_msg}")
    
    # Fall back to strategy-based selection
    return selector.select_version(strategy=fallback_strategy)