"""
Multi-version model support with intelligent version selection logic.

This module provides comprehensive version selection capabilities for managing
multiple model versions, including automatic version resolution, fallback
strategies, and version constraint handling.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Callable, Any

from .semantic_version import SemanticVersion, parse_version
from .artifact_manager import ArtifactManager, ArtifactManifest


logger = logging.getLogger(__name__)


class VersionSelectionStrategy(Enum):
    """Strategy for selecting model versions."""

    LATEST = "latest"  # Always use the latest version
    STABLE = "stable"  # Use the latest stable version
    SPECIFIC = "specific"  # Use a specific version
    COMPATIBLE = "compatible"  # Use the latest compatible version
    FALLBACK = "fallback"  # Use fallback strategy with multiple options


class VersionSelectionError(Exception):
    """Exception raised for version selection errors."""

    def __init__(
        self,
        message: str,
        requested_version: Optional[str] = None,
        available_versions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.requested_version = requested_version
        self.available_versions = available_versions


@dataclass
class VersionSelectionResult:
    """Result of version selection operation."""

    selected_version: SemanticVersion
    strategy_used: VersionSelectionStrategy
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    available_versions: List[SemanticVersion] = None
    selection_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.selection_metadata is None:
            self.selection_metadata = {}


@dataclass
class ModelVersionInfo:
    """Information about a model version."""

    version: SemanticVersion
    manifest: ArtifactManifest
    is_available: bool
    integrity_status: Dict[str, bool]
    last_accessed: Optional[str] = None
    usage_count: int = 0
    performance_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class VersionSelector:
    """
    Multi-version model selector with intelligent version resolution.

    Provides comprehensive version selection capabilities including:
    - Automatic version resolution based on constraints
    - Fallback strategies for unavailable versions
    - Performance-based version selection
    - Version compatibility checking
    """

    def __init__(
        self,
        artifact_manager: ArtifactManager,
        default_strategy: VersionSelectionStrategy = VersionSelectionStrategy.STABLE,
    ):
        """
        Initialize version selector.

        Args:
            artifact_manager: Artifact manager instance
            default_strategy: Default selection strategy
        """
        self.artifact_manager = artifact_manager
        self.default_strategy = default_strategy
        self.version_cache: Dict[SemanticVersion, ModelVersionInfo] = {}
        self.selection_history: List[VersionSelectionResult] = []
        self.custom_selectors: Dict[str, Callable] = {}

        # Load available versions into cache
        self._refresh_version_cache()

    def select_version(
        self,
        version_spec: Optional[Union[str, SemanticVersion]] = None,
        strategy: Optional[VersionSelectionStrategy] = None,
        constraints: Optional[List[str]] = None,
        fallback_strategies: Optional[List[VersionSelectionStrategy]] = None,
        performance_weight: float = 0.0,
    ) -> VersionSelectionResult:
        """
        Select a model version based on specified criteria.

        Args:
            version_spec: Specific version or version constraint
            strategy: Selection strategy to use
            constraints: List of version constraints to satisfy
            fallback_strategies: Fallback strategies if primary selection fails
            performance_weight: Weight for performance-based selection (0.0-1.0)

        Returns:
            VersionSelectionResult with selected version and metadata

        Raises:
            VersionSelectionError: If no suitable version can be found
        """
        strategy = strategy or self.default_strategy
        if fallback_strategies is None:
            fallback_strategies = [VersionSelectionStrategy.LATEST, VersionSelectionStrategy.STABLE]

        # Refresh version cache
        self._refresh_version_cache()

        if not self.version_cache:
            raise VersionSelectionError("No model versions available")

        # Try primary strategy
        try:
            result = self._select_with_strategy(
                version_spec, strategy, constraints, performance_weight
            )
            self.selection_history.append(result)
            return result
        except VersionSelectionError as e:
            logger.warning(f"Primary selection strategy failed: {e}")

            # Try fallback strategies
            for fallback_strategy in fallback_strategies:
                try:
                    result = self._select_with_strategy(
                        version_spec, fallback_strategy, constraints, performance_weight
                    )
                    result.fallback_used = True
                    result.fallback_reason = f"Primary strategy '{strategy.value}' failed: {e}"
                    self.selection_history.append(result)
                    return result
                except VersionSelectionError:
                    continue

            # If all strategies fail
            available_versions = [str(v) for v in self.version_cache.keys()]
            raise VersionSelectionError(
                f"No suitable version found for spec '{version_spec}' with strategy '{strategy.value}'",
                str(version_spec) if version_spec else None,
                available_versions,
            )

    def get_version_info(self, version: Union[str, SemanticVersion]) -> Optional[ModelVersionInfo]:
        """
        Get detailed information about a specific version.

        Args:
            version: Version to get information for

        Returns:
            ModelVersionInfo if found, None otherwise
        """
        version = parse_version(version)
        return self.version_cache.get(version)

    def list_available_versions(self, stable_only: bool = False) -> List[SemanticVersion]:
        """
        List all available versions.

        Args:
            stable_only: If True, return only stable versions

        Returns:
            List of available versions
        """
        versions = list(self.version_cache.keys())

        if stable_only:
            versions = [v for v in versions if v.is_stable()]

        return sorted(versions)

    def get_latest_version(self, stable_only: bool = False) -> Optional[SemanticVersion]:
        """
        Get the latest available version.

        Args:
            stable_only: If True, return only stable versions

        Returns:
            Latest version if available, None otherwise
        """
        versions = self.list_available_versions(stable_only)
        return versions[-1] if versions else None

    def check_version_compatibility(
        self, version: Union[str, SemanticVersion], constraints: List[str]
    ) -> bool:
        """
        Check if a version satisfies given constraints.

        Args:
            version: Version to check
            constraints: List of version constraints

        Returns:
            True if version satisfies all constraints
        """
        version = parse_version(version)

        for constraint in constraints:
            if not version.satisfies(constraint):
                return False

        return True

    def register_custom_selector(self, name: str, selector_func: Callable) -> None:
        """
        Register a custom version selector function.

        Args:
            name: Name for the custom selector
            selector_func: Function that takes (versions, criteria) and returns selected version
        """
        self.custom_selectors[name] = selector_func

    def get_selection_history(self, limit: Optional[int] = None) -> List[VersionSelectionResult]:
        """
        Get version selection history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of recent selection results
        """
        if limit is None:
            return self.selection_history.copy()

        return self.selection_history[-limit:]

    def _select_with_strategy(
        self,
        version_spec: Optional[Union[str, SemanticVersion]],
        strategy: VersionSelectionStrategy,
        constraints: Optional[List[str]],
        performance_weight: float,
    ) -> VersionSelectionResult:
        """Select version using specific strategy."""

        available_versions = self.list_available_versions()

        # Apply constraints if provided
        if constraints:
            available_versions = [
                v for v in available_versions if self.check_version_compatibility(v, constraints)
            ]

            # For constraint filtering, prefer stable versions over prerelease
            stable_versions = [v for v in available_versions if v.is_stable()]
            if stable_versions:
                available_versions = stable_versions

        if not available_versions:
            raise VersionSelectionError("No versions satisfy the given constraints")

        # Select based on strategy
        if strategy == VersionSelectionStrategy.LATEST:
            selected_version = max(available_versions)
        elif strategy == VersionSelectionStrategy.STABLE:
            stable_versions = [v for v in available_versions if v.is_stable()]
            if not stable_versions:
                raise VersionSelectionError("No stable versions available")
            selected_version = max(stable_versions)
        elif strategy == VersionSelectionStrategy.SPECIFIC:
            if not version_spec:
                raise VersionSelectionError("Specific version required but not provided")
            selected_version = parse_version(version_spec)
            if selected_version not in available_versions:
                raise VersionSelectionError(f"Version {selected_version} not available")
        elif strategy == VersionSelectionStrategy.COMPATIBLE:
            if not version_spec:
                raise VersionSelectionError(
                    "Version specification required for compatible selection"
                )
            target_version = parse_version(version_spec)
            compatible_versions = [
                v
                for v in available_versions
                if v.major == target_version.major and v >= target_version
            ]
            if not compatible_versions:
                raise VersionSelectionError(f"No compatible versions found for {target_version}")
            selected_version = max(compatible_versions)
        else:
            raise VersionSelectionError(f"Unknown selection strategy: {strategy}")

        # Apply performance weighting if specified
        if performance_weight > 0.0:
            selected_version = self._apply_performance_weighting(
                available_versions, selected_version, performance_weight
            )

        return VersionSelectionResult(
            selected_version=selected_version,
            strategy_used=strategy,
            available_versions=available_versions,
            selection_metadata={
                "version_spec": str(version_spec) if version_spec else None,
                "constraints": constraints,
                "performance_weight": performance_weight,
            },
        )

    def _apply_performance_weighting(
        self,
        available_versions: List[SemanticVersion],
        default_version: SemanticVersion,
        performance_weight: float,
    ) -> SemanticVersion:
        """Apply performance-based weighting to version selection."""

        # Get performance metrics for available versions
        version_metrics = {}
        for version in available_versions:
            version_info = self.version_cache.get(version)
            if version_info and version_info.performance_metrics:
                version_metrics[version] = version_info.performance_metrics

        if not version_metrics:
            return default_version

        # Calculate performance scores (higher is better)
        performance_scores = {}
        for version, metrics in version_metrics.items():
            # Simple scoring based on latency (lower is better) and throughput (higher is better)
            latency_score = 1.0 / (metrics.get("avg_latency_ms", 100.0) + 1.0)
            throughput_score = metrics.get("throughput_rps", 0.0) / 1000.0
            performance_scores[version] = latency_score + throughput_score

        # Weight the performance scores
        max_performance = max(performance_scores.values()) if performance_scores else 1.0
        weighted_scores = {}

        for version in available_versions:
            base_score = 1.0 if version == default_version else 0.5
            performance_score = performance_scores.get(version, 0.0) / max_performance
            weighted_scores[version] = (
                1.0 - performance_weight
            ) * base_score + performance_weight * performance_score

        # Select version with highest weighted score
        best_version = max(available_versions, key=lambda v: weighted_scores.get(v, 0.0))

        logger.info(
            f"Performance-weighted selection: {best_version} (score: {weighted_scores[best_version]:.3f})"
        )
        return best_version

    def _refresh_version_cache(self) -> None:
        """Refresh the version cache with current available versions."""
        self.version_cache.clear()

        for version in self.artifact_manager.list_versions():
            manifest = self.artifact_manager.get_manifest(version)
            if not manifest:
                continue

            # Check integrity
            integrity_status = self.artifact_manager.verify_integrity(version)
            is_available = all(integrity_status.values())

            version_info = ModelVersionInfo(
                version=version,
                manifest=manifest,
                is_available=is_available,
                integrity_status=integrity_status,
            )

            self.version_cache[version] = version_info

    def update_performance_metrics(
        self, version: Union[str, SemanticVersion], metrics: Dict[str, float]
    ) -> None:
        """
        Update performance metrics for a version.

        Args:
            version: Version to update
            metrics: Performance metrics dictionary
        """
        version = parse_version(version)
        if version in self.version_cache:
            self.version_cache[version].performance_metrics.update(metrics)

    def get_recommended_version(
        self, use_case: str = "general", performance_requirements: Optional[Dict[str, float]] = None
    ) -> Optional[SemanticVersion]:
        """
        Get recommended version based on use case and performance requirements.

        Args:
            use_case: Use case identifier (e.g., "production", "development", "testing")
            performance_requirements: Performance requirements dictionary

        Returns:
            Recommended version if available
        """
        available_versions = self.list_available_versions(stable_only=True)

        if not available_versions:
            return None

        # Use case-based recommendations
        if use_case == "production":
            # Prefer latest stable version for production
            return max(available_versions)
        elif use_case == "development":
            # Prefer latest version (including prerelease) for development
            all_versions = self.list_available_versions(stable_only=False)
            return max(all_versions) if all_versions else None
        elif use_case == "testing":
            # Prefer stable version for testing
            return max(available_versions)
        else:
            # Default to latest stable
            return max(available_versions)


class VersionManager:
    """
    High-level version management interface.

    Provides simplified interface for common version management operations
    including automatic version selection, fallback handling, and monitoring.
    """

    def __init__(self, artifact_manager: ArtifactManager):
        """
        Initialize version manager.

        Args:
            artifact_manager: Artifact manager instance
        """
        self.artifact_manager = artifact_manager
        self.selector = VersionSelector(artifact_manager)
        self.current_version: Optional[SemanticVersion] = None
        self.version_callbacks: List[Callable] = []

    def get_model_version(
        self, version_spec: Optional[str] = None, auto_fallback: bool = True
    ) -> SemanticVersion:
        """
        Get model version with automatic fallback.

        Args:
            version_spec: Version specification
            auto_fallback: Whether to use automatic fallback

        Returns:
            Selected model version
        """
        try:
            result = self.selector.select_version(version_spec)
            self.current_version = result.selected_version
            self._notify_version_change(result.selected_version)
            return result.selected_version
        except VersionSelectionError as e:
            if auto_fallback:
                logger.warning(f"Version selection failed, using fallback: {e}")
                fallback_result = self.selector.select_version(
                    strategy=VersionSelectionStrategy.STABLE
                )
                self.current_version = fallback_result.selected_version
                self._notify_version_change(fallback_result.selected_version)
                return fallback_result.selected_version
            else:
                raise

    def register_version_change_callback(self, callback: Callable[[SemanticVersion], None]) -> None:
        """
        Register callback for version changes.

        Args:
            callback: Function to call when version changes
        """
        self.version_callbacks.append(callback)

    def _notify_version_change(self, new_version: SemanticVersion) -> None:
        """Notify registered callbacks of version change."""
        for callback in self.version_callbacks:
            try:
                callback(new_version)
            except Exception as e:
                logger.error(f"Error in version change callback: {e}")

    def get_version_status(self) -> Dict[str, Any]:
        """
        Get current version status and statistics.

        Returns:
            Version status dictionary
        """
        return {
            "current_version": str(self.current_version) if self.current_version else None,
            "available_versions": [str(v) for v in self.selector.list_available_versions()],
            "stable_versions": [
                str(v) for v in self.selector.list_available_versions(stable_only=True)
            ],
            "latest_version": str(self.selector.get_latest_version())
            if self.selector.get_latest_version()
            else None,
            "selection_history_count": len(self.selector.get_selection_history()),
        }
