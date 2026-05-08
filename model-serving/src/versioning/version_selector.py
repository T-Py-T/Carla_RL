"""
Multi-version model support with intelligent version selection logic.

This module provides the `VersionSelector` and `VersionManager` classes used
by the serving layer to discover model artifact versions on disk and pick
the one that best matches the requested strategy (latest, latest stable,
exact, compatible, ...).

Both an `ArtifactManager` (richer metadata + integrity checks) and a plain
`Path` pointing at an artifacts root are accepted. The latter form is what
`server.py` passes in during startup.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from ..exceptions import ModelLoadingError
from .artifact_manager import ArtifactManager, ArtifactManifest
from .semantic_version import SemanticVersion, parse_version

logger = logging.getLogger(__name__)


class VersionSelectionStrategy(Enum):
    """Strategy used when selecting a model version.

    `STABLE` and `LATEST_STABLE` are aliases with the same `.value`, as are
    `SPECIFIC` and `EXACT`, so callers that use either naming convention
    work without a branch.
    """

    LATEST = "latest"
    STABLE = "stable"
    LATEST_STABLE = "stable"
    SPECIFIC = "specific"
    EXACT = "specific"
    COMPATIBLE = "compatible"
    FALLBACK = "fallback"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


class VersionSelectionError(Exception):
    """Exception raised when no suitable model version can be selected."""

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
    """Outcome of a version selection call, including fallback context."""

    selected_version: SemanticVersion
    strategy_used: VersionSelectionStrategy
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    available_versions: Optional[List[SemanticVersion]] = None
    selection_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelVersionInfo:
    """Cached information about a model version discovered on disk."""

    version: SemanticVersion
    manifest: Optional[ArtifactManifest] = None
    is_available: bool = True
    integrity_status: Dict[str, bool] = field(default_factory=dict)
    last_accessed: Optional[str] = None
    usage_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    model_card: Dict[str, Any] = field(default_factory=dict)


def _coerce_to_artifact_manager(
    source: Union[Path, str, ArtifactManager],
) -> Tuple[ArtifactManager, Path]:
    """Normalize `source` into `(ArtifactManager, artifacts_root)`.

    Accepting both forms lets existing callers pass either a rich
    `ArtifactManager` (used by tests) or a plain path to the artifacts
    directory (used by the server).
    """
    if isinstance(source, ArtifactManager):
        # The historical attribute is `artifacts_dir`; fall back to
        # `artifacts_root` for forward compatibility if it ever gets renamed.
        root = getattr(source, "artifacts_dir", None) or getattr(
            source, "artifacts_root", None
        )
        if root is None:
            raise VersionSelectionError(
                "ArtifactManager instance is missing artifacts_dir/artifacts_root"
            )
        return source, Path(root)
    artifacts_root = Path(source)
    return ArtifactManager(str(artifacts_root)), artifacts_root


class VersionSelector:
    """Multi-version model selector with intelligent version resolution.

    Capabilities:
    - Discover versions on disk via directory layout or an `ArtifactManager`
    - Select by strategy (`LATEST`, `STABLE`/`LATEST_STABLE`, `SPECIFIC`/`EXACT`,
      `COMPATIBLE`, `PERFORMANCE_OPTIMIZED`)
    - Fall back automatically if the primary strategy fails
    - Load and expose per-version metadata from `model_card.yaml`
    """

    _CACHE_TTL_SECONDS = 60

    def __init__(
        self,
        source: Union[Path, str, ArtifactManager],
        default_strategy: VersionSelectionStrategy = VersionSelectionStrategy.STABLE,
    ):
        self.artifact_manager, self.artifacts_root = _coerce_to_artifact_manager(source)
        self.default_strategy = default_strategy
        self.version_cache: Dict[SemanticVersion, ModelVersionInfo] = {}
        self.selection_history: List[VersionSelectionResult] = []
        self.custom_selectors: Dict[str, Callable] = {}
        self._last_scan_time: float = 0.0
        self._refresh_version_cache()

    # ------------------------------------------------------------------
    # Discovery + cache
    # ------------------------------------------------------------------
    def _refresh_version_cache(self) -> None:
        """Rebuild the version cache from the artifacts root."""
        self.version_cache.clear()

        if not self.artifacts_root.exists():
            self._last_scan_time = time.time()
            return

        seen: Dict[SemanticVersion, ModelVersionInfo] = {}

        # Prefer the ArtifactManager view when it knows about versions, but
        # always fall back to directory scanning so plain artifact folders
        # without a manifest still show up.
        try:
            manager_versions = self.artifact_manager.list_versions()
        except Exception:
            manager_versions = []

        for version in manager_versions:
            manifest = None
            integrity: Dict[str, bool] = {}
            try:
                manifest = self.artifact_manager.get_manifest(version)
            except Exception:
                manifest = None
            try:
                integrity = self.artifact_manager.verify_integrity(version)
            except Exception:
                integrity = {}
            is_available = all(integrity.values()) if integrity else True
            seen[version] = ModelVersionInfo(
                version=version,
                manifest=manifest,
                is_available=is_available,
                integrity_status=integrity,
            )

        for item in self.artifacts_root.iterdir():
            if not item.is_dir():
                continue
            try:
                version = parse_version(item.name)
            except Exception:
                continue
            info = seen.get(version) or ModelVersionInfo(version=version)
            model_card_path = item / "model_card.yaml"
            if model_card_path.exists():
                try:
                    with open(model_card_path) as f:
                        model_card = yaml.safe_load(f) or {}
                    info.model_card = model_card
                    perf = model_card.get("performance_metrics")
                    if isinstance(perf, dict):
                        # Flatten nested metric groups so callers can read
                        # `latency_p50_ms` or similar without walking sub-dicts.
                        flat: Dict[str, float] = {}
                        for key, value in perf.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, (int, float)):
                                        flat[f"{key}_{sub_key}"] = float(sub_value)
                            elif isinstance(value, (int, float)):
                                flat[key] = float(value)
                        info.performance_metrics = flat
                except Exception:
                    pass
            seen[version] = info

        self.version_cache = seen
        self._last_scan_time = time.time()

    def discover_versions(self, force_rescan: bool = False) -> List[SemanticVersion]:
        """Return all discovered versions, sorted newest first."""
        if force_rescan or (time.time() - self._last_scan_time) > self._CACHE_TTL_SECONDS:
            self._refresh_version_cache()
        return sorted(self.version_cache.keys(), reverse=True)

    def list_available_versions(self, stable_only: bool = False) -> List[SemanticVersion]:
        """Return cached versions in ascending order."""
        versions = list(self.version_cache.keys())
        if stable_only:
            versions = [v for v in versions if v.is_stable()]
        return sorted(versions)

    def get_latest_version(self, stable_only: bool = False) -> Optional[SemanticVersion]:
        versions = self.list_available_versions(stable_only=stable_only)
        return versions[-1] if versions else None

    def get_version_info(self, version: Union[str, SemanticVersion]) -> Dict[str, Any]:
        """Return a dict with model card + performance metrics for `version`."""
        parsed = parse_version(version)
        info = self.version_cache.get(parsed)
        if info is None:
            self._refresh_version_cache()
            info = self.version_cache.get(parsed)
        if info is None:
            raise ModelLoadingError(
                f"Version directory not found for {parsed}",
                details={"artifacts_root": str(self.artifacts_root)},
            )
        version_dir = self.artifacts_root / str(parsed)
        files = list(version_dir.iterdir()) if version_dir.exists() else []
        return {
            "version": str(parsed),
            "parsed_version": parsed,
            "directory": version_dir,
            "is_prerelease": parsed.is_prerelease(),
            "is_stable": parsed.is_stable(),
            "files": files,
            "is_available": info.is_available,
            "integrity_status": info.integrity_status,
            "performance_metrics": info.performance_metrics,
            "model_card": info.model_card,
        }

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def check_version_compatibility(
        self, version: Union[str, SemanticVersion], constraints: List[str]
    ) -> bool:
        parsed = parse_version(version)
        return all(parsed.satisfies(constraint) for constraint in constraints)

    def select_version(
        self,
        strategy: Optional[Union[VersionSelectionStrategy, str]] = None,
        target_version: Optional[Union[str, SemanticVersion]] = None,
        *,
        version_spec: Optional[Union[str, SemanticVersion]] = None,
        exact_version: Optional[str] = None,
        minimum_version: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        exclude_prereleases: bool = False,
        fallback_strategies: Optional[List[VersionSelectionStrategy]] = None,
        performance_weight: float = 0.0,
        performance_threshold: Optional[Dict[str, float]] = None,
        force_rescan: bool = False,
    ) -> VersionSelectionResult:
        """Select the best matching version.

        Accepts both the legacy keyword `version_spec`/`exact_version` and the
        test-suite keyword `target_version`; all three are interchangeable.
        Returns a `VersionSelectionResult` so callers can introspect whether
        a fallback strategy was used.
        """
        if force_rescan:
            self._refresh_version_cache()

        strategy = self._coerce_strategy(strategy) if strategy is not None else self.default_strategy
        spec = target_version or version_spec or exact_version
        if fallback_strategies is None:
            fallback_strategies = [
                VersionSelectionStrategy.LATEST,
                VersionSelectionStrategy.STABLE,
            ]

        if not self.version_cache:
            raise VersionSelectionError("No model versions available")

        try:
            result = self._select_with_strategy(
                spec,
                strategy,
                constraints=constraints,
                minimum_version=minimum_version,
                exclude_prereleases=exclude_prereleases,
                performance_weight=performance_weight,
                performance_threshold=performance_threshold,
            )
        except VersionSelectionError as primary_error:
            logger.warning("Primary selection strategy failed: %s", primary_error)
            for fallback in fallback_strategies:
                if fallback == strategy:
                    continue
                try:
                    fb_result = self._select_with_strategy(
                        spec,
                        fallback,
                        constraints=constraints,
                        minimum_version=minimum_version,
                        exclude_prereleases=exclude_prereleases,
                        performance_weight=performance_weight,
                        performance_threshold=performance_threshold,
                    )
                except VersionSelectionError:
                    continue
                fb_result.fallback_used = True
                fb_result.fallback_reason = (
                    f"Primary strategy '{strategy.name}' failed: {primary_error}"
                )
                self.selection_history.append(fb_result)
                return fb_result
            raise VersionSelectionError(
                f"No suitable version found for spec '{spec}' with strategy '{strategy.name}'",
                requested_version=str(spec) if spec else None,
                available_versions=[str(v) for v in self.version_cache.keys()],
            )

        self.selection_history.append(result)
        return result

    def _select_with_strategy(
        self,
        spec: Optional[Union[str, SemanticVersion]],
        strategy: VersionSelectionStrategy,
        *,
        constraints: Optional[List[str]],
        minimum_version: Optional[str],
        exclude_prereleases: bool,
        performance_weight: float,
        performance_threshold: Optional[Dict[str, float]],
    ) -> VersionSelectionResult:
        available = self.list_available_versions()

        if minimum_version:
            try:
                min_parsed = parse_version(minimum_version)
            except Exception as exc:
                raise VersionSelectionError(
                    f"Invalid minimum version: {minimum_version}"
                ) from exc
            available = [v for v in available if v >= min_parsed]

        if exclude_prereleases:
            available = [v for v in available if v.is_stable()]

        if constraints:
            available = [
                v for v in available if self.check_version_compatibility(v, constraints)
            ]
            stable_only = [v for v in available if v.is_stable()]
            if stable_only:
                available = stable_only

        if not available:
            raise VersionSelectionError("No versions satisfy the given constraints")

        if strategy == VersionSelectionStrategy.LATEST:
            selected = max(available)
        elif strategy in (
            VersionSelectionStrategy.STABLE,
            VersionSelectionStrategy.LATEST_STABLE,
        ):
            stable = [v for v in available if v.is_stable()]
            if not stable:
                raise VersionSelectionError("No stable versions available")
            selected = max(stable)
        elif strategy in (
            VersionSelectionStrategy.SPECIFIC,
            VersionSelectionStrategy.EXACT,
        ):
            if not spec:
                raise VersionSelectionError("Specific version required but not provided")
            target = parse_version(spec)
            if target not in available:
                raise VersionSelectionError(f"Version {target} not available")
            selected = target
        elif strategy == VersionSelectionStrategy.COMPATIBLE:
            if not spec:
                raise VersionSelectionError(
                    "Version specification required for compatible selection"
                )
            target = parse_version(spec)
            compatible = [
                v for v in available if v.major == target.major and v >= target
            ]
            # Compatibility selection should prefer stable releases over
            # prereleases: a caller asking for "compatible with v1.0.0" wants
            # a drop-in replacement, not v1.2.0-beta. Only fall back to
            # prereleases if the caller opted in via `exclude_prereleases=False`
            # (already filtered above) AND no stable version is available.
            stable_compatible = [v for v in compatible if v.is_stable()]
            if stable_compatible:
                compatible = stable_compatible
            if not compatible:
                raise VersionSelectionError(
                    f"No compatible versions found for {target}"
                )
            selected = max(compatible)
        elif strategy == VersionSelectionStrategy.PERFORMANCE_OPTIMIZED:
            selected = self._select_performance_optimized(
                available, performance_threshold
            )
            if selected is None:
                raise VersionSelectionError(
                    "No versions meet the performance threshold"
                )
        else:
            raise VersionSelectionError(f"Unknown selection strategy: {strategy}")

        if performance_weight > 0.0:
            selected = self._apply_performance_weighting(
                available, selected, performance_weight
            )

        return VersionSelectionResult(
            selected_version=selected,
            strategy_used=strategy,
            available_versions=available,
            selection_metadata={
                "version_spec": str(spec) if spec else None,
                "constraints": constraints,
                "minimum_version": minimum_version,
                "exclude_prereleases": exclude_prereleases,
                "performance_weight": performance_weight,
                "performance_threshold": performance_threshold,
            },
        )

    def _apply_performance_weighting(
        self,
        available: List[SemanticVersion],
        default: SemanticVersion,
        weight: float,
    ) -> SemanticVersion:
        metrics = {
            v: self.version_cache[v].performance_metrics
            for v in available
            if self.version_cache.get(v) and self.version_cache[v].performance_metrics
        }
        if not metrics:
            return default
        scores: Dict[SemanticVersion, float] = {}
        for version, perf in metrics.items():
            latency_score = 1.0 / (perf.get("inference_latency_p50_ms", 100.0) + 1.0)
            throughput_score = perf.get("inference_throughput_obs_per_sec", 0.0) / 1000.0
            scores[version] = latency_score + throughput_score
        max_score = max(scores.values()) or 1.0
        weighted: Dict[SemanticVersion, float] = {}
        for version in available:
            base = 1.0 if version == default else 0.5
            perf_score = scores.get(version, 0.0) / max_score
            weighted[version] = (1.0 - weight) * base + weight * perf_score
        return max(available, key=lambda v: weighted.get(v, 0.0))

    def _select_performance_optimized(
        self,
        versions: List[SemanticVersion],
        threshold: Optional[Dict[str, float]],
    ) -> Optional[SemanticVersion]:
        scored: List[Tuple[SemanticVersion, float]] = []
        for version in versions:
            info = self.version_cache.get(version)
            perf = info.performance_metrics if info else {}
            if threshold:
                ok = True
                for metric, bound in threshold.items():
                    if metric not in perf:
                        ok = False
                        break
                    if "latency" in metric.lower() or metric.lower().endswith("_ms"):
                        if perf[metric] > bound:
                            ok = False
                            break
                    else:
                        if perf[metric] < bound:
                            ok = False
                            break
                if not ok:
                    continue
            scored.append((version, self._performance_score(perf)))
        if not scored:
            return None
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0]

    @staticmethod
    def _performance_score(perf: Dict[str, float]) -> float:
        score = 0.0
        if "inference_latency_p50_ms" in perf:
            score += max(0.0, 100.0 - perf["inference_latency_p50_ms"])
        if "inference_throughput_obs_per_sec" in perf:
            score += perf["inference_throughput_obs_per_sec"] / 10.0
        if "memory_usage_mb" in perf:
            score += max(0.0, (2000.0 - perf["memory_usage_mb"]) / 100.0)
        return score

    @staticmethod
    def _coerce_strategy(
        strategy: Union[VersionSelectionStrategy, str],
    ) -> VersionSelectionStrategy:
        if isinstance(strategy, VersionSelectionStrategy):
            return strategy
        if isinstance(strategy, str):
            key = strategy.upper()
            if hasattr(VersionSelectionStrategy, key):
                return VersionSelectionStrategy[key]
            try:
                return VersionSelectionStrategy(strategy.lower())
            except ValueError:
                pass
        raise VersionSelectionError(f"Unknown selection strategy: {strategy}")

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------
    def register_custom_selector(self, name: str, selector_func: Callable) -> None:
        self.custom_selectors[name] = selector_func

    def get_selection_history(
        self, limit: Optional[int] = None
    ) -> List[VersionSelectionResult]:
        return (
            self.selection_history.copy()
            if limit is None
            else self.selection_history[-limit:]
        )

    def update_performance_metrics(
        self, version: Union[str, SemanticVersion], metrics: Dict[str, float]
    ) -> None:
        parsed = parse_version(version)
        info = self.version_cache.get(parsed)
        if info is not None:
            info.performance_metrics.update(metrics)

    def validate_version_availability(
        self, version_str: str
    ) -> Tuple[bool, Optional[str]]:
        """Return `(is_available, error_message)` for the requested version."""
        try:
            version = parse_version(version_str)
        except Exception as exc:
            return False, f"Invalid version format: {exc}"
        version_dir = self.artifacts_root / str(version)
        if not version_dir.exists():
            return False, f"Version directory not found: {version_dir}"
        required = ["model_card.yaml"]
        missing = [name for name in required if not (version_dir / name).exists()]
        if missing:
            return False, f"Missing required files: {missing}"
        return True, None

    def get_recommended_version(
        self,
        use_case: str = "general",
        performance_requirements: Optional[Dict[str, float]] = None,
    ) -> Optional[SemanticVersion]:
        stable = self.list_available_versions(stable_only=True)
        if not stable:
            return None
        if use_case == "development":
            all_versions = self.list_available_versions(stable_only=False)
            return max(all_versions) if all_versions else None
        return max(stable)


class VersionManager:
    """High-level wrapper over `VersionSelector` for simple call-sites."""

    def __init__(
        self,
        source: Union[Path, str, ArtifactManager],
    ):
        self.artifact_manager, self.artifacts_root = _coerce_to_artifact_manager(source)
        self.selector = VersionSelector(self.artifact_manager)
        self.current_version: Optional[SemanticVersion] = None
        self.version_callbacks: List[Callable[[SemanticVersion], None]] = []

    def get_available_versions(self) -> List[SemanticVersion]:
        return self.selector.list_available_versions()

    def get_model_version(
        self,
        version_spec: Optional[str] = None,
        auto_fallback: bool = True,
    ) -> SemanticVersion:
        try:
            if version_spec:
                result = self.selector.select_version(
                    VersionSelectionStrategy.SPECIFIC, target_version=version_spec
                )
            else:
                result = self.selector.select_version(VersionSelectionStrategy.STABLE)
            self.current_version = result.selected_version
            self._notify_version_change(result.selected_version)
            return result.selected_version
        except VersionSelectionError as exc:
            if not auto_fallback:
                raise
            logger.warning("Version selection failed, using fallback: %s", exc)
            fallback = self.selector.select_version(VersionSelectionStrategy.STABLE)
            self.current_version = fallback.selected_version
            self._notify_version_change(fallback.selected_version)
            return fallback.selected_version

    def register_version_change_callback(
        self, callback: Callable[[SemanticVersion], None]
    ) -> None:
        self.version_callbacks.append(callback)

    def _notify_version_change(self, version: SemanticVersion) -> None:
        for callback in self.version_callbacks:
            try:
                callback(version)
            except Exception as exc:
                logger.error("Error in version change callback: %s", exc)

    def get_version_status(self) -> Dict[str, Any]:
        return {
            "current_version": str(self.current_version) if self.current_version else None,
            "available_versions": [
                str(v) for v in self.selector.list_available_versions()
            ],
            "stable_versions": [
                str(v) for v in self.selector.list_available_versions(stable_only=True)
            ],
            "latest_version": (
                str(self.selector.get_latest_version())
                if self.selector.get_latest_version()
                else None
            ),
            "selection_history_count": len(self.selector.get_selection_history()),
        }


def select_best_version(
    artifacts_root: Union[Path, str, ArtifactManager],
    strategy: VersionSelectionStrategy = VersionSelectionStrategy.LATEST_STABLE,
    **kwargs: Any,
) -> Optional[SemanticVersion]:
    """Select the best matching version, returning a bare `SemanticVersion`."""
    selector = VersionSelector(artifacts_root)
    try:
        result = selector.select_version(strategy=strategy, **kwargs)
        return result.selected_version
    except VersionSelectionError:
        return None


def get_version_from_environment(
    artifacts_root: Union[Path, str, ArtifactManager],
    env_var: str = "MODEL_VERSION",
    fallback_strategy: VersionSelectionStrategy = VersionSelectionStrategy.LATEST_STABLE,
) -> Optional[SemanticVersion]:
    """Resolve a model version from `env_var`, falling back to `fallback_strategy`.

    Designed for the serving entrypoint: operators pin a version via
    `MODEL_VERSION` when they want determinism, otherwise the server picks
    the latest stable release on disk.
    """
    selector = VersionSelector(artifacts_root)
    env_version = os.getenv(env_var)
    if env_version:
        available, error = selector.validate_version_availability(env_version)
        if available:
            return parse_version(env_version)
        logger.warning(
            "Environment version %s not available: %s", env_version, error
        )
    try:
        result = selector.select_version(strategy=fallback_strategy)
        return result.selected_version
    except VersionSelectionError:
        return None
