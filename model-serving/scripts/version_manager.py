#!/usr/bin/env python3
"""
CLI tool for version management and multi-version model support.

Provides command-line interface for managing model versions, selecting versions,
and monitoring version status in the Policy-as-a-Service system.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from versioning import (
    VersionManager, VersionSelector, VersionSelectionStrategy,
    VersionSelectionError, ArtifactManager
)
from semantic_version import parse_version


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_artifact_manager(artifacts_dir: str) -> ArtifactManager:
    """Set up artifact manager with specified directory."""
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        logger.error(f"Artifacts directory does not exist: {artifacts_dir}")
        sys.exit(1)
    
    return ArtifactManager(artifacts_path)


def list_versions(artifact_manager: ArtifactManager, stable_only: bool = False) -> None:
    """List available versions."""
    selector = VersionSelector(artifact_manager)
    versions = selector.list_available_versions(stable_only=stable_only)
    
    if not versions:
        print("No versions available")
        return
    
    print(f"Available versions ({'stable only' if stable_only else 'all'}):")
    for version in versions:
        version_info = selector.get_version_info(version)
        status = "✓" if version_info and version_info.is_available else "✗"
        print(f"  {status} {version}")
        
        if version_info and version_info.performance_metrics:
            metrics = version_info.performance_metrics
            latency = metrics.get('avg_latency_ms', 'N/A')
            throughput = metrics.get('throughput_rps', 'N/A')
            print(f"    Performance: {latency}ms latency, {throughput} RPS")


def select_version(artifact_manager: ArtifactManager, 
                  version_spec: Optional[str] = None,
                  strategy: str = "stable",
                  constraints: Optional[List[str]] = None,
                  performance_weight: float = 0.0,
                  json_output: bool = False) -> None:
    """Select a version using specified criteria."""
    selector = VersionSelector(artifact_manager)
    
    try:
        strategy_enum = VersionSelectionStrategy(strategy)
        result = selector.select_version(
            version_spec=version_spec,
            strategy=strategy_enum,
            constraints=constraints,
            performance_weight=performance_weight
        )
        
        if json_output:
            output = {
                "selected_version": str(result.selected_version),
                "strategy_used": result.strategy_used.value,
                "fallback_used": result.fallback_used,
                "fallback_reason": result.fallback_reason,
                "available_versions": [str(v) for v in result.available_versions] if result.available_versions else None,
                "selection_metadata": result.selection_metadata
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Selected version: {result.selected_version}")
            print(f"Strategy used: {result.strategy_used.value}")
            if result.fallback_used:
                print(f"Fallback used: {result.fallback_reason}")
            
            if result.available_versions:
                print(f"Available versions: {', '.join(str(v) for v in result.available_versions)}")
    
    except VersionSelectionError as e:
        logger.error(f"Version selection failed: {e}")
        if e.available_versions:
            print(f"Available versions: {', '.join(e.available_versions)}")
        sys.exit(1)


def get_version_info(artifact_manager: ArtifactManager, version: str, json_output: bool = False) -> None:
    """Get detailed information about a specific version."""
    selector = VersionSelector(artifact_manager)
    version_obj = parse_version(version)
    info = selector.get_version_info(version_obj)
    
    if not info:
        print(f"Version {version} not found")
        return
    
    if json_output:
        output = {
            "version": str(info.version),
            "is_available": info.is_available,
            "integrity_status": info.integrity_status,
            "last_accessed": info.last_accessed,
            "usage_count": info.usage_count,
            "performance_metrics": info.performance_metrics,
            "manifest": {
                "version": info.manifest.version,
                "created_at": info.manifest.created_at,
                "model_type": info.manifest.model_type,
                "description": info.manifest.description,
                "artifacts": info.manifest.artifacts,
                "dependencies": info.manifest.dependencies,
                "metadata": info.manifest.metadata
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Version: {info.version}")
        print(f"Available: {info.is_available}")
        print(f"Integrity Status: {info.integrity_status}")
        print(f"Usage Count: {info.usage_count}")
        
        if info.performance_metrics:
            print("Performance Metrics:")
            for metric, value in info.performance_metrics.items():
                print(f"  {metric}: {value}")
        
        print(f"Created: {info.manifest.created_at}")
        print(f"Model Type: {info.manifest.model_type}")
        print(f"Description: {info.manifest.description}")
        print(f"Artifacts: {len(info.manifest.artifacts)} files")


def check_compatibility(artifact_manager: ArtifactManager, 
                       version: str, 
                       constraints: List[str]) -> None:
    """Check if a version satisfies given constraints."""
    selector = VersionSelector(artifact_manager)
    version_obj = parse_version(version)
    
    compatible = selector.check_version_compatibility(version_obj, constraints)
    
    print(f"Version {version} is {'compatible' if compatible else 'incompatible'} with constraints:")
    for constraint in constraints:
        print(f"  {constraint}: {version_obj.satisfies(constraint)}")


def get_recommendation(artifact_manager: ArtifactManager, 
                      use_case: str = "general",
                      performance_requirements: Optional[dict] = None) -> None:
    """Get recommended version for a use case."""
    selector = VersionSelector(artifact_manager)
    recommended = selector.get_recommended_version(use_case, performance_requirements)
    
    if recommended:
        print(f"Recommended version for {use_case}: {recommended}")
    else:
        print(f"No version available for use case: {use_case}")


def show_selection_history(artifact_manager: ArtifactManager, limit: Optional[int] = None) -> None:
    """Show version selection history."""
    selector = VersionSelector(artifact_manager)
    history = selector.get_selection_history(limit)
    
    if not history:
        print("No selection history available")
        return
    
    print(f"Version Selection History (last {len(history)} selections):")
    for i, result in enumerate(history, 1):
        print(f"  {i}. {result.selected_version} ({result.strategy_used.value})")
        if result.fallback_used:
            print(f"     Fallback: {result.fallback_reason}")


def update_performance_metrics(artifact_manager: ArtifactManager,
                              version: str,
                              metrics_file: str) -> None:
    """Update performance metrics for a version."""
    selector = VersionSelector(artifact_manager)
    version_obj = parse_version(version)
    
    # Load metrics from file
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    selector.update_performance_metrics(version_obj, metrics)
    print(f"Updated performance metrics for version {version}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Version management CLI for Policy-as-a-Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available versions
  python version_manager.py list --artifacts-dir /path/to/artifacts
  
  # Select latest stable version
  python version_manager.py select --artifacts-dir /path/to/artifacts --strategy stable
  
  # Select specific version
  python version_manager.py select --artifacts-dir /path/to/artifacts --version v1.2.3 --strategy specific
  
  # Get version information
  python version_manager.py info --artifacts-dir /path/to/artifacts --version v1.2.3
  
  # Check version compatibility
  python version_manager.py check --artifacts-dir /path/to/artifacts --version v1.2.3 --constraints ">=1.0.0" "<2.0.0"
        """
    )
    
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Path to artifacts directory"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available versions")
    list_parser.add_argument(
        "--stable-only",
        action="store_true",
        help="Show only stable versions"
    )
    
    # Select command
    select_parser = subparsers.add_parser("select", help="Select a version")
    select_parser.add_argument(
        "--version",
        help="Version specification"
    )
    select_parser.add_argument(
        "--strategy",
        choices=["latest", "stable", "specific", "compatible"],
        default="stable",
        help="Selection strategy"
    )
    select_parser.add_argument(
        "--constraints",
        nargs="*",
        help="Version constraints"
    )
    select_parser.add_argument(
        "--performance-weight",
        type=float,
        default=0.0,
        help="Performance weighting (0.0-1.0)"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get version information")
    info_parser.add_argument(
        "--version",
        required=True,
        help="Version to get information for"
    )
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check version compatibility")
    check_parser.add_argument(
        "--version",
        required=True,
        help="Version to check"
    )
    check_parser.add_argument(
        "--constraints",
        nargs="+",
        required=True,
        help="Constraints to check against"
    )
    
    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Get version recommendation")
    recommend_parser.add_argument(
        "--use-case",
        choices=["production", "development", "testing", "general"],
        default="general",
        help="Use case for recommendation"
    )
    recommend_parser.add_argument(
        "--performance-requirements",
        help="Path to JSON file with performance requirements"
    )
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show selection history")
    history_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of history entries"
    )
    
    # Update metrics command
    metrics_parser = subparsers.add_parser("update-metrics", help="Update performance metrics")
    metrics_parser.add_argument(
        "--version",
        required=True,
        help="Version to update metrics for"
    )
    metrics_parser.add_argument(
        "--metrics-file",
        required=True,
        help="Path to JSON file with metrics"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up artifact manager
    try:
        artifact_manager = setup_artifact_manager(args.artifacts_dir)
    except Exception as e:
        logger.error(f"Failed to set up artifact manager: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "list":
            list_versions(artifact_manager, args.stable_only)
        
        elif args.command == "select":
            select_version(
                artifact_manager,
                args.version,
                args.strategy,
                args.constraints,
                args.performance_weight,
                args.json
            )
        
        elif args.command == "info":
            get_version_info(artifact_manager, args.version, args.json)
        
        elif args.command == "check":
            check_compatibility(artifact_manager, args.version, args.constraints)
        
        elif args.command == "recommend":
            performance_requirements = None
            if args.performance_requirements:
                with open(args.performance_requirements, 'r') as f:
                    performance_requirements = json.load(f)
            
            get_recommendation(artifact_manager, args.use_case, performance_requirements)
        
        elif args.command == "history":
            show_selection_history(artifact_manager, args.limit)
        
        elif args.command == "update-metrics":
            update_performance_metrics(artifact_manager, args.version, args.metrics_file)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
