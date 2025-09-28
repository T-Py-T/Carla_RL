#!/usr/bin/env uv run python
"""
CLI tool for managing model artifacts with integrity validation.

This script provides command-line interface for pinning, validating,
and managing model artifacts with SHA-256 hash integrity checking.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from versioning.artifact_manager import ArtifactManager, ArtifactIntegrityError
from versioning.semantic_version import VersionError


def pin_artifacts(manager: ArtifactManager, version: str, artifacts_dir: Path, args) -> None:
    """Pin artifacts for a version."""
    try:
        print(f"Pinning artifacts for version {version}...")
        manifest = manager.pin_artifacts(version, artifacts_dir)
        
        print(f"Successfully pinned {len(manifest.artifacts)} artifacts:")
        for artifact_path, hash_value in manifest.artifacts.items():
            print(f"  {artifact_path}: {hash_value}")
        
        if args.output:
            output_file = Path(args.output)
            with open(output_file, 'w') as f:
                json.dump(manifest.to_dict(), f, indent=2)
            print(f"Manifest saved to {output_file}")
            
    except ArtifactIntegrityError as e:
        print(f"Error pinning artifacts: {e}", file=sys.stderr)
        sys.exit(1)
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def validate_artifacts(manager: ArtifactManager, version: str, artifacts_dir: Path) -> None:
    """Validate artifacts for a version."""
    try:
        print(f"Validating artifacts for version {version}...")
        result = manager.validate_artifacts(version, artifacts_dir)
        
        if result:
            print("All artifacts are valid!")
        else:
            print("Artifact validation failed!", file=sys.stderr)
            sys.exit(1)
            
    except ArtifactIntegrityError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        if e.artifact_path:
            print(f"  File: {e.artifact_path}", file=sys.stderr)
        if e.expected_hash:
            print(f"  Expected hash: {e.expected_hash}", file=sys.stderr)
        if e.actual_hash:
            print(f"  Actual hash: {e.actual_hash}", file=sys.stderr)
        sys.exit(1)
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def list_artifacts(manager: ArtifactManager, version: str) -> None:
    """List artifacts for a version."""
    try:
        artifacts = manager.list_artifacts(version)
        
        if not artifacts:
            print(f"No artifacts found for version {version}")
            return
        
        print(f"Artifacts for version {version}:")
        for artifact in artifacts:
            hash_value = manager.get_artifact_hash(version, artifact)
            print(f"  {artifact}: {hash_value}")
            
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def list_versions(manager: ArtifactManager) -> None:
    """List all available versions."""
    versions = manager.list_versions()
    
    if not versions:
        print("No versions found")
        return
    
    print("Available versions:")
    for version in versions:
        manifest = manager.get_manifest(version)
        artifact_count = len(manifest.artifacts) if manifest else 0
        print(f"  {version} ({artifact_count} artifacts)")


def show_manifest(manager: ArtifactManager, version: str) -> None:
    """Show manifest for a version."""
    try:
        manifest = manager.get_manifest(version)
        
        if not manifest:
            print(f"No manifest found for version {version}")
            return
        
        print(f"Manifest for version {version}:")
        print(json.dumps(manifest.to_dict(), indent=2))
        
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def verify_integrity(manager: ArtifactManager, version: str) -> None:
    """Verify integrity of artifacts for a version."""
    try:
        print(f"Verifying integrity for version {version}...")
        results = manager.verify_integrity(version)
        
        if not results:
            print(f"No artifacts found for version {version}")
            return
        
        valid_count = sum(results.values())
        total_count = len(results)
        
        print(f"Integrity check results: {valid_count}/{total_count} artifacts valid")
        
        for artifact_path, is_valid in results.items():
            status = "VALID" if is_valid else "INVALID"
            print(f"  {artifact_path}: {status}")
        
        if valid_count != total_count:
            print("Some artifacts failed integrity check!", file=sys.stderr)
            sys.exit(1)
        else:
            print("All artifacts passed integrity check!")
            
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def generate_report(manager: ArtifactManager, version: str, output_file: Optional[Path]) -> None:
    """Generate comprehensive integrity report."""
    try:
        print(f"Generating integrity report for version {version}...")
        report = manager.get_integrity_report(version)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_file}")
        else:
            print(json.dumps(report, indent=2))
            
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def delete_version(manager: ArtifactManager, version: str, force: bool = False) -> None:
    """Delete a version and its artifacts."""
    try:
        if not force:
            response = input(f"Are you sure you want to delete version {version}? (y/N): ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                return
        
        print(f"Deleting version {version}...")
        result = manager.delete_version(version)
        
        if result:
            print(f"Successfully deleted version {version}")
        else:
            print(f"Failed to delete version {version}", file=sys.stderr)
            sys.exit(1)
            
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage model artifacts with integrity validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pin artifacts for a version
  %(prog)s pin v1.2.3 /path/to/artifacts

  # Validate artifacts
  %(prog)s validate v1.2.3 /path/to/artifacts

  # List all versions
  %(prog)s list-versions

  # Show manifest for a version
  %(prog)s show-manifest v1.2.3

  # Generate integrity report
  %(prog)s report v1.2.3 --output report.json
        """
    )
    
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base directory for artifact storage (default: artifacts)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pin command
    pin_parser = subparsers.add_parser("pin", help="Pin artifacts for a version")
    pin_parser.add_argument("version", help="Version to pin (e.g., v1.2.3)")
    pin_parser.add_argument("artifacts_dir", type=Path, help="Directory containing artifacts")
    pin_parser.add_argument("--output", help="Output file for manifest JSON")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate artifacts for a version")
    validate_parser.add_argument("version", help="Version to validate (e.g., v1.2.3)")
    validate_parser.add_argument("artifacts_dir", type=Path, help="Directory containing artifacts")
    
    # List artifacts command
    list_artifacts_parser = subparsers.add_parser("list-artifacts", help="List artifacts for a version")
    list_artifacts_parser.add_argument("version", help="Version to list artifacts for (e.g., v1.2.3)")
    
    # List versions command
    subparsers.add_parser("list-versions", help="List all available versions")
    
    # Show manifest command
    show_manifest_parser = subparsers.add_parser("show-manifest", help="Show manifest for a version")
    show_manifest_parser.add_argument("version", help="Version to show manifest for (e.g., v1.2.3)")
    
    # Verify integrity command
    verify_parser = subparsers.add_parser("verify", help="Verify integrity of artifacts for a version")
    verify_parser.add_argument("version", help="Version to verify (e.g., v1.2.3)")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate integrity report for a version")
    report_parser.add_argument("version", help="Version to report on (e.g., v1.2.3)")
    report_parser.add_argument("--output", type=Path, help="Output file for report JSON")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a version and its artifacts")
    delete_parser.add_argument("version", help="Version to delete (e.g., v1.2.3)")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize artifact manager
    manager = ArtifactManager(args.artifacts_dir)
    
    # Execute command
    if args.command == "pin":
        pin_artifacts(manager, args.version, args.artifacts_dir, args)
    elif args.command == "validate":
        validate_artifacts(manager, args.version, args.artifacts_dir)
    elif args.command == "list-artifacts":
        list_artifacts(manager, args.version)
    elif args.command == "list-versions":
        list_versions(manager)
    elif args.command == "show-manifest":
        show_manifest(manager, args.version)
    elif args.command == "verify":
        verify_integrity(manager, args.version)
    elif args.command == "report":
        generate_report(manager, args.version, args.output)
    elif args.command == "delete":
        delete_version(manager, args.version, args.force)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
