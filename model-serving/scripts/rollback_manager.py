#!/usr/bin/env uv run python
"""
CLI tool for managing artifact rollback operations.

This script provides command-line interface for rolling back to previous
model versions with safety checks and validation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from versioning.artifact_manager import ArtifactManager
from versioning.integrity_validator import IntegrityValidator
from versioning.rollback_manager import RollbackManager, RollbackError
from versioning.semantic_version import parse_version, VersionError


def rollback_to_version(
    manager: RollbackManager,
    target_version: str,
    current_version: Optional[str],
    args
) -> None:
    """Rollback to a specific version."""
    try:
        print(f"Rolling back to version {target_version}...")
        
        operation = manager.rollback_to_version(
            target_version=target_version,
            current_version=current_version,
            reason=args.reason,
            create_backup=args.backup,
            validate_target=args.validate
        )
        
        print(f"Rollback completed successfully!")
        print(f"Operation ID: {operation.operation_id}")
        print(f"From version: {operation.from_version}")
        print(f"To version: {operation.to_version}")
        print(f"Status: {operation.status}")
        
        if operation.backup_location:
            print(f"Backup created at: {operation.backup_location}")
        
        if operation.artifacts_affected:
            print(f"Affected artifacts: {', '.join(operation.artifacts_affected)}")
        
        if args.output:
            output_file = Path(args.output)
            with open(output_file, 'w') as f:
                json.dump(operation.to_dict(), f, indent=2)
            print(f"Operation details saved to {output_file}")
            
    except RollbackError as e:
        print(f"Rollback error: {e}", file=sys.stderr)
        if e.current_version:
            print(f"  Current version: {e.current_version}", file=sys.stderr)
        if e.target_version:
            print(f"  Target version: {e.target_version}", file=sys.stderr)
        sys.exit(1)
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def rollback_operation(manager: RollbackManager, operation_id: str, args) -> None:
    """Rollback a previous rollback operation."""
    try:
        print(f"Rolling back operation {operation_id}...")
        
        operation = manager.rollback_operation(
            operation_id=operation_id,
            reason=args.reason
        )
        
        print(f"Rollback reversal completed successfully!")
        print(f"Operation ID: {operation.operation_id}")
        print(f"From version: {operation.from_version}")
        print(f"To version: {operation.to_version}")
        print(f"Status: {operation.status}")
        
    except RollbackError as e:
        print(f"Rollback error: {e}", file=sys.stderr)
        sys.exit(1)


def list_operations(manager: RollbackManager, args) -> None:
    """List rollback operations."""
    operations = manager.list_rollback_operations(args.status)
    
    if not operations:
        print("No rollback operations found")
        return
    
    print(f"Found {len(operations)} rollback operations:")
    print()
    
    for operation in operations:
        print(f"Operation ID: {operation.operation_id}")
        print(f"  From: {operation.from_version} -> To: {operation.to_version}")
        print(f"  Status: {operation.status}")
        print(f"  Reason: {operation.reason}")
        print(f"  Timestamp: {operation.timestamp}")
        if operation.artifacts_affected:
            print(f"  Affected artifacts: {', '.join(operation.artifacts_affected)}")
        if operation.error_message:
            print(f"  Error: {operation.error_message}")
        print()


def show_operation(manager: RollbackManager, operation_id: str) -> None:
    """Show details of a specific rollback operation."""
    operation = manager.get_rollback_operation(operation_id)
    
    if not operation:
        print(f"Rollback operation {operation_id} not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Rollback Operation Details:")
    print(f"  Operation ID: {operation.operation_id}")
    print(f"  From version: {operation.from_version}")
    print(f"  To version: {operation.to_version}")
    print(f"  Status: {operation.status}")
    print(f"  Reason: {operation.reason}")
    print(f"  Timestamp: {operation.timestamp}")
    print(f"  Backup location: {operation.backup_location or 'None'}")
    print(f"  Error message: {operation.error_message or 'None'}")
    print(f"  Affected artifacts: {', '.join(operation.artifacts_affected) or 'None'}")


def list_versions(manager: RollbackManager) -> None:
    """List available versions for rollback."""
    versions = manager.get_available_versions()
    
    if not versions:
        print("No versions available for rollback")
        return
    
    print("Available versions for rollback:")
    for version in versions:
        info = manager.get_version_info(version)
        status = "✓" if info["exists"] and info["integrity_status"] == "valid" else "✗"
        print(f"  {status} {version} ({info['artifacts_count']} artifacts)")


def show_version_info(manager: RollbackManager, version: str) -> None:
    """Show detailed information about a version."""
    try:
        info = manager.get_version_info(version)
        
        if not info["exists"]:
            print(f"Version {version} does not exist")
            return
        
        print(f"Version Information for {version}:")
        print(f"  Exists: {info['exists']}")
        print(f"  Artifacts count: {info['artifacts_count']}")
        print(f"  Integrity status: {info['integrity_status']}")
        print(f"  Version directory: {info['version_dir']}")
        
        if info['integrity_results']:
            print(f"  Integrity details:")
            for artifact, is_valid in info['integrity_results'].items():
                status = "✓" if is_valid else "✗"
                print(f"    {status} {artifact}")
        
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def cleanup_operations(manager: RollbackManager, args) -> None:
    """Clean up old rollback operations."""
    try:
        cleaned_count = manager.cleanup_old_rollbacks(keep_last_n=args.keep)
        print(f"Cleaned up {cleaned_count} old rollback operations")
        print(f"Kept the last {args.keep} operations")
    except Exception as e:
        print(f"Cleanup error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage artifact rollback operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rollback to a specific version
  %(prog)s rollback v1.1.0 --reason "Reverting due to performance issues"

  # Rollback with backup and validation
  %(prog)s rollback v1.0.0 --backup --validate

  # List all rollback operations
  %(prog)s list-operations

  # Show details of a specific operation
  %(prog)s show-operation abc12345

  # List available versions
  %(prog)s list-versions

  # Show version information
  %(prog)s show-version v1.2.0

  # Clean up old operations
  %(prog)s cleanup --keep 10
        """
    )
    
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base directory for artifact storage (default: artifacts)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a specific version")
    rollback_parser.add_argument("target_version", help="Target version to rollback to (e.g., v1.1.0)")
    rollback_parser.add_argument("--current-version", help="Current version (if not provided, will be detected)")
    rollback_parser.add_argument("--reason", default="Manual rollback", help="Reason for rollback")
    rollback_parser.add_argument("--backup", action="store_true", help="Create backup of current state")
    rollback_parser.add_argument("--validate", action="store_true", help="Validate target version integrity")
    rollback_parser.add_argument("--output", help="Output file for operation details JSON")
    
    # Rollback operation command
    rollback_op_parser = subparsers.add_parser("rollback-operation", help="Rollback a previous rollback operation")
    rollback_op_parser.add_argument("operation_id", help="Operation ID to rollback")
    rollback_op_parser.add_argument("--reason", default="Rollback reversal", help="Reason for rollback reversal")
    
    # List operations command
    list_ops_parser = subparsers.add_parser("list-operations", help="List rollback operations")
    list_ops_parser.add_argument("--status", help="Filter by status (pending, in_progress, completed, failed, rolled_back)")
    
    # Show operation command
    show_op_parser = subparsers.add_parser("show-operation", help="Show details of a rollback operation")
    show_op_parser.add_argument("operation_id", help="Operation ID to show")
    
    # List versions command
    subparsers.add_parser("list-versions", help="List available versions for rollback")
    
    # Show version command
    show_version_parser = subparsers.add_parser("show-version", help="Show detailed information about a version")
    show_version_parser.add_argument("version", help="Version to show information for (e.g., v1.2.0)")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old rollback operations")
    cleanup_parser.add_argument("--keep", type=int, default=10, help="Number of recent operations to keep (default: 10)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize managers
    artifact_manager = ArtifactManager(args.artifacts_dir)
    integrity_validator = IntegrityValidator(artifact_manager)
    rollback_manager = RollbackManager(artifact_manager, integrity_validator)
    
    # Execute command
    if args.command == "rollback":
        rollback_to_version(rollback_manager, args.target_version, args.current_version, args)
    elif args.command == "rollback-operation":
        rollback_operation(rollback_manager, args.operation_id, args)
    elif args.command == "list-operations":
        list_operations(rollback_manager, args)
    elif args.command == "show-operation":
        show_operation(rollback_manager, args.operation_id)
    elif args.command == "list-versions":
        list_versions(rollback_manager)
    elif args.command == "show-version":
        show_version_info(rollback_manager, args.version)
    elif args.command == "cleanup":
        cleanup_operations(rollback_manager, args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
