#!/usr/bin/env python3
"""
CLI tool for artifact migration management.

Provides command-line interface for managing artifact migrations,
creating migration plans, and executing version upgrades.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from versioning import ArtifactManager, MigrationManager, MigrationError
from versioning.migration_manager import create_builtin_migration_steps


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_migration_manager(artifacts_dir: str) -> MigrationManager:
    """Set up migration manager with specified directory."""
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        logger.error(f"Artifacts directory does not exist: {artifacts_dir}")
        sys.exit(1)
    
    artifact_manager = ArtifactManager(artifacts_path)
    migration_manager = MigrationManager(artifact_manager)
    
    # Register built-in migration steps
    for step in create_builtin_migration_steps():
        migration_manager.register_migration_step(step)
    
    return migration_manager


def list_migration_plans(migration_manager: MigrationManager, json_output: bool = False) -> None:
    """List all migration plans."""
    plans = migration_manager.list_migration_plans()
    
    if not plans:
        print("No migration plans available")
        return
    
    if json_output:
        output = [plan.to_dict() for plan in plans]
        print(json.dumps(output, indent=2))
    else:
        print("Available Migration Plans:")
        print("-" * 50)
        for plan in plans:
            print(f"Plan ID: {plan.plan_id}")
            print(f"  From: {plan.from_version} -> To: {plan.to_version}")
            print(f"  Steps: {len(plan.steps)}")
            print(f"  Created: {plan.created_at}")
            if plan.description:
                print(f"  Description: {plan.description}")
            print()


def list_migration_results(migration_manager: MigrationManager, 
                          status: Optional[str] = None,
                          json_output: bool = False) -> None:
    """List migration results."""
    results = migration_manager.list_migration_results(status)
    
    if not results:
        print("No migration results found")
        return
    
    if json_output:
        output = [result.to_dict() for result in results]
        print(json.dumps(output, indent=2))
    else:
        print("Migration Results:")
        print("-" * 50)
        for result in results:
            print(f"Migration ID: {result.migration_id}")
            print(f"  Plan: {result.plan_id}")
            print(f"  From: {result.from_version} -> To: {result.to_version}")
            print(f"  Status: {result.status}")
            print(f"  Started: {result.started_at}")
            if result.completed_at:
                print(f"  Completed: {result.completed_at}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            print(f"  Steps Completed: {len(result.steps_completed)}")
            if result.steps_failed:
                print(f"  Steps Failed: {len(result.steps_failed)}")
            print()


def create_migration_plan(migration_manager: MigrationManager,
                         from_version: str,
                         to_version: str,
                         description: str = "",
                         json_output: bool = False) -> None:
    """Create a migration plan."""
    try:
        plan = migration_manager.create_migration_plan(from_version, to_version, description)
        
        if json_output:
            print(json.dumps(plan.to_dict(), indent=2))
        else:
            print(f"Created migration plan: {plan.plan_id}")
            print(f"  From: {plan.from_version} -> To: {plan.to_version}")
            print(f"  Steps: {len(plan.steps)}")
            print(f"  Description: {plan.description}")
    
    except MigrationError as e:
        logger.error(f"Failed to create migration plan: {e}")
        sys.exit(1)


def execute_migration(migration_manager: MigrationManager,
                     plan_id: str,
                     create_backup: bool = True,
                     validate_after: bool = True,
                     json_output: bool = False) -> None:
    """Execute a migration plan."""
    try:
        result = migration_manager.execute_migration(plan_id, create_backup, validate_after)
        
        if json_output:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Migration executed: {result.migration_id}")
            print(f"  Status: {result.status}")
            print(f"  From: {result.from_version} -> To: {result.to_version}")
            print(f"  Started: {result.started_at}")
            if result.completed_at:
                print(f"  Completed: {result.completed_at}")
            if result.backup_location:
                print(f"  Backup: {result.backup_location}")
            print(f"  Steps Completed: {len(result.steps_completed)}")
    
    except MigrationError as e:
        logger.error(f"Migration execution failed: {e}")
        sys.exit(1)


def rollback_migration(migration_manager: MigrationManager,
                      migration_id: str,
                      json_output: bool = False) -> None:
    """Rollback a migration."""
    try:
        result = migration_manager.rollback_migration(migration_id)
        
        if json_output:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Migration rollback executed: {result.migration_id}")
            print(f"  Status: {result.status}")
            print(f"  From: {result.from_version} -> To: {result.to_version}")
            print(f"  Started: {result.started_at}")
            if result.completed_at:
                print(f"  Completed: {result.completed_at}")
            print(f"  Steps Completed: {len(result.steps_completed)}")
    
    except MigrationError as e:
        logger.error(f"Migration rollback failed: {e}")
        sys.exit(1)


def validate_migration_plan(migration_manager: MigrationManager,
                          plan_id: str) -> None:
    """Validate a migration plan."""
    is_valid = migration_manager.validate_migration_plan(plan_id)
    
    if is_valid:
        print(f"Migration plan {plan_id} is valid")
    else:
        print(f"Migration plan {plan_id} is invalid")
        sys.exit(1)


def get_migration_plan(migration_manager: MigrationManager,
                      plan_id: str,
                      json_output: bool = False) -> None:
    """Get migration plan details."""
    plan = migration_manager.get_migration_plan(plan_id)
    
    if not plan:
        print(f"Migration plan not found: {plan_id}")
        sys.exit(1)
    
    if json_output:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(f"Migration Plan: {plan.plan_id}")
        print(f"  From: {plan.from_version} -> To: {plan.to_version}")
        print(f"  Created: {plan.created_at}")
        print(f"  Description: {plan.description}")
        print(f"  Steps ({len(plan.steps)}):")
        for i, step in enumerate(plan.steps, 1):
            print(f"    {i}. {step.name} ({step.step_id})")
            print(f"       Description: {step.description}")
            if step.dependencies:
                print(f"       Dependencies: {', '.join(step.dependencies)}")
            if step.required_artifacts:
                print(f"       Required Artifacts: {', '.join(step.required_artifacts)}")
            if step.created_artifacts:
                print(f"       Creates: {', '.join(step.created_artifacts)}")


def get_migration_result(migration_manager: MigrationManager,
                        migration_id: str,
                        json_output: bool = False) -> None:
    """Get migration result details."""
    result = migration_manager.get_migration_result(migration_id)
    
    if not result:
        print(f"Migration result not found: {migration_id}")
        sys.exit(1)
    
    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Migration Result: {result.migration_id}")
        print(f"  Plan: {result.plan_id}")
        print(f"  From: {result.from_version} -> To: {result.to_version}")
        print(f"  Status: {result.status}")
        print(f"  Started: {result.started_at}")
        if result.completed_at:
            print(f"  Completed: {result.completed_at}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        if result.backup_location:
            print(f"  Backup: {result.backup_location}")
        if result.rollback_location:
            print(f"  Rollback: {result.rollback_location}")
        print(f"  Steps Completed: {len(result.steps_completed)}")
        if result.steps_completed:
            print(f"    {', '.join(result.steps_completed)}")
        if result.steps_failed:
            print(f"  Steps Failed: {len(result.steps_failed)}")
            print(f"    {', '.join(result.steps_failed)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Artifact migration management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all migration plans
  python migration_manager.py --artifacts-dir /path/to/artifacts list-plans
  
  # Create a migration plan
  python migration_manager.py --artifacts-dir /path/to/artifacts create-plan --from v1.0.0 --to v2.0.0
  
  # Execute a migration
  python migration_manager.py --artifacts-dir /path/to/artifacts execute --plan-id plan_v1.0.0_v2.0.0_20240101_120000
  
  # Rollback a migration
  python migration_manager.py --artifacts-dir /path/to/artifacts rollback --migration-id migration_abc12345
  
  # List migration results
  python migration_manager.py --artifacts-dir /path/to/artifacts list-results --status completed
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
    
    # List plans command
    list_plans_parser = subparsers.add_parser("list-plans", help="List migration plans")
    
    # List results command
    list_results_parser = subparsers.add_parser("list-results", help="List migration results")
    list_results_parser.add_argument(
        "--status",
        choices=["pending", "in_progress", "completed", "failed", "rolled_back"],
        help="Filter by status"
    )
    
    # Create plan command
    create_plan_parser = subparsers.add_parser("create-plan", help="Create migration plan")
    create_plan_parser.add_argument(
        "--from",
        dest="from_version",
        required=True,
        help="Source version"
    )
    create_plan_parser.add_argument(
        "--to",
        dest="to_version",
        required=True,
        help="Target version"
    )
    create_plan_parser.add_argument(
        "--description",
        default="",
        help="Plan description"
    )
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute migration")
    execute_parser.add_argument(
        "--plan-id",
        required=True,
        help="Migration plan ID"
    )
    execute_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation"
    )
    execute_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after migration"
    )
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migration")
    rollback_parser.add_argument(
        "--migration-id",
        required=True,
        help="Migration ID to rollback"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate migration plan")
    validate_parser.add_argument(
        "--plan-id",
        required=True,
        help="Migration plan ID"
    )
    
    # Get plan command
    get_plan_parser = subparsers.add_parser("get-plan", help="Get migration plan details")
    get_plan_parser.add_argument(
        "--plan-id",
        required=True,
        help="Migration plan ID"
    )
    
    # Get result command
    get_result_parser = subparsers.add_parser("get-result", help="Get migration result details")
    get_result_parser.add_argument(
        "--migration-id",
        required=True,
        help="Migration ID"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up migration manager
    try:
        migration_manager = setup_migration_manager(args.artifacts_dir)
    except Exception as e:
        logger.error(f"Failed to set up migration manager: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "list-plans":
            list_migration_plans(migration_manager, args.json)
        
        elif args.command == "list-results":
            list_migration_results(migration_manager, args.status, args.json)
        
        elif args.command == "create-plan":
            create_migration_plan(
                migration_manager,
                args.from_version,
                args.to_version,
                args.description,
                args.json
            )
        
        elif args.command == "execute":
            execute_migration(
                migration_manager,
                args.plan_id,
                not args.no_backup,
                not args.no_validate,
                args.json
            )
        
        elif args.command == "rollback":
            rollback_migration(migration_manager, args.migration_id, args.json)
        
        elif args.command == "validate":
            validate_migration_plan(migration_manager, args.plan_id)
        
        elif args.command == "get-plan":
            get_migration_plan(migration_manager, args.plan_id, args.json)
        
        elif args.command == "get-result":
            get_migration_result(migration_manager, args.migration_id, args.json)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
