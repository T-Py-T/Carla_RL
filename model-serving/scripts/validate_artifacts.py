#!/usr/bin/env uv run python
"""
Comprehensive artifact validation CLI tool with detailed reporting.

This script provides a comprehensive command-line interface for validating
model artifacts with detailed reporting, integrity checks, and analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.versioning.artifact_manager import ArtifactManager
from src.versioning.integrity_validator import (
    IntegrityValidationError,
    IntegrityValidator,
)
from src.versioning.semantic_version import VersionError, parse_version


def validate_version_artifacts(
    manager: ArtifactManager,
    validator: IntegrityValidator,
    version: str,
    artifacts_dir: Optional[Path],
    args,
) -> None:
    """Validate artifacts for a specific version."""
    try:
        print(f"Validating artifacts for version {version}...")
        artifacts_dir = _resolve_version_dir(manager, version, artifacts_dir)
        is_valid, report = validator.validate_model_artifacts(
            version, artifacts_dir, args.required_artifacts, args.strict
        )
        _print_validation_result(version, is_valid, report)
        if args.verbose:
            _print_artifact_details(report["artifacts"])
        if args.output:
            _save_json(report, Path(args.output), "Detailed report")
        if not is_valid and args.strict:
            sys.exit(1)

    except IntegrityValidationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def _resolve_version_dir(
    manager: ArtifactManager, version: str, artifacts_dir: Optional[Path]
) -> Path:
    directory = artifacts_dir or manager.versions_dir / version
    if not directory.exists():
        print(f"Error: Version directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    return directory


def _print_validation_result(version: str, is_valid: bool, report: dict) -> None:
    print(f"\nValidation Results for {version}:")
    print(f"  Status: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  Total artifacts: {report['total_artifacts']}")
    print(f"  Valid artifacts: {report['valid_artifacts']}")
    print(f"  Invalid artifacts: {report['invalid_artifacts']}")
    print(f"  Missing artifacts: {report['missing_artifacts']}")
    print(f"  Errors: {len(report['errors'])}")
    _print_messages("Errors", report["errors"])
    _print_messages("Warnings", report["warnings"])


def _print_messages(title: str, messages: list[str]) -> None:
    if not messages:
        return
    print(f"\n{title}:")
    for message in messages:
        print(f"  - {message}")


def _print_artifact_details(artifacts: dict) -> None:
    print("\nArtifact Details:")
    for artifact_path, details in artifacts.items():
        status = details["status"]
        print(f"  {'✓' if status == 'valid' else '✗'} {artifact_path}")
        if status == "invalid":
            print(f"    Expected: {details.get('expected_hash', 'N/A')}")
            print(f"    Actual:   {details.get('actual_hash', 'N/A')}")
        elif status in {"missing", "error"}:
            print(f"    Error: {details.get('error', 'N/A')}")


def _save_json(data: dict, output_file: Path, label: str) -> None:
    with open(output_file, "w") as output:
        json.dump(data, output, indent=2)
    print(f"\n{label} saved to {output_file}")


def validate_all_versions(manager: ArtifactManager, validator: IntegrityValidator, args) -> None:
    """Validate all available versions."""
    versions = manager.list_versions()

    if not versions:
        print("No versions found to validate")
        return

    print(f"Validating {len(versions)} versions...")
    print("=" * 50)

    results = {}
    total_valid = 0
    total_invalid = 0

    for version in versions:
        version_str, is_valid, result = _validate_one_version(manager, validator, version, args)
        results[version_str] = result
        total_valid += int(is_valid)
        total_invalid += int(not is_valid)

    print("=" * 50)
    print(f"Summary: {total_valid} valid, {total_invalid} invalid")

    # Save results if requested
    if args.output:
        _save_json(
            {
                "summary": {
                    "total_versions": len(versions),
                    "valid_versions": total_valid,
                    "invalid_versions": total_invalid,
                },
                "results": results,
            },
            Path(args.output),
            "Results",
        )

    # Exit with error code if any versions failed
    if total_invalid > 0 and args.strict:
        sys.exit(1)


def _validate_one_version(manager, validator, version, args):
    """Validate and print one version for the validate-all command."""
    version_str = str(version)
    artifacts_dir = manager.versions_dir / version_str
    if not artifacts_dir.exists():
        print(f"⚠️  {version_str}: Version directory not found")
        return version_str, False, {"status": "missing", "error": "Directory not found"}
    try:
        is_valid, report = validator.validate_model_artifacts(
            version, artifacts_dir, args.required_artifacts, strict_mode=False
        )
    except Exception as exc:
        print(f"✗ {version_str}: ERROR - {exc}")
        return version_str, False, {"status": "error", "error": str(exc)}

    status = "PASSED" if is_valid else "FAILED"
    print(
        f"{'✓' if is_valid else '✗'} {version_str}: {status} "
        f"({report['valid_artifacts']}/{report['total_artifacts']} artifacts)"
    )
    if not is_valid and args.verbose:
        _print_limited_errors(report["errors"])
    return (
        version_str,
        is_valid,
        {
            "status": "valid" if is_valid else "invalid",
            "report": report,
        },
    )


def _print_limited_errors(errors: list[str]) -> None:
    for error in errors[:3]:
        print(f"    - {error}")
    if len(errors) > 3:
        print(f"    ... and {len(errors) - 3} more errors")


def generate_integrity_report(
    manager: ArtifactManager, validator: IntegrityValidator, version: str, args
) -> None:
    """Generate comprehensive integrity report for a version."""
    try:
        print(f"Generating integrity report for version {version}...")

        # Get version info
        manifest = manager.get_manifest(parse_version(version))
        if not manifest:
            print(f"Error: No manifest found for version {version}", file=sys.stderr)
            sys.exit(1)

        # Get integrity results
        version_dir = manager.versions_dir / version
        if not version_dir.exists():
            print(f"Error: Version directory not found: {version_dir}", file=sys.stderr)
            sys.exit(1)

        integrity_results = manager.verify_integrity(parse_version(version))

        report = _build_integrity_report(
            manager, validator, version, version_dir, manifest, integrity_results
        )
        _print_integrity_summary(version, report)
        if args.verbose:
            _print_integrity_details(report["artifacts"])
        output_file = Path(args.output) if args.output else Path(f"integrity_report_{version}.json")
        _save_json(report, output_file, "Report")

    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        sys.exit(1)


def _build_integrity_report(manager, validator, version, version_dir, manifest, integrity_results):
    """Build the serializable integrity report."""
    valid_count = sum(integrity_results.values())
    report = {
        "version": version,
        "timestamp": validator._get_current_timestamp(),
        "manifest": manifest.to_dict(),
        "integrity_status": "valid" if all(integrity_results.values()) else "invalid",
        "integrity_results": integrity_results,
        "artifacts_summary": {
            "total_artifacts": len(manifest.artifacts),
            "valid_artifacts": valid_count,
            "invalid_artifacts": len(integrity_results) - valid_count,
        },
        "artifacts": {},
    }
    for artifact_path, expected_hash in manifest.artifacts.items():
        report["artifacts"][artifact_path] = _artifact_integrity_info(
            manager,
            version_dir / artifact_path,
            expected_hash,
            integrity_results.get(artifact_path, False),
        )
    return report


def _artifact_integrity_info(manager, full_path, expected_hash, is_valid):
    """Build detailed integrity information for one artifact."""
    exists = full_path.exists()
    info = {
        "path": str(full_path),
        "expected_hash": expected_hash,
        "exists": exists,
        "valid": is_valid,
        "size": full_path.stat().st_size if exists else 0,
    }
    if not exists:
        return info
    try:
        actual_hash = manager.calculate_file_hash(full_path)
        info.update(actual_hash=actual_hash, hash_match=actual_hash == expected_hash)
    except Exception as exc:
        info["hash_error"] = str(exc)
    return info


def _print_integrity_summary(version, report):
    summary = report["artifacts_summary"]
    print(f"\nIntegrity Report for {version}:")
    print(f"  Status: {report['integrity_status'].upper()}")
    print(f"  Total artifacts: {summary['total_artifacts']}")
    print(f"  Valid artifacts: {summary['valid_artifacts']}")
    print(f"  Invalid artifacts: {summary['invalid_artifacts']}")


def _print_integrity_details(artifacts):
    print("\nArtifact Details:")
    for artifact_path, info in artifacts.items():
        print(f"  {'✓' if info['valid'] else '✗'} {artifact_path}")
        print(f"    Size: {info['size']} bytes")
        print(f"    Expected hash: {info['expected_hash']}")
        if "actual_hash" in info:
            print(f"    Actual hash: {info['actual_hash']}")
            print(f"    Hash match: {info['hash_match']}")
        if "hash_error" in info:
            print(f"    Error: {info['hash_error']}")


def compare_versions(manager: ArtifactManager, version1: str, version2: str, args) -> None:
    """Compare two versions."""
    try:
        print(f"Comparing versions {version1} and {version2}...")

        manifest1 = _require_manifest(manager, version1)
        manifest2 = _require_manifest(manager, version2)

        # Compare artifacts
        artifacts1 = set(manifest1.artifacts.keys())
        artifacts2 = set(manifest2.artifacts.keys())

        common_artifacts = artifacts1 & artifacts2
        only_in_v1 = artifacts1 - artifacts2
        only_in_v2 = artifacts2 - artifacts1

        # Find changed artifacts
        changed_artifacts = []
        for artifact in common_artifacts:
            if manifest1.artifacts[artifact] != manifest2.artifacts[artifact]:
                changed_artifacts.append(artifact)

        _print_comparison_summary(
            version1,
            version2,
            artifacts1,
            artifacts2,
            common_artifacts,
            changed_artifacts,
            only_in_v1,
            only_in_v2,
        )
        if args.verbose:
            _print_comparison_details(
                version1, version2, manifest1, manifest2, changed_artifacts, only_in_v1, only_in_v2
            )

        # Save comparison if requested
        if args.output:
            comparison = {
                "version1": version1,
                "version2": version2,
                "artifacts1": list(artifacts1),
                "artifacts2": list(artifacts2),
                "common_artifacts": list(common_artifacts),
                "changed_artifacts": changed_artifacts,
                "only_in_v1": list(only_in_v1),
                "only_in_v2": list(only_in_v2),
                "summary": {
                    "total_v1": len(artifacts1),
                    "total_v2": len(artifacts2),
                    "common": len(common_artifacts),
                    "changed": len(changed_artifacts),
                    "only_v1": len(only_in_v1),
                    "only_v2": len(only_in_v2),
                },
            }

            output_file = Path(args.output)
            with open(output_file, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to {output_file}")

    except VersionError as e:
        print(f"Invalid version format: {e}", file=sys.stderr)
        sys.exit(1)


def _require_manifest(manager, version):
    manifest = manager.get_manifest(parse_version(version))
    if not manifest:
        print(f"Error: No manifest found for version {version}", file=sys.stderr)
        sys.exit(1)
    return manifest


def _print_comparison_summary(
    version1, version2, artifacts1, artifacts2, common, changed, only_v1, only_v2
):
    print("\nVersion Comparison:")
    print(f"  {version1}: {len(artifacts1)} artifacts")
    print(f"  {version2}: {len(artifacts2)} artifacts")
    print(f"  Common: {len(common)} artifacts")
    print(f"  Changed: {len(changed)} artifacts")
    print(f"  Only in {version1}: {len(only_v1)} artifacts")
    print(f"  Only in {version2}: {len(only_v2)} artifacts")


def _print_comparison_details(version1, version2, manifest1, manifest2, changed, only_v1, only_v2):
    if changed:
        print("\nChanged Artifacts:")
        for artifact in changed:
            print(f"  {artifact}")
            print(f"    {version1}: {manifest1.artifacts[artifact]}")
            print(f"    {version2}: {manifest2.artifacts[artifact]}")
    _print_artifact_names(f"Only in {version1}", only_v1)
    _print_artifact_names(f"Only in {version2}", only_v2)


def _print_artifact_names(title, artifacts):
    if not artifacts:
        return
    print(f"\n{title}:")
    for artifact in artifacts:
        print(f"  {artifact}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive artifact validation tool with detailed reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a specific version
  %(prog)s validate v1.2.3 --verbose

  # Validate all versions
  %(prog)s validate-all --output results.json

  # Generate integrity report
  %(prog)s report v1.2.3 --output integrity_report.json

  # Compare two versions
  %(prog)s compare v1.1.0 v1.2.0 --verbose

  # Validate with specific required artifacts
  %(prog)s validate v1.2.3 --required model.pt config.yaml
        """,
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base directory for artifact storage (default: artifacts)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--strict", action="store_true", help="Exit with error code on validation failures"
    )
    parser.add_argument("--output", "-o", help="Output file for reports (JSON format)")
    parser.add_argument("--required", nargs="*", help="List of required artifact files to validate")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate artifacts for a specific version"
    )
    validate_parser.add_argument("version", help="Version to validate (e.g., v1.2.3)")
    validate_parser.add_argument(
        "--artifacts-dir", type=Path, help="Directory containing artifacts to validate"
    )

    # Validate all command
    subparsers.add_parser("validate-all", help="Validate all available versions")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive integrity report")
    report_parser.add_argument("version", help="Version to generate report for (e.g., v1.2.3)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two versions")
    compare_parser.add_argument("version1", help="First version to compare")
    compare_parser.add_argument("version2", help="Second version to compare")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize managers
    artifact_manager = ArtifactManager(args.artifacts_dir)
    integrity_validator = IntegrityValidator(artifact_manager)

    # Execute command
    if args.command == "validate":
        validate_version_artifacts(
            artifact_manager, integrity_validator, args.version, args.artifacts_dir, args
        )
    elif args.command == "validate-all":
        validate_all_versions(artifact_manager, integrity_validator, args)
    elif args.command == "report":
        generate_integrity_report(artifact_manager, integrity_validator, args.version, args)
    elif args.command == "compare":
        compare_versions(artifact_manager, args.version1, args.version2, args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
