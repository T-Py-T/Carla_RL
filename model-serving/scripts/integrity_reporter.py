#!/usr/bin/env python3
"""
CLI tool for generating comprehensive artifact integrity reports.

Provides detailed validation results, integrity analysis, and reporting
capabilities for the Policy-as-a-Service artifact management system.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.versioning.artifact_manager import ArtifactManager
from src.versioning.semantic_version import parse_version


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


def generate_integrity_report(artifact_manager: ArtifactManager, 
                            version: Optional[str] = None,
                            output_format: str = "text",
                            output_file: Optional[str] = None,
                            include_metadata: bool = True,
                            include_artifacts: bool = True) -> Dict[str, Any]:
    """Generate comprehensive integrity report."""
    
    if version:
        # Generate report for specific version
        version_obj = parse_version(version)
        report = artifact_manager.get_integrity_report(version_obj)
        
        # Enhance report with additional details
        enhanced_report = enhance_integrity_report(artifact_manager, report, include_metadata, include_artifacts)
        
        return {
            "report_type": "single_version",
            "version": str(version_obj),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report": enhanced_report
        }
    else:
        # Generate report for all versions
        all_versions = artifact_manager.list_versions()
        reports = {}
        
        for version_obj in all_versions:
            report = artifact_manager.get_integrity_report(version_obj)
            enhanced_report = enhance_integrity_report(artifact_manager, report, include_metadata, include_artifacts)
            reports[str(version_obj)] = enhanced_report
        
        # Generate summary
        summary = generate_summary_report(reports)
        
        return {
            "report_type": "all_versions",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_versions": len(all_versions),
            "summary": summary,
            "reports": reports
        }


def enhance_integrity_report(artifact_manager: ArtifactManager, 
                           report: Dict[str, Any], 
                           include_metadata: bool = True,
                           include_artifacts: bool = True) -> Dict[str, Any]:
    """Enhance integrity report with additional details."""
    
    enhanced = report.copy()
    
    if include_metadata and "manifest" in report:
        manifest = report["manifest"]
        enhanced["metadata"] = {
            "created_at": manifest.get("created_at", "Unknown"),
            "model_type": manifest.get("model_type", "Unknown"),
            "description": manifest.get("description", "No description"),
            "dependencies": manifest.get("dependencies", []),
            "custom_metadata": manifest.get("metadata", {})
        }
    
    if include_artifacts and "artifacts" in report:
        artifacts = report["artifacts"]
        enhanced["artifact_details"] = {}
        
        for artifact_path, is_valid in artifacts.items():
            artifact_info = {
                "valid": is_valid,
                "path": artifact_path,
                "size": "Unknown",
                "last_modified": "Unknown"
            }
            
            # Try to get file size and modification time
            try:
                version_dir = artifact_manager.versions_dir / report["version"]
                artifact_file = version_dir / artifact_path
                if artifact_file.exists():
                    stat = artifact_file.stat()
                    artifact_info["size"] = stat.st_size
                    artifact_info["last_modified"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                pass
            
            enhanced["artifact_details"][artifact_path] = artifact_info
    
    # Add integrity score
    if "artifacts" in report:
        total_artifacts = len(report["artifacts"])
        valid_artifacts = sum(report["artifacts"].values())
        enhanced["integrity_score"] = (valid_artifacts / total_artifacts * 100) if total_artifacts > 0 else 0
    
    return enhanced


def generate_summary_report(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary report for all versions."""
    
    total_versions = len(reports)
    valid_versions = sum(1 for report in reports.values() if report.get("status") == "valid")
    invalid_versions = total_versions - valid_versions
    
    total_artifacts = 0
    valid_artifacts = 0
    
    for report in reports.values():
        if "artifacts" in report:
            total_artifacts += len(report["artifacts"])
            valid_artifacts += sum(report["artifacts"].values())
    
    overall_integrity = (valid_artifacts / total_artifacts * 100) if total_artifacts > 0 else 0
    
    # Find versions with issues
    problematic_versions = []
    for version, report in reports.items():
        if report.get("status") != "valid":
            problematic_versions.append({
                "version": version,
                "status": report.get("status", "unknown"),
                "issues": report.get("invalid_artifacts", 0)
            })
    
    return {
        "total_versions": total_versions,
        "valid_versions": valid_versions,
        "invalid_versions": invalid_versions,
        "total_artifacts": total_artifacts,
        "valid_artifacts": valid_artifacts,
        "overall_integrity_percentage": round(overall_integrity, 2),
        "problematic_versions": problematic_versions
    }


def format_text_report(report_data: Dict[str, Any]) -> str:
    """Format report data as human-readable text."""
    
    output = []
    output.append("=" * 80)
    output.append("ARTIFACT INTEGRITY REPORT")
    output.append("=" * 80)
    output.append(f"Generated: {report_data['generated_at']}")
    output.append("")
    
    if report_data["report_type"] == "single_version":
        version = report_data["version"]
        report = report_data["report"]
        
        output.append(f"VERSION: {version}")
        output.append("-" * 40)
        output.append(f"Status: {report['status'].upper()}")
        output.append(f"Integrity Score: {report.get('integrity_score', 0):.1f}%")
        output.append(f"Total Artifacts: {report.get('total_artifacts', 0)}")
        output.append(f"Valid Artifacts: {report.get('valid_artifacts', 0)}")
        output.append(f"Invalid Artifacts: {report.get('invalid_artifacts', 0)}")
        output.append("")
        
        if "metadata" in report:
            metadata = report["metadata"]
            output.append("METADATA:")
            output.append(f"  Created: {metadata.get('created_at', 'Unknown')}")
            output.append(f"  Model Type: {metadata.get('model_type', 'Unknown')}")
            output.append(f"  Description: {metadata.get('description', 'No description')}")
            if metadata.get('dependencies'):
                output.append(f"  Dependencies: {', '.join(metadata['dependencies'])}")
            output.append("")
        
        if "artifact_details" in report:
            output.append("ARTIFACT DETAILS:")
            for artifact_path, details in report["artifact_details"].items():
                status = "✓" if details["valid"] else "✗"
                size = details.get("size", "Unknown")
                if isinstance(size, int):
                    size = f"{size:,} bytes"
                output.append(f"  {status} {artifact_path} ({size})")
            output.append("")
    
    else:  # all_versions
        summary = report_data["summary"]
        output.append("SUMMARY:")
        output.append(f"  Total Versions: {summary['total_versions']}")
        output.append(f"  Valid Versions: {summary['valid_versions']}")
        output.append(f"  Invalid Versions: {summary['invalid_versions']}")
        output.append(f"  Total Artifacts: {summary['total_artifacts']}")
        output.append(f"  Valid Artifacts: {summary['valid_artifacts']}")
        output.append(f"  Overall Integrity: {summary['overall_integrity_percentage']}%")
        output.append("")
        
        if summary["problematic_versions"]:
            output.append("PROBLEMATIC VERSIONS:")
            for version_info in summary["problematic_versions"]:
                output.append(f"  {version_info['version']}: {version_info['status']} ({version_info['issues']} issues)")
            output.append("")
        
        # List all versions with their status
        output.append("VERSION STATUS:")
        for version, report in report_data["reports"].items():
            status = report.get("status", "unknown").upper()
            integrity_score = report.get("integrity_score", 0)
            output.append(f"  {version}: {status} ({integrity_score:.1f}%)")
    
    output.append("=" * 80)
    return "\n".join(output)


def format_json_report(report_data: Dict[str, Any]) -> str:
    """Format report data as JSON."""
    return json.dumps(report_data, indent=2, default=str)


def format_csv_report(report_data: Dict[str, Any]) -> str:
    """Format report data as CSV."""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    if report_data["report_type"] == "single_version":
        report = report_data["report"]
        writer.writerow(["Version", "Status", "Integrity_Score", "Total_Artifacts", "Valid_Artifacts", "Invalid_Artifacts"])
        writer.writerow([
            report_data["version"],
            report.get("status", "unknown"),
            report.get("integrity_score", 0),
            report.get("total_artifacts", 0),
            report.get("valid_artifacts", 0),
            report.get("invalid_artifacts", 0)
        ])
    else:  # all_versions
        writer.writerow(["Version", "Status", "Integrity_Score", "Total_Artifacts", "Valid_Artifacts", "Invalid_Artifacts"])
        for version, report in report_data["reports"].items():
            writer.writerow([
                version,
                report.get("status", "unknown"),
                report.get("integrity_score", 0),
                report.get("total_artifacts", 0),
                report.get("valid_artifacts", 0),
                report.get("invalid_artifacts", 0)
            ])
    
    return output.getvalue()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive artifact integrity reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for specific version
  python integrity_reporter.py --artifacts-dir /path/to/artifacts --version v1.2.3
  
  # Generate report for all versions
  python integrity_reporter.py --artifacts-dir /path/to/artifacts
  
  # Generate JSON report and save to file
  python integrity_reporter.py --artifacts-dir /path/to/artifacts --format json --output report.json
  
  # Generate CSV report without metadata
  python integrity_reporter.py --artifacts-dir /path/to/artifacts --format csv --no-metadata
        """
    )
    
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Path to artifacts directory"
    )
    
    parser.add_argument(
        "--version",
        help="Specific version to report on (if not provided, reports on all versions)"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file (if not provided, prints to stdout)"
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude metadata from report"
    )
    
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Exclude artifact details from report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up artifact manager
    try:
        artifact_manager = setup_artifact_manager(args.artifacts_dir)
    except Exception as e:
        logger.error(f"Failed to set up artifact manager: {e}")
        sys.exit(1)
    
    # Generate report
    try:
        report_data = generate_integrity_report(
            artifact_manager,
            args.version,
            args.format,
            args.output,
            not args.no_metadata,
            not args.no_artifacts
        )
        
        # Format report
        if args.format == "text":
            formatted_report = format_text_report(report_data)
        elif args.format == "json":
            formatted_report = format_json_report(report_data)
        elif args.format == "csv":
            formatted_report = format_csv_report(report_data)
        else:
            raise ValueError(f"Unknown format: {args.format}")
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_report)
            print(f"Report saved to {args.output}")
        else:
            print(formatted_report)
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
