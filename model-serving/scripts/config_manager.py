#!/usr/bin/env python3
"""
Configuration management CLI tool.

Provides command-line interface for managing configurations, validation,
hot-reloading, and schema generation.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    AppConfig, load_config, ConfigValidator, TemplateEngine,
    compare_configs, generate_schema_docs, SchemaFormat
)
from config.settings import Environment


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Configuration management tool for model serving",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load and validate configuration")
    load_parser.add_argument("--config", "-c", help="Configuration file path")
    load_parser.add_argument("--env-prefix", default="", help="Environment variable prefix")
    load_parser.add_argument("--validate", action="store_true", help="Validate configuration")
    load_parser.add_argument("--format", choices=["json", "yaml"], default="yaml", help="Output format")
    load_parser.add_argument("--output", "-o", help="Output file path")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate configuration from template")
    generate_parser.add_argument("--template", "-t", required=True, help="Template name")
    generate_parser.add_argument("--env", "-e", required=True, help="Target environment")
    generate_parser.add_argument("--output", "-o", required=True, help="Output file path")
    generate_parser.add_argument("--variables", help="Template variables (JSON string)")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--config", "-c", required=True, help="Configuration file path")
    validate_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", help="Output format")
    
    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two configurations")
    diff_parser.add_argument("config1", help="First configuration file")
    diff_parser.add_argument("config2", help="Second configuration file")
    diff_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", help="Output format")
    diff_parser.add_argument("--output", "-o", help="Output file path")
    
    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Generate configuration schema")
    schema_parser.add_argument("--format", choices=["json", "yaml", "markdown", "html", "rst"], 
                              default="markdown", help="Output format")
    schema_parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "load":
            return handle_load_command(args)
        elif args.command == "generate":
            return handle_generate_command(args)
        elif args.command == "validate":
            return handle_validate_command(args)
        elif args.command == "diff":
            return handle_diff_command(args)
        elif args.command == "schema":
            return handle_schema_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_load_command(args) -> int:
    """Handle load command."""
    # Load configuration
    config = load_config(
        config_file=args.config,
        env_prefix=args.env_prefix
    )
    
    # Validate if requested
    if args.validate:
        validator = ConfigValidator()
        result = validator.validate(config)
        
        if not result.is_valid:
            print("Configuration validation failed:", file=sys.stderr)
            for issue in result.errors:
                print(f"  ERROR: {issue.field}: {issue.message}", file=sys.stderr)
            return 1
        else:
            print("Configuration is valid")
    
    # Output configuration
    config_dict = config.model_dump()
    
    if args.format == "json":
        output = json.dumps(config_dict, indent=2, default=str)
    else:  # yaml
        import yaml
        output = yaml.dump(config_dict, default_flow_style=False, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Configuration saved to {args.output}")
    else:
        print(output)
    
    return 0


def handle_generate_command(args) -> int:
    """Handle generate command."""
    # Parse environment
    try:
        environment = Environment(args.env)
    except ValueError:
        print(f"Invalid environment: {args.env}", file=sys.stderr)
        return 1
    
    # Parse variables if provided
    variables = {}
    if args.variables:
        try:
            variables = json.loads(args.variables)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON for variables: {e}", file=sys.stderr)
            return 1
    
    # Generate configuration
    engine = TemplateEngine()
    engine.generate_config(
        template_name=args.template,
        environment=environment,
        context=variables,
        output_file=Path(args.output)
    )
    
    print(f"Configuration generated: {args.output}")
    return 0


def handle_validate_command(args) -> int:
    """Handle validate command."""
    # Load configuration
    config = load_config(config_file=args.config)
    
    # Validate
    validator = ConfigValidator()
    result = validator.validate(config)
    
    # Format output
    if args.format == "json":
        output = json.dumps({
            "valid": result.is_valid,
            "summary": result.summary,
            "issues": [
                {
                    "field": issue.field,
                    "message": issue.message,
                    "severity": issue.severity.value,
                    "value": issue.value,
                    "expected": issue.expected
                }
                for issue in result.issues
            ]
        }, indent=2)
    elif args.format == "yaml":
        import yaml
        output = yaml.dump({
            "valid": result.is_valid,
            "summary": result.summary,
            "issues": [
                {
                    "field": issue.field,
                    "message": issue.message,
                    "severity": issue.severity.value,
                    "value": issue.value,
                    "expected": issue.expected
                }
                for issue in result.issues
            ]
        }, default_flow_style=False, indent=2)
    else:  # text
        from config.validation import format_validation_result
        output = format_validation_result(result)
    
    print(output)
    return 0 if result.is_valid else 1


def handle_diff_command(args) -> int:
    """Handle diff command."""
    # Load configurations
    config1 = load_config(config_file=args.config1)
    config2 = load_config(config_file=args.config2)
    
    # Compare
    diff_result = compare_configs(config1, config2)
    
    # Format output
    if args.format == "json":
        output = json.dumps({
            "has_changes": diff_result.has_changes,
            "total_changes": diff_result.total_changes,
            "summary": diff_result.summary,
            "items": [
                {
                    "path": item.path,
                    "diff_type": item.diff_type.value,
                    "old_value": item.old_value,
                    "new_value": item.new_value,
                    "change_description": item.change_description,
                    "severity": item.severity,
                    "category": item.category
                }
                for item in diff_result.items
            ]
        }, indent=2)
    elif args.format == "yaml":
        import yaml
        output = yaml.dump({
            "has_changes": diff_result.has_changes,
            "total_changes": diff_result.total_changes,
            "summary": diff_result.summary,
            "items": [
                {
                    "path": item.path,
                    "diff_type": item.diff_type.value,
                    "old_value": item.old_value,
                    "new_value": item.new_value,
                    "change_description": item.change_description,
                    "severity": item.severity,
                    "category": item.category
                }
                for item in diff_result.items
            ]
        }, default_flow_style=False, indent=2)
    else:  # text
        from config.diff import format_diff_result
        output = format_diff_result(diff_result)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Diff saved to {args.output}")
    else:
        print(output)
    
    return 0


def handle_schema_command(args) -> int:
    """Handle schema command."""
    # Generate schema
    output = generate_schema_docs(
        config_class=AppConfig,
        format=SchemaFormat(args.format),
        output_file=Path(args.output) if args.output else None
    )
    
    if not args.output:
        print(output)
    else:
        print(f"Schema saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())