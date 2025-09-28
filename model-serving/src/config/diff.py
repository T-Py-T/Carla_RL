"""
Configuration diff tools for deployment validation.

Provides tools for comparing configurations and generating
detailed diff reports for deployment validation.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from deepdiff import DeepDiff
import yaml

from .settings import BaseConfig, AppConfig


class DiffType(str, Enum):
    """Types of configuration differences."""
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass
class DiffItem:
    """Individual configuration difference item."""
    path: str
    diff_type: DiffType
    old_value: Any = None
    new_value: Any = None
    change_description: str = ""
    severity: str = "info"  # info, warning, error
    category: str = "general"  # server, model, logging, etc.


@dataclass
class DiffResult:
    """Complete configuration diff result."""
    has_changes: bool
    total_changes: int
    items: List[DiffItem] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, List[DiffItem]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Categorize differences and create summary."""
        if not self.summary:
            self.summary = {
                "added": 0,
                "removed": 0,
                "changed": 0,
                "unchanged": 0
            }
        
        # Count by type
        for item in self.items:
            self.summary[item.diff_type.value] += 1
            
            # Categorize by category
            if item.category not in self.categories:
                self.categories[item.category] = []
            self.categories[item.category].append(item)
        
        self.has_changes = self.summary["added"] + self.summary["removed"] + self.summary["changed"] > 0


class ConfigDiff:
    """Configuration difference calculator."""
    
    def __init__(self):
        """Initialize configuration diff calculator."""
        self._ignore_paths: Set[str] = set()
        self._custom_comparators: Dict[str, callable] = {}
    
    def compare_configs(
        self,
        config1: Union[BaseConfig, Dict[str, Any]],
        config2: Union[BaseConfig, Dict[str, Any]],
        ignore_paths: Optional[List[str]] = None
    ) -> DiffResult:
        """
        Compare two configurations.
        
        Args:
            config1: First configuration (old)
            config2: Second configuration (new)
            ignore_paths: Paths to ignore in comparison
            
        Returns:
            Diff result with all differences
        """
        # Convert to dictionaries if needed
        if isinstance(config1, BaseConfig):
            dict1 = config1.model_dump()
        else:
            dict1 = config1
        
        if isinstance(config2, BaseConfig):
            dict2 = config2.model_dump()
        else:
            dict2 = config2
        
        # Use DeepDiff for comparison
        diff = DeepDiff(
            dict1,
            dict2,
            ignore_order=True,
            exclude_paths=ignore_paths or self._ignore_paths
        )
        
        # Convert DeepDiff result to our format
        items = self._convert_deepdiff_to_items(diff, dict1, dict2)
        
        return DiffResult(
            has_changes=len(items) > 0,
            total_changes=len(items),
            items=items
        )
    
    def _convert_deepdiff_to_items(
        self,
        diff: DeepDiff,
        old_dict: Dict[str, Any],
        new_dict: Dict[str, Any]
    ) -> List[DiffItem]:
        """Convert DeepDiff result to DiffItem list."""
        items = []
        
        # Handle added items
        for path in diff.get("dictionary_item_added", []):
            items.append(DiffItem(
                path=path,
                diff_type=DiffType.ADDED,
                new_value=self._get_nested_value(new_dict, path),
                change_description=f"Added new configuration: {path}",
                severity="info",
                category=self._get_category_from_path(path)
            ))
        
        # Handle removed items
        for path in diff.get("dictionary_item_removed", []):
            items.append(DiffItem(
                path=path,
                diff_type=DiffType.REMOVED,
                old_value=self._get_nested_value(old_dict, path),
                change_description=f"Removed configuration: {path}",
                severity="warning",
                category=self._get_category_from_path(path)
            ))
        
        # Handle changed items
        for path, changes in diff.get("values_changed", {}).items():
            old_value = changes.get("old_value")
            new_value = changes.get("new_value")
            
            severity = self._get_change_severity(path, old_value, new_value)
            
            items.append(DiffItem(
                path=path,
                diff_type=DiffType.CHANGED,
                old_value=old_value,
                new_value=new_value,
                change_description=f"Changed {path}: {old_value} → {new_value}",
                severity=severity,
                category=self._get_category_from_path(path)
            ))
        
        # Handle type changes
        for path, changes in diff.get("type_changes", {}).items():
            old_value = changes.get("old_value")
            new_value = changes.get("new_value")
            old_type = changes.get("old_type")
            new_type = changes.get("new_type")
            
            items.append(DiffItem(
                path=path,
                diff_type=DiffType.CHANGED,
                old_value=old_value,
                new_value=new_value,
                change_description=f"Type changed {path}: {old_type} → {new_type}",
                severity="error",
                category=self._get_category_from_path(path)
            ))
        
        return items
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split(".")
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _get_category_from_path(self, path: str) -> str:
        """Get category from configuration path."""
        if "." in path:
            return path.split(".")[0]
        return "general"
    
    def _get_change_severity(self, path: str, old_value: Any, new_value: Any) -> str:
        """Determine severity of configuration change."""
        # Critical paths that should be errors
        critical_paths = [
            "security.enabled",
            "server.port",
            "model.model_path",
            "database.backend"
        ]
        
        if any(critical in path for critical in critical_paths):
            return "error"
        
        # Security-related changes are warnings
        if "security" in path or "password" in path.lower() or "key" in path.lower():
            return "warning"
        
        # Server configuration changes are warnings
        if "server" in path:
            return "warning"
        
        # Model configuration changes are warnings
        if "model" in path:
            return "warning"
        
        return "info"
    
    def add_ignore_path(self, path: str) -> None:
        """
        Add path to ignore list.
        
        Args:
            path: Path pattern to ignore
        """
        self._ignore_paths.add(path)
    
    def add_custom_comparator(self, path: str, comparator: callable) -> None:
        """
        Add custom comparator for specific path.
        
        Args:
            path: Path pattern
            comparator: Comparison function
        """
        self._custom_comparators[path] = comparator


def compare_configs(
    config1: Union[BaseConfig, Dict[str, Any]],
    config2: Union[BaseConfig, Dict[str, Any]],
    ignore_paths: Optional[List[str]] = None
) -> DiffResult:
    """
    Compare two configurations.
    
    Args:
        config1: First configuration (old)
        config2: Second configuration (new)
        ignore_paths: Paths to ignore in comparison
        
    Returns:
        Diff result
    """
    diff_calculator = ConfigDiff()
    return diff_calculator.compare_configs(config1, config2, ignore_paths)


def compare_config_files(
    file1: Union[str, Path],
    file2: Union[str, Path],
    format: str = "auto"
) -> DiffResult:
    """
    Compare two configuration files.
    
    Args:
        file1: First configuration file
        file2: Second configuration file
        format: File format (json, yaml, auto)
        
    Returns:
        Diff result
    """
    # Load configurations from files
    config1 = _load_config_from_file(file1, format)
    config2 = _load_config_from_file(file2, format)
    
    return compare_configs(config1, config2)


def _load_config_from_file(file_path: Union[str, Path], format: str = "auto") -> Dict[str, Any]:
    """Load configuration from file."""
    file_path = Path(file_path)
    
    if format == "auto":
        format = file_path.suffix.lower().lstrip(".")
    
    with open(file_path, "r", encoding="utf-8") as f:
        if format in ["json"]:
            return json.load(f)
        elif format in ["yaml", "yml"]:
            return yaml.safe_load(f) or {}
        else:
            raise ValueError(f"Unsupported format: {format}")


def format_diff_result(result: DiffResult, format: str = "text") -> str:
    """
    Format diff result as string.
    
    Args:
        result: Diff result to format
        format: Output format (text, json, yaml)
        
    Returns:
        Formatted diff result
    """
    if format == "text":
        return _format_diff_text(result)
    elif format == "json":
        return json.dumps(_diff_result_to_dict(result), indent=2)
    elif format == "yaml":
        return yaml.dump(_diff_result_to_dict(result), default_flow_style=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _format_diff_text(result: DiffResult) -> str:
    """Format diff result as text."""
    lines = []
    
    # Header
    lines.append("Configuration Diff Report")
    lines.append("=" * 50)
    lines.append(f"Has Changes: {result.has_changes}")
    lines.append(f"Total Changes: {result.total_changes}")
    lines.append("")
    
    # Summary
    lines.append("Summary:")
    lines.append("-" * 20)
    for change_type, count in result.summary.items():
        if count > 0:
            lines.append(f"  {change_type.title()}: {count}")
    lines.append("")
    
    # Changes by category
    for category, items in result.categories.items():
        if items:
            lines.append(f"{category.title()} Changes:")
            lines.append("-" * 30)
            
            for item in items:
                severity_icon = {
                    "error": "❌",
                    "warning": "⚠️",
                    "info": "ℹ️"
                }.get(item.severity, "ℹ️")
                
                lines.append(f"  {severity_icon} {item.path}")
                lines.append(f"    Type: {item.diff_type.value}")
                lines.append(f"    Description: {item.change_description}")
                
                if item.old_value is not None:
                    lines.append(f"    Old Value: {item.old_value}")
                if item.new_value is not None:
                    lines.append(f"    New Value: {item.new_value}")
                
                lines.append("")
    
    return "\n".join(lines)


def _diff_result_to_dict(result: DiffResult) -> Dict[str, Any]:
    """Convert diff result to dictionary."""
    return {
        "has_changes": result.has_changes,
        "total_changes": result.total_changes,
        "summary": result.summary,
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
            for item in result.items
        ],
        "categories": {
            category: [
                {
                    "path": item.path,
                    "diff_type": item.diff_type.value,
                    "old_value": item.old_value,
                    "new_value": item.new_value,
                    "change_description": item.change_description,
                    "severity": item.severity
                }
                for item in items
            ]
            for category, items in result.categories.items()
        }
    }


def validate_deployment_config(
    current_config: AppConfig,
    target_config: AppConfig,
    deployment_rules: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate configuration for deployment.
    
    Args:
        current_config: Current configuration
        target_config: Target configuration
        deployment_rules: Optional deployment validation rules
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    diff_result = compare_configs(current_config, target_config)
    errors = []
    
    # Check for critical changes
    critical_changes = [
        item for item in diff_result.items
        if item.severity == "error" and item.diff_type != DiffType.UNCHANGED
    ]
    
    if critical_changes:
        errors.append("Critical configuration changes detected:")
        for item in critical_changes:
            errors.append(f"  - {item.path}: {item.change_description}")
    
    # Check for security changes
    security_changes = [
        item for item in diff_result.items
        if "security" in item.path and item.diff_type != DiffType.UNCHANGED
    ]
    
    if security_changes:
        errors.append("Security configuration changes detected:")
        for item in security_changes:
            errors.append(f"  - {item.path}: {item.change_description}")
    
    # Apply custom deployment rules
    if deployment_rules:
        for rule_name, rule_config in deployment_rules.items():
            if not _apply_deployment_rule(rule_name, rule_config, diff_result):
                errors.append(f"Deployment rule failed: {rule_name}")
    
    return len(errors) == 0, errors


def _apply_deployment_rule(rule_name: str, rule_config: Dict[str, Any], diff_result: DiffResult) -> bool:
    """Apply deployment validation rule."""
    if rule_name == "no_critical_changes":
        return len([item for item in diff_result.items if item.severity == "error"]) == 0
    
    elif rule_name == "no_security_changes":
        return len([item for item in diff_result.items if "security" in item.path]) == 0
    
    elif rule_name == "max_changes":
        max_changes = rule_config.get("max_changes", 10)
        return diff_result.total_changes <= max_changes
    
    elif rule_name == "allowed_categories":
        allowed = set(rule_config.get("categories", []))
        changed_categories = set(result.categories.keys())
        return changed_categories.issubset(allowed)
    
    return True
