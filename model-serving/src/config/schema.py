"""
Configuration schema documentation generator.

Provides tools for generating configuration schema documentation
in various formats (JSON, YAML, Markdown, HTML).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .settings import AppConfig, Environment


class SchemaFormat(str, Enum):
    """Schema output formats."""
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"


@dataclass
class FieldInfo:
    """Information about a configuration field."""
    name: str
    type: str
    description: str = ""
    default: Any = None
    required: bool = False
    enum_values: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    examples: List[Any] = field(default_factory=list)
    deprecated: bool = False
    deprecation_message: str = ""


@dataclass
class SchemaInfo:
    """Complete schema information."""
    title: str
    description: str
    version: str
    fields: List[FieldInfo] = field(default_factory=list)
    sections: Dict[str, List[FieldInfo]] = field(default_factory=dict)
    examples: Dict[str, Any] = field(default_factory=dict)


class SchemaGenerator:
    """Configuration schema generator."""
    
    def __init__(self):
        """Initialize schema generator."""
        self._field_mappings = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object"
        }
    
    def generate_schema(self, config_class: type = AppConfig) -> SchemaInfo:
        """
        Generate schema for configuration class.
        
        Args:
            config_class: Configuration class to analyze
            
        Returns:
            Schema information
        """
        schema_info = SchemaInfo(
            title="Model Serving Configuration Schema",
            description="Complete configuration schema for the model serving system",
            version="1.0.0"
        )
        
        # Analyze the configuration class
        fields = self._analyze_config_class(config_class)
        schema_info.fields = fields
        
        # Group fields by section
        schema_info.sections = self._group_fields_by_section(fields)
        
        # Generate examples
        schema_info.examples = self._generate_examples(config_class)
        
        return schema_info
    
    def _analyze_config_class(self, config_class: type) -> List[FieldInfo]:
        """Analyze configuration class and extract field information."""
        fields = []
        
        # Get field information from Pydantic model
        if hasattr(config_class, "model_fields"):
            for field_name, field_info in config_class.model_fields.items():
                field_data = self._extract_field_info(field_name, field_info)
                if field_data:
                    fields.append(field_data)
        
        return fields
    
    def _extract_field_info(self, field_name: str, field_info: Any) -> Optional[FieldInfo]:
        """Extract information from Pydantic field."""
        # Get field type
        field_type = self._get_field_type(field_info)
        
        # Get description
        description = field_info.description or ""
        
        # Get default value
        default = field_info.default if hasattr(field_info, "default") else None
        
        # Check if required
        required = default is None and not hasattr(field_info, "default_factory")
        
        # Get constraints
        constraints = self._extract_constraints(field_info)
        
        # Get enum values if applicable
        enum_values = self._get_enum_values(field_info)
        
        return FieldInfo(
            name=field_name,
            type=field_type,
            description=description,
            default=default,
            required=required,
            enum_values=enum_values,
            min_value=constraints.get("min_value"),
            max_value=constraints.get("max_value"),
            pattern=constraints.get("pattern"),
            examples=constraints.get("examples", [])
        )
    
    def _get_field_type(self, field_info: Any) -> str:
        """Get field type as string."""
        if hasattr(field_info, "annotation"):
            annotation = field_info.annotation
            
            # Handle Union types
            if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                types = annotation.__args__
                if len(types) == 2 and type(None) in types:
                    # Optional type
                    non_none_type = next(t for t in types if t is not type(None))
                    return self._map_type(non_none_type)
                else:
                    return "union"
            
            # Handle generic types
            if hasattr(annotation, "__origin__"):
                origin = annotation.__origin__
                if origin is list:
                    return "array"
                elif origin is dict:
                    return "object"
                else:
                    return self._map_type(origin)
            
            # Handle enum types
            if hasattr(annotation, "__members__"):
                return "enum"
            
            # Handle basic types
            return self._map_type(annotation)
        
        return "unknown"
    
    def _map_type(self, type_class: type) -> str:
        """Map Python type to schema type."""
        type_name = type_class.__name__
        return self._field_mappings.get(type_name, type_name.lower())
    
    def _extract_constraints(self, field_info: Any) -> Dict[str, Any]:
        """Extract validation constraints from field."""
        constraints = {}
        
        # Check for validators
        if hasattr(field_info, "constraints"):
            for constraint in field_info.constraints:
                if hasattr(constraint, "ge"):
                    constraints["min_value"] = constraint.ge
                if hasattr(constraint, "le"):
                    constraints["max_value"] = constraint.le
                if hasattr(constraint, "pattern"):
                    constraints["pattern"] = constraint.pattern
        
        return constraints
    
    def _get_enum_values(self, field_info: Any) -> Optional[List[str]]:
        """Get enum values if field is an enum."""
        if hasattr(field_info, "annotation"):
            annotation = field_info.annotation
            if hasattr(annotation, "__members__"):
                return list(annotation.__members__.keys())
        return None
    
    def _group_fields_by_section(self, fields: List[FieldInfo]) -> Dict[str, List[FieldInfo]]:
        """Group fields by configuration section."""
        sections = {}
        
        for field_info in fields:
            # Determine section based on field name
            if "." in field_info.name:
                section = field_info.name.split(".")[0]
            else:
                section = "general"
            
            if section not in sections:
                sections[section] = []
            sections[section].append(field_info)
        
        return sections
    
    def _generate_examples(self, config_class: type) -> Dict[str, Any]:
        """Generate example configurations."""
        examples = {}
        
        # Generate example for each environment
        for env in Environment:
            try:
                # Create example config
                example_config = config_class()
                example_config.environment = env
                
                # Convert to dict
                example_dict = example_config.model_dump(exclude_unset=True)
                examples[f"{env.value}_example"] = example_dict
                
            except Exception:
                # Skip if example generation fails
                pass
        
        return examples
    
    def export_schema(
        self,
        schema_info: SchemaInfo,
        format: SchemaFormat,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Export schema in specified format.
        
        Args:
            schema_info: Schema information to export
            format: Output format
            output_file: Optional output file path
            
        Returns:
            Exported schema content
        """
        if format == SchemaFormat.JSON:
            content = self._export_json(schema_info)
        elif format == SchemaFormat.YAML:
            content = self._export_yaml(schema_info)
        elif format == SchemaFormat.MARKDOWN:
            content = self._export_markdown(schema_info)
        elif format == SchemaFormat.HTML:
            content = self._export_html(schema_info)
        elif format == SchemaFormat.RST:
            content = self._export_rst(schema_info)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
        
        return content
    
    def _export_json(self, schema_info: SchemaInfo) -> str:
        """Export schema as JSON."""
        schema_dict = {
            "title": schema_info.title,
            "description": schema_info.description,
            "version": schema_info.version,
            "fields": [
                {
                    "name": field.name,
                    "type": field.type,
                    "description": field.description,
                    "default": field.default,
                    "required": field.required,
                    "enum_values": field.enum_values,
                    "min_value": field.min_value,
                    "max_value": field.max_value,
                    "pattern": field.pattern,
                    "examples": field.examples,
                    "deprecated": field.deprecated,
                    "deprecation_message": field.deprecation_message
                }
                for field in schema_info.fields
            ],
            "sections": {
                section: [
                    {
                        "name": field.name,
                        "type": field.type,
                        "description": field.description,
                        "default": field.default,
                        "required": field.required
                    }
                    for field in fields
                ]
                for section, fields in schema_info.sections.items()
            },
            "examples": schema_info.examples
        }
        
        return json.dumps(schema_dict, indent=2, default=str)
    
    def _export_yaml(self, schema_info: SchemaInfo) -> str:
        """Export schema as YAML."""
        import yaml
        
        schema_dict = {
            "title": schema_info.title,
            "description": schema_info.description,
            "version": schema_info.version,
            "fields": [
                {
                    "name": field.name,
                    "type": field.type,
                    "description": field.description,
                    "default": field.default,
                    "required": field.required,
                    "enum_values": field.enum_values,
                    "min_value": field.min_value,
                    "max_value": field.max_value,
                    "pattern": field.pattern,
                    "examples": field.examples,
                    "deprecated": field.deprecated,
                    "deprecation_message": field.deprecation_message
                }
                for field in schema_info.fields
            ],
            "sections": {
                section: [
                    {
                        "name": field.name,
                        "type": field.type,
                        "description": field.description,
                        "default": field.default,
                        "required": field.required
                    }
                    for field in fields
                ]
                for section, fields in schema_info.sections.items()
            },
            "examples": schema_info.examples
        }
        
        return yaml.dump(schema_dict, default_flow_style=False, indent=2)
    
    def _export_markdown(self, schema_info: SchemaInfo) -> str:
        """Export schema as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# {schema_info.title}")
        lines.append("")
        lines.append(schema_info.description)
        lines.append("")
        lines.append(f"**Version:** {schema_info.version}")
        lines.append("")
        
        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for section in schema_info.sections.keys():
            lines.append(f"- [{section.title()}](#{section.lower()})")
        lines.append("- [Examples](#examples)")
        lines.append("")
        
        # Sections
        for section, fields in schema_info.sections.items():
            lines.append(f"## {section.title()}")
            lines.append("")
            
            if fields:
                lines.append("| Field | Type | Required | Default | Description |")
                lines.append("|-------|------|----------|---------|-------------|")
                
                for field in fields:
                    required_str = "Yes" if field.required else "No"
                    default_str = str(field.default) if field.default is not None else "-"
                    description_str = field.description or "-"
                    
                    lines.append(f"| `{field.name}` | {field.type} | {required_str} | {default_str} | {description_str} |")
                
                lines.append("")
                
                # Field details
                for field in fields:
                    if field.description or field.enum_values or field.examples:
                        lines.append(f"### {field.name}")
                        lines.append("")
                        
                        if field.description:
                            lines.append(f"**Description:** {field.description}")
                            lines.append("")
                        
                        if field.enum_values:
                            lines.append(f"**Valid Values:** {', '.join(f'`{v}`' for v in field.enum_values)}")
                            lines.append("")
                        
                        if field.examples:
                            lines.append("**Examples:**")
                            for example in field.examples:
                                lines.append(f"- `{example}`")
                            lines.append("")
        
        # Examples
        lines.append("## Examples")
        lines.append("")
        
        for example_name, example_config in schema_info.examples.items():
            lines.append(f"### {example_name.replace('_', ' ').title()}")
            lines.append("")
            lines.append("```yaml")
            import yaml
            lines.append(yaml.dump(example_config, default_flow_style=False, indent=2))
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_html(self, schema_info: SchemaInfo) -> str:
        """Export schema as HTML."""
        lines = []
        
        # HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append(f"<title>{schema_info.title}</title>")
        lines.append("<style>")
        lines.append("""
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
        .required { color: #d32f2f; font-weight: bold; }
        .optional { color: #666; }
        """)
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")
        
        # Title
        lines.append(f"<h1>{schema_info.title}</h1>")
        lines.append(f"<p>{schema_info.description}</p>")
        lines.append(f"<p><strong>Version:</strong> {schema_info.version}</p>")
        
        # Sections
        for section, fields in schema_info.sections.items():
            lines.append(f"<h2>{section.title()}</h2>")
            
            if fields:
                lines.append("<table>")
                lines.append("<tr><th>Field</th><th>Type</th><th>Required</th><th>Default</th><th>Description</th></tr>")
                
                for field in fields:
                    required_class = "required" if field.required else "optional"
                    required_text = "Yes" if field.required else "No"
                    default_text = str(field.default) if field.default is not None else "-"
                    
                    lines.append("<tr>")
                    lines.append(f"<td><code>{field.name}</code></td>")
                    lines.append(f"<td>{field.type}</td>")
                    lines.append(f"<td class='{required_class}'>{required_text}</td>")
                    lines.append(f"<td>{default_text}</td>")
                    lines.append(f"<td>{field.description or '-'}</td>")
                    lines.append("</tr>")
                
                lines.append("</table>")
        
        # Examples
        lines.append("<h2>Examples</h2>")
        for example_name, example_config in schema_info.examples.items():
            lines.append(f"<h3>{example_name.replace('_', ' ').title()}</h3>")
            lines.append("<pre>")
            import yaml
            lines.append(yaml.dump(example_config, default_flow_style=False, indent=2))
            lines.append("</pre>")
        
        lines.append("</body>")
        lines.append("</html>")
        
        return "\n".join(lines)
    
    def _export_rst(self, schema_info: SchemaInfo) -> str:
        """Export schema as reStructuredText."""
        lines = []
        
        # Title
        lines.append(schema_info.title)
        lines.append("=" * len(schema_info.title))
        lines.append("")
        lines.append(schema_info.description)
        lines.append("")
        lines.append(f"**Version:** {schema_info.version}")
        lines.append("")
        
        # Sections
        for section, fields in schema_info.sections.items():
            lines.append(section.title())
            lines.append("-" * len(section.title()))
            lines.append("")
            
            if fields:
                lines.append(".. list-table::")
                lines.append("   :header-rows: 1")
                lines.append("   :widths: 20 10 10 15 45")
                lines.append("")
                lines.append("   * - Field")
                lines.append("     - Type")
                lines.append("     - Required")
                lines.append("     - Default")
                lines.append("     - Description")
                
                for field in fields:
                    required_text = "Yes" if field.required else "No"
                    default_text = str(field.default) if field.default is not None else "-"
                    
                    lines.append(f"   * - ``{field.name}``")
                    lines.append(f"     - {field.type}")
                    lines.append(f"     - {required_text}")
                    lines.append(f"     - {default_text}")
                    lines.append(f"     - {field.description or '-'}")
                
                lines.append("")
        
        return "\n".join(lines)


# Convenience functions
def generate_schema_docs(
    config_class: type = AppConfig,
    format: SchemaFormat = SchemaFormat.MARKDOWN,
    output_file: Optional[Path] = None
) -> str:
    """
    Generate configuration schema documentation.
    
    Args:
        config_class: Configuration class to document
        format: Output format
        output_file: Optional output file path
        
    Returns:
        Generated documentation
    """
    generator = SchemaGenerator()
    schema_info = generator.generate_schema(config_class)
    return generator.export_schema(schema_info, format, output_file)


def export_schema_json(
    config_class: type = AppConfig,
    output_file: Optional[Path] = None
) -> str:
    """Export schema as JSON."""
    return generate_schema_docs(config_class, SchemaFormat.JSON, output_file)


def export_schema_yaml(
    config_class: type = AppConfig,
    output_file: Optional[Path] = None
) -> str:
    """Export schema as YAML."""
    return generate_schema_docs(config_class, SchemaFormat.YAML, output_file)
