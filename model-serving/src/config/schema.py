"""
Configuration schema documentation generator.

Provides tools for generating configuration schema documentation
in various formats (JSON, YAML, Markdown, HTML).
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
            "dict": "object",
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
            version="1.0.0",
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
            examples=constraints.get("examples", []),
        )

    def _get_field_type(self, field_info: Any) -> str:
        """Get field type as string."""
        annotation = getattr(field_info, "annotation", None)
        if annotation is None:
            return "unknown"
        origin = getattr(annotation, "__origin__", None)
        if origin is Union:
            return self._map_union_type(annotation)
        if origin is not None:
            return self._map_generic_type(origin)
        if hasattr(annotation, "__members__"):
            return "enum"
        return self._map_type(annotation)

    def _map_union_type(self, annotation: Any) -> str:
        """Map a union, preserving the simpler type for Optional values."""
        union_types = annotation.__args__
        if len(union_types) == 2 and type(None) in union_types:
            concrete = next(item for item in union_types if item is not type(None))
            return self._map_type(concrete)
        return "union"

    def _map_generic_type(self, origin: type) -> str:
        """Map the origin of a generic annotation."""
        generic_types = {list: "array", dict: "object"}
        return generic_types.get(origin, self._map_type(origin))

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
        self, schema_info: SchemaInfo, format: SchemaFormat, output_file: Optional[Path] = None
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
                    "deprecation_message": field.deprecation_message,
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
                        "required": field.required,
                    }
                    for field in fields
                ]
                for section, fields in schema_info.sections.items()
            },
            "examples": schema_info.examples,
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
                    "deprecation_message": field.deprecation_message,
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
                        "required": field.required,
                    }
                    for field in fields
                ]
                for section, fields in schema_info.sections.items()
            },
            "examples": schema_info.examples,
        }

        return yaml.dump(schema_dict, default_flow_style=False, indent=2)

    def _export_markdown(self, schema_info: SchemaInfo) -> str:
        """Export schema as Markdown."""
        lines = [
            f"# {schema_info.title}",
            "",
            schema_info.description,
            "",
            f"**Version:** {schema_info.version}",
            "",
            "## Table of Contents",
            "",
        ]
        lines.extend(
            f"- [{section.title()}](#{section.lower()})" for section in schema_info.sections
        )
        lines.extend(["- [Examples](#examples)", ""])
        for section, fields in schema_info.sections.items():
            lines.extend(self._markdown_section(section, fields))
        lines.extend(["## Examples", ""])
        for example_name, example_config in schema_info.examples.items():
            lines.extend(self._markdown_example(example_name, example_config))
        return "\n".join(lines)

    def _markdown_section(self, section: str, fields: List[FieldInfo]) -> List[str]:
        """Render one Markdown field section."""
        lines = [f"## {section.title()}", ""]
        if not fields:
            return lines
        lines.extend(
            [
                "| Field | Type | Required | Default | Description |",
                "|-------|------|----------|---------|-------------|",
            ]
        )
        lines.extend(self._markdown_field_row(field_info) for field_info in fields)
        lines.append("")
        for field_info in fields:
            lines.extend(self._markdown_field_details(field_info))
        return lines

    @staticmethod
    def _markdown_field_row(field: FieldInfo) -> str:
        required = "Yes" if field.required else "No"
        default = str(field.default) if field.default is not None else "-"
        return (
            f"| `{field.name}` | {field.type} | {required} | {default} | "
            f"{field.description or '-'} |"
        )

    @staticmethod
    def _markdown_field_details(field: FieldInfo) -> List[str]:
        if not (field.description or field.enum_values or field.examples):
            return []
        lines = [f"### {field.name}", ""]
        if field.description:
            lines.extend([f"**Description:** {field.description}", ""])
        if field.enum_values:
            values = ", ".join(f"`{value}`" for value in field.enum_values)
            lines.extend([f"**Valid Values:** {values}", ""])
        if field.examples:
            lines.append("**Examples:**")
            lines.extend(f"- `{example}`" for example in field.examples)
            lines.append("")
        return lines

    @staticmethod
    def _markdown_example(name: str, config: Dict[str, Any]) -> List[str]:
        import yaml

        return [
            f"### {name.replace('_', ' ').title()}",
            "",
            "```yaml",
            yaml.dump(config, default_flow_style=False, indent=2),
            "```",
            "",
        ]

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

        for section, fields in schema_info.sections.items():
            lines.extend(self._html_section(section, fields))

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

    def _html_section(self, section: str, fields: List[FieldInfo]) -> List[str]:
        """Render one HTML field section."""
        lines = [f"<h2>{section.title()}</h2>"]
        if not fields:
            return lines
        lines.extend(
            [
                "<table>",
                "<tr><th>Field</th><th>Type</th><th>Required</th><th>Default</th><th>Description</th></tr>",
            ]
        )
        for field_info in fields:
            lines.extend(self._html_field_row(field_info))
        lines.append("</table>")
        return lines

    @staticmethod
    def _html_field_row(field: FieldInfo) -> List[str]:
        required_class = "required" if field.required else "optional"
        required_text = "Yes" if field.required else "No"
        default_text = str(field.default) if field.default is not None else "-"
        return [
            "<tr>",
            f"<td><code>{field.name}</code></td>",
            f"<td>{field.type}</td>",
            f"<td class='{required_class}'>{required_text}</td>",
            f"<td>{default_text}</td>",
            f"<td>{field.description or '-'}</td>",
            "</tr>",
        ]

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
    output_file: Optional[Path] = None,
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


def export_schema_json(config_class: type = AppConfig, output_file: Optional[Path] = None) -> str:
    """Export schema as JSON."""
    return generate_schema_docs(config_class, SchemaFormat.JSON, output_file)


def export_schema_yaml(config_class: type = AppConfig, output_file: Optional[Path] = None) -> str:
    """Export schema as YAML."""
    return generate_schema_docs(config_class, SchemaFormat.YAML, output_file)
