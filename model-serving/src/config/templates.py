"""
Configuration templates for different environments.

Provides template-based configuration generation with variable substitution
and environment-specific template management.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, Template

from .settings import Environment as ConfigEnvironment


@dataclass
class ConfigTemplate:
    """Configuration template with variables and content."""
    
    name: str
    description: str = ""
    environment: Optional[Environment] = None
    template_content: str = ""
    variables: Dict[str, Any] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.variables is None:
            self.variables = {}
        if self.dependencies is None:
            self.dependencies = []
    
    def render(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render template with context.
        
        Args:
            context: Template context variables
            
        Returns:
            Rendered template content
        """
        if context is None:
            context = {}
        
        # Merge template variables with context
        render_context = {**self.variables, **context}
        
        # Use Jinja2 for template rendering
        template = Template(self.template_content)
        return template.render(**render_context)
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """
        Get template variable value.
        
        Args:
            name: Variable name
            default: Default value if not found
            
        Returns:
            Variable value or default
        """
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any) -> None:
        """
        Set template variable.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.variables[name] = value


class TemplateEngine:
    """Template engine for configuration generation."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize template engine.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir or Path("config/templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Custom filters
        self._register_filters()
        
        # Template cache
        self._templates: Dict[str, ConfigTemplate] = {}
        self._load_builtin_templates()
    
    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        def env_var(name: str, default: str = "") -> str:
            """Get environment variable with default."""
            return os.getenv(name, default)
        
        def bool_str(value: Any) -> str:
            """Convert boolean to string."""
            return str(bool(value)).lower()
        
        def join_list(items: List[str], separator: str = ",") -> str:
            """Join list items with separator."""
            return separator.join(str(item) for item in items)
        
        def indent_text(text: str, spaces: int = 2) -> str:
            """Indent text by specified spaces."""
            lines = text.split('\n')
            indented = [f"{' ' * spaces}{line}" for line in lines]
            return '\n'.join(indented)
        
        self.jinja_env.filters['env_var'] = env_var
        self.jinja_env.filters['bool_str'] = bool_str
        self.jinja_env.filters['join_list'] = join_list
        self.jinja_env.filters['indent_text'] = indent_text
    
    def _load_builtin_templates(self) -> None:
        """Load built-in configuration templates."""
        # Development template
        dev_template = ConfigTemplate(
            name="development",
            description="Development environment configuration",
            environment=ConfigEnvironment.DEVELOPMENT,
            template_content="""
# Development Configuration
debug: true
environment: development

server:
  host: "0.0.0.0"
  port: {{ port | default(8000) }}
  workers: 1
  reload: true
  log_level: "DEBUG"

model:
  backend: "{{ model_backend | default('pytorch') }}"
  device: "{{ device | default('auto') }}"
  batch_size: {{ batch_size | default(1) }}
  optimize: false

logging:
  level: "DEBUG"
  json_format: false
  file_path: "{{ log_file | default('logs/dev.log') }}"

monitoring:
  enabled: false

security:
  enabled: false
""",
            variables={
                "port": 8000,
                "model_backend": "pytorch",
                "device": "auto",
                "batch_size": 1,
                "log_file": "logs/dev.log"
            }
        )
        
        # Production template
        prod_template = ConfigTemplate(
            name="production",
            description="Production environment configuration",
            environment=ConfigEnvironment.PRODUCTION,
            template_content="""
# Production Configuration
debug: false
environment: production

server:
  host: "{{ host | default('0.0.0.0') }}"
  port: {{ port | default(8000) }}
  workers: {{ workers | default(4) }}
  reload: false
  log_level: "WARNING"
  timeout: 60.0
  max_connections: 1000

model:
  backend: "{{ model_backend | default('pytorch') }}"
  device: "{{ device | default('auto') }}"
  batch_size: {{ batch_size | default(8) }}
  max_batch_size: {{ max_batch_size | default(32) }}
  optimize: true
  cache_models: true

logging:
  level: "WARNING"
  json_format: true
  file_path: "{{ log_file | default('logs/app.log') }}"
  max_file_size: 10485760  # 10MB
  backup_count: 5

monitoring:
  enabled: true
  metrics_enabled: true
  tracing_enabled: true
  prometheus_enabled: true
  metrics_port: {{ metrics_port | default(9090) }}

security:
  enabled: true
  api_key_required: true
  cors_enabled: true
  rate_limit_enabled: true
  rate_limit_requests: {{ rate_limit | default(100) }}

database:
  backend: "{{ db_backend | default('postgresql') }}"
  host: "{{ db_host | env_var('DB_HOST', 'localhost') }}"
  port: {{ db_port | default(5432) }}
  name: "{{ db_name | env_var('DB_NAME', 'model_serving') }}"
  username: "{{ db_user | env_var('DB_USER', '') }}"
  password: "{{ db_password | env_var('DB_PASSWORD', '') }}"

cache:
  backend: "{{ cache_backend | default('redis') }}"
  host: "{{ cache_host | env_var('CACHE_HOST', 'localhost') }}"
  port: {{ cache_port | default(6379) }}
  password: "{{ cache_password | env_var('CACHE_PASSWORD', '') }}"
""",
            variables={
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "model_backend": "pytorch",
                "device": "auto",
                "batch_size": 8,
                "max_batch_size": 32,
                "log_file": "logs/app.log",
                "metrics_port": 9090,
                "rate_limit": 100,
                "db_backend": "postgresql",
                "db_host": "localhost",
                "db_port": 5432,
                "db_name": "model_serving",
                "cache_backend": "redis",
                "cache_host": "localhost",
                "cache_port": 6379
            }
        )
        
        # Testing template
        test_template = ConfigTemplate(
            name="testing",
            description="Testing environment configuration",
            environment=ConfigEnvironment.TESTING,
            template_content="""
# Testing Configuration
debug: true
environment: testing

server:
  host: "127.0.0.1"
  port: {{ port | default(8001) }}
  workers: 1
  reload: false
  log_level: "DEBUG"

model:
  backend: "{{ model_backend | default('pytorch') }}"
  device: "cpu"
  batch_size: 1
  optimize: false
  cache_models: false

logging:
  level: "DEBUG"
  json_format: false

monitoring:
  enabled: false

security:
  enabled: false

database:
  backend: "sqlite"
  name: "test_model_serving"

cache:
  backend: "memory"
""",
            variables={
                "port": 8001,
                "model_backend": "pytorch"
            }
        )
        
        # Add templates to cache
        self._templates["development"] = dev_template
        self._templates["production"] = prod_template
        self._templates["testing"] = test_template
    
    def add_template(self, template: ConfigTemplate) -> None:
        """
        Add template to engine.
        
        Args:
            template: Template to add
        """
        self._templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """
        Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template instance or None if not found
        """
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """
        List all available templates.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def render_template(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Render template to string or file.
        
        Args:
            name: Template name
            context: Template context
            output_file: Optional output file path
            
        Returns:
            Rendered template content
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Template not found: {name}")
        
        content = template.render(context)
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
        
        return content
    
    def load_template_from_file(self, file_path: Union[str, Path]) -> ConfigTemplate:
        """
        Load template from file.
        
        Args:
            file_path: Template file path
            
        Returns:
            Loaded template
        """
        file_path = Path(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract metadata from YAML front matter if present
        variables = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                import yaml
                try:
                    metadata = yaml.safe_load(parts[1])
                    variables = metadata.get("variables", {})
                    content = parts[2].strip()
                except yaml.YAMLError:
                    pass
        
        template = ConfigTemplate(
            name=file_path.stem,
            template_content=content,
            variables=variables
        )
        
        self.add_template(template)
        return template
    
    def save_template(self, template: ConfigTemplate, file_path: Optional[Path] = None) -> None:
        """
        Save template to file.
        
        Args:
            template: Template to save
            file_path: Output file path
        """
        if file_path is None:
            file_path = self.templates_dir / f"{template.name}.yaml"
        
        # Create YAML front matter with variables
        front_matter = "---\n"
        if template.variables:
            import yaml
            front_matter += yaml.dump({"variables": template.variables}, default_flow_style=False)
        front_matter += "---\n\n"
        
        content = front_matter + template.template_content
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def generate_config(
        self,
        template_name: str,
        environment: Environment,
        context: Optional[Dict[str, Any]] = None,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate configuration from template.
        
        Args:
            template_name: Template name
            environment: Target environment
            context: Additional context variables
            output_file: Optional output file path
            
        Returns:
            Generated configuration content
        """
        if context is None:
            context = {}
        
        # Add environment-specific context
        context.update({
            "environment": environment.value,
            "env": environment.value
        })
        
        return self.render_template(template_name, context, output_file)


def create_template(
    name: str,
    content: str,
    environment: Optional[Environment] = None,
    variables: Optional[Dict[str, Any]] = None,
    description: str = ""
) -> ConfigTemplate:
    """
    Create a configuration template.
    
    Args:
        name: Template name
        content: Template content
        environment: Target environment
        variables: Template variables
        description: Template description
        
    Returns:
        Created template
    """
    return ConfigTemplate(
        name=name,
        description=description,
        environment=environment,
        template_content=content,
        variables=variables or {}
    )


def create_environment_template(environment: Environment) -> ConfigTemplate:
    """
    Create template for specific environment.
    
    Args:
        environment: Target environment
        
    Returns:
        Environment-specific template
    """
    engine = TemplateEngine()
    return engine.get_template(environment.value)


def generate_config_from_template(
    template_name: str,
    environment: Environment,
    context: Optional[Dict[str, Any]] = None,
    output_file: Optional[Path] = None
) -> str:
    """
    Generate configuration from template.
    
    Args:
        template_name: Template name
        environment: Target environment
        context: Additional context
        output_file: Optional output file
        
    Returns:
        Generated configuration
    """
    engine = TemplateEngine()
    return engine.generate_config(template_name, environment, context, output_file)
