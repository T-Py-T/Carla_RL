"""
Hierarchical configuration loading system.

Provides configuration loading from multiple sources with proper precedence:
environment variables > configuration files > default values.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from .settings import BaseConfig, AppConfig

T = TypeVar('T', bound=BaseConfig)


class ConfigLoader:
    """Configuration loader with hierarchical source management."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Base directory for configuration files
        """
        self.config_dir = config_dir or Path("config")
        self.sources: List[tuple] = []
    
    def add_file_source(self, file_path: Union[str, Path], required: bool = False) -> "ConfigLoader":
        """
        Add configuration file source.
        
        Args:
            file_path: Path to configuration file
            required: Whether file is required to exist
            
        Returns:
            Self for method chaining
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.config_dir / file_path
        
        if required and not file_path.exists():
            raise FileNotFoundError(f"Required configuration file not found: {file_path}")
        
        if file_path.exists():
            self.sources.append(("file", str(file_path)))
        
        return self
    
    def add_env_file_source(self, env_file: Union[str, Path]) -> "ConfigLoader":
        """
        Add environment file source.
        
        Args:
            env_file: Path to environment file
            
        Returns:
            Self for method chaining
        """
        env_file = Path(env_file)
        if not env_file.is_absolute():
            env_file = self.config_dir / env_file
        
        if env_file.exists():
            self.sources.append(("env_file", str(env_file)))
        
        return self
    
    def add_env_source(self, prefix: str = "") -> "ConfigLoader":
        """
        Add environment variable source.
        
        Args:
            prefix: Environment variable prefix (e.g., "APP_")
            
        Returns:
            Self for method chaining
        """
        if prefix:
            # Load .env files if they exist
            env_files = [
                self.config_dir / ".env",
                self.config_dir / f".env.{os.getenv('ENVIRONMENT', 'development')}",
                Path.cwd() / ".env",
                Path.cwd() / f".env.{os.getenv('ENVIRONMENT', 'development')}"
            ]
            
            for env_file in env_files:
                if env_file.exists():
                    self.sources.append(("env_file", str(env_file)))
        
        self.sources.append(("env", prefix))
        return self
    
    def add_default_source(self, config: BaseConfig) -> "ConfigLoader":
        """
        Add default configuration source.
        
        Args:
            config: Default configuration object
            
        Returns:
            Self for method chaining
        """
        self.sources.append(("default", config))
        return self
    
    def load_config(self, config_class: Type[T] = AppConfig) -> T:
        """
        Load configuration using hierarchical sources.
        
        Args:
            config_class: Configuration class to instantiate
            
        Returns:
            Loaded configuration object
            
        Raises:
            ValidationError: If configuration validation fails
            FileNotFoundError: If required files are missing
        """
        # Start with empty configuration
        config_data = {}
        
        # Load from sources in order (defaults first, env last)
        for source_type, source_value in self.sources:
            if source_type == "default":
                # Merge default configuration
                default_data = source_value.model_dump(exclude_unset=True)
                config_data = self._merge_configs(config_data, default_data)
                
            elif source_type == "file":
                # Load from file
                file_data = self._load_file(source_value)
                config_data = self._merge_configs(config_data, file_data)
                
            elif source_type == "env_file":
                # Load from environment file
                env_data = self._load_env_file(source_value)
                config_data = self._merge_configs(config_data, env_data)
                
            elif source_type == "env":
                # Load from environment variables
                env_data = self._load_env_vars(source_value)
                config_data = self._merge_configs(config_data, env_data)
        
        # Validate and create configuration object
        try:
            return config_class.model_validate(config_data)
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _load_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                return json.load(f)
            elif suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif suffix == '.toml':
                import toml
                return toml.load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_env_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from environment file."""
        env_data = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    env_data[key] = value
        
        return env_data
    
    def _load_env_vars(self, prefix: str = "") -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_data = {}
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix.upper()):
                continue
            
            # Remove prefix if present
            if prefix:
                config_key = key[len(prefix):].lstrip("_")
            else:
                config_key = key
            
            # Convert to nested dictionary structure
            self._set_nested_value(env_data, config_key.lower(), value)
        
        return env_data
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: str) -> None:
        """Set nested dictionary value from dot-notation key."""
        # Handle both dot notation and underscore notation
        if "." in key:
            keys = key.split(".")
        else:
            # For underscore notation, split on underscores
            keys = key.split("_")
        
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                # If the value is not a dict, convert it to one
                current[k] = {}
            current = current[k]
        
        # Convert value to appropriate type
        converted_value = self._convert_env_value(value)
        current[keys[-1]] = converted_value
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool, None]:
        """Convert environment variable string to appropriate type."""
        if value.lower() in ("true", "yes", "on", "1"):
            return True
        elif value.lower() in ("false", "no", "off", "0"):
            return False
        elif value.lower() in ("null", "none", ""):
            return None
        elif value.isdigit():
            return int(value)
        elif value.replace(".", "", 1).isdigit():
            return float(value)
        else:
            return value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries recursively."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: BaseConfig, file_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            file_path: Output file path
            format: Output format (json, yaml, toml)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = config.model_dump(exclude_unset=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            if format.lower() == "json":
                json.dump(config_data, f, indent=2, default=str)
            elif format.lower() in ["yaml", "yml"]:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == "toml":
                import toml
                toml.dump(config_data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    env_prefix: str = "APP_",
    config_class: Type[T] = AppConfig
) -> T:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Optional configuration file path
        env_prefix: Environment variable prefix
        config_class: Configuration class to instantiate
        
    Returns:
        Loaded configuration object
    """
    loader = ConfigLoader()
    
    # Add default source
    loader.add_default_source(config_class())
    
    # Add file source if provided
    if config_file:
        loader.add_file_source(config_file, required=True)
    
    # Add environment source
    loader.add_env_source(env_prefix)
    
    return loader.load_config(config_class)


def create_config_loader(config_dir: Optional[Path] = None) -> ConfigLoader:
    """
    Create a new configuration loader.
    
    Args:
        config_dir: Base directory for configuration files
        
    Returns:
        New configuration loader instance
    """
    return ConfigLoader(config_dir)