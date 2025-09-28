"""
Hierarchical configuration loading (env > file > defaults).

Provides flexible configuration loading with support for multiple sources
and formats, with environment variables taking precedence.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from pydantic import BaseModel, ValidationError

from .settings import AppConfig, BaseConfig

T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """Configuration loader with hierarchical loading support."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Supported file formats
        self.supported_formats = {".json", ".yaml", ".yml", ".toml", ".env"}
        
        # Configuration sources (in order of precedence)
        self.sources = []
    
    def add_file_source(self, file_path: Union[str, Path], required: bool = False) -> "ConfigLoader":
        """
        Add file-based configuration source.
        
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
    
    def add_env_source(self, prefix: str = "", env_file: Optional[Union[str, Path]] = None) -> "ConfigLoader":
        """
        Add environment variable source.
        
        Args:
            prefix: Environment variable prefix
            env_file: Path to .env file
            
        Returns:
            Self for method chaining
        """
        if env_file:
            env_file = Path(env_file)
            if not env_file.is_absolute():
                env_file = self.config_dir / env_file
            
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
                # Load from .env file
                env_data = self._load_env_file(source_value)
                config_data = self._merge_configs(config_data, env_data)
                
            elif source_type == "env":
                # Load from environment variables
                env_data = self._load_env_vars(source_value)
                config_data = self._merge_configs(config_data, env_data)
        
        # Create configuration object
        try:
            return config_class(**config_data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _load_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if suffix == ".json":
                    return json.load(f)
                elif suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif suffix == ".toml":
                    import toml
                    return toml.load(f)
                elif suffix == ".env":
                    return self._load_env_file(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load configuration file {file_path}: {e}")
    
    def _load_env_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from .env file."""
        env_data = {}
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Remove quotes if present
                        value = value.strip('"').strip("'")
                        env_data[key.strip()] = value
        except Exception as e:
            raise ValueError(f"Failed to load .env file {file_path}: {e}")
        
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
    env_prefix: str = "",
    config_class: Type[T] = AppConfig,
    config_dir: Optional[Path] = None
) -> T:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Configuration file path
        env_prefix: Environment variable prefix
        config_class: Configuration class to instantiate
        config_dir: Configuration directory
        
    Returns:
        Loaded configuration object
    """
    loader = ConfigLoader(config_dir)
    
    # Add default configuration
    loader.add_default_source(config_class())
    
    # Add file sources
    if config_file:
        loader.add_file_source(config_file, required=True)
    else:
        # Try common configuration files
        common_files = ["config.yaml", "config.yml", "config.json", "config.toml"]
        for file_name in common_files:
            loader.add_file_source(file_name, required=False)
    
    # Add environment sources only if prefix is provided
    if env_prefix:
        loader.add_env_source(env_prefix)
    
    return loader.load_config(config_class)


def create_config_loader(
    config_dir: Optional[Path] = None,
    env_prefix: str = "",
    config_files: Optional[List[Union[str, Path]]] = None
) -> ConfigLoader:
    """
    Create a pre-configured ConfigLoader.
    
    Args:
        config_dir: Configuration directory
        env_prefix: Environment variable prefix
        config_files: List of configuration files to load
        
    Returns:
        Configured ConfigLoader instance
    """
    loader = ConfigLoader(config_dir)
    
    # Add file sources
    if config_files:
        for file_path in config_files:
            loader.add_file_source(file_path)
    
    # Add environment source
    loader.add_env_source(env_prefix)
    
    return loader
