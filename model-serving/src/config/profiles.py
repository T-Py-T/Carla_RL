"""
Environment-specific configuration profiles.

Provides configuration profiles for different environments with
template-based configuration generation.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .settings import AppConfig, Environment


class ProfileType(str, Enum):
    """Configuration profile types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    CUSTOM = "custom"


@dataclass
class ConfigProfile:
    """Configuration profile with environment-specific settings."""
    
    name: str
    profile_type: ProfileType
    environment: Environment
    description: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def apply_to_config(self, config: AppConfig) -> AppConfig:
        """
        Apply profile settings to configuration.
        
        Args:
            config: Base configuration to modify
            
        Returns:
            Modified configuration
        """
        # Apply base settings
        config_dict = config.model_dump()
        self._apply_settings(config_dict, self.settings)
        
        # Apply overrides
        self._apply_settings(config_dict, self.overrides)
        
        # Create new config instance
        return AppConfig(**config_dict)
    
    def _apply_settings(self, config_dict: Dict[str, Any], settings: Dict[str, Any]) -> None:
        """Apply settings to configuration dictionary."""
        for key, value in settings.items():
            if "." in key:
                # Nested setting
                self._set_nested_value(config_dict, key, value)
            else:
                # Top-level setting
                config_dict[key] = value
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested value in dictionary using dot notation."""
        keys = key.split(".")
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get_variable(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get profile variable value.
        
        Args:
            name: Variable name
            default: Default value if not found
            
        Returns:
            Variable value or default
        """
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: str) -> None:
        """
        Set profile variable.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.variables[name] = value


class EnvironmentProfile(ConfigProfile):
    """Environment-specific configuration profile."""
    
    def __init__(
        self,
        environment: Environment,
        base_settings: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize environment profile.
        
        Args:
            environment: Target environment
            base_settings: Base configuration settings
            overrides: Environment-specific overrides
        """
        super().__init__(
            name=f"{environment.value}_profile",
            profile_type=ProfileType(environment.value),
            environment=environment,
            description=f"Configuration profile for {environment.value} environment"
        )
        
        # Set environment-specific settings
        if base_settings:
            self.settings.update(base_settings)
        
        if overrides:
            self.overrides.update(overrides)
        
        # Apply environment-specific defaults
        self._apply_environment_defaults()
    
    def _apply_environment_defaults(self) -> None:
        """Apply environment-specific default settings."""
        if self.environment == Environment.DEVELOPMENT:
            self.settings.update({
                "debug": True,
                "server.reload": True,
                "server.workers": 1,
                "logging.level": "DEBUG",
                "monitoring.enabled": False,
                "security.enabled": False
            })
        
        elif self.environment == Environment.STAGING:
            self.settings.update({
                "debug": False,
                "server.reload": False,
                "server.workers": 2,
                "logging.level": "INFO",
                "monitoring.enabled": True,
                "security.enabled": True,
                "security.api_key_required": True
            })
        
        elif self.environment == Environment.PRODUCTION:
            self.settings.update({
                "debug": False,
                "server.reload": False,
                "server.workers": 4,
                "logging.level": "WARNING",
                "monitoring.enabled": True,
                "security.enabled": True,
                "security.api_key_required": True,
                "security.rate_limit_enabled": True,
                "server.timeout": 60.0
            })
        
        elif self.environment == Environment.TESTING:
            self.settings.update({
                "debug": True,
                "server.reload": False,
                "server.workers": 1,
                "logging.level": "DEBUG",
                "monitoring.enabled": False,
                "security.enabled": False,
                "database.backend": "sqlite",
                "database.name": "test_model_serving",
                "cache.backend": "memory"
            })


class ProfileManager:
    """Manager for configuration profiles."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize profile manager.
        
        Args:
            profiles_dir: Directory containing profile files
        """
        self.profiles_dir = profiles_dir or Path("config/profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, ConfigProfile] = {}
        self._load_builtin_profiles()
    
    def _load_builtin_profiles(self) -> None:
        """Load built-in environment profiles."""
        for env in Environment:
            profile = EnvironmentProfile(env)
            self._profiles[profile.name] = profile
    
    def add_profile(self, profile: ConfigProfile) -> None:
        """
        Add configuration profile.
        
        Args:
            profile: Profile to add
        """
        self._profiles[profile.name] = profile
    
    def get_profile(self, name: str) -> Optional[ConfigProfile]:
        """
        Get profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            Profile instance or None if not found
        """
        return self._profiles.get(name)
    
    def list_profiles(self) -> List[str]:
        """
        List all available profiles.
        
        Returns:
            List of profile names
        """
        return list(self._profiles.keys())
    
    def get_environment_profile(self, environment: Environment) -> Optional[ConfigProfile]:
        """
        Get profile for specific environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Environment profile or None if not found
        """
        profile_name = f"{environment.value}_profile"
        return self._profiles.get(profile_name)
    
    def create_profile_from_config(
        self,
        name: str,
        config: AppConfig,
        description: str = ""
    ) -> ConfigProfile:
        """
        Create profile from existing configuration.
        
        Args:
            name: Profile name
            config: Configuration to base profile on
            description: Profile description
            
        Returns:
            Created profile
        """
        profile = ConfigProfile(
            name=name,
            profile_type=ProfileType.CUSTOM,
            environment=config.environment,
            description=description,
            settings=config.model_dump(exclude_unset=True)
        )
        
        self.add_profile(profile)
        return profile
    
    def save_profile(self, profile: ConfigProfile, file_path: Optional[Path] = None) -> None:
        """
        Save profile to file.
        
        Args:
            profile: Profile to save
            file_path: Output file path
        """
        if file_path is None:
            file_path = self.profiles_dir / f"{profile.name}.yaml"
        
        import yaml
        
        profile_data = {
            "name": profile.name,
            "profile_type": profile.profile_type.value,
            "environment": profile.environment.value,
            "description": profile.description,
            "settings": profile.settings,
            "overrides": profile.overrides,
            "variables": profile.variables,
            "dependencies": profile.dependencies
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(profile_data, f, default_flow_style=False, indent=2)
    
    def load_profile(self, file_path: Union[str, Path]) -> ConfigProfile:
        """
        Load profile from file.
        
        Args:
            file_path: Profile file path
            
        Returns:
            Loaded profile
        """
        import yaml
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        profile = ConfigProfile(
            name=data["name"],
            profile_type=ProfileType(data["profile_type"]),
            environment=Environment(data["environment"]),
            description=data.get("description", ""),
            settings=data.get("settings", {}),
            overrides=data.get("overrides", {}),
            variables=data.get("variables", {}),
            dependencies=data.get("dependencies", [])
        )
        
        self.add_profile(profile)
        return profile


def create_profile(
    name: str,
    environment: Environment,
    settings: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    description: str = ""
) -> ConfigProfile:
    """
    Create a configuration profile.
    
    Args:
        name: Profile name
        environment: Target environment
        settings: Base settings
        overrides: Environment-specific overrides
        description: Profile description
        
    Returns:
        Created profile
    """
    profile = ConfigProfile(
        name=name,
        profile_type=ProfileType.CUSTOM,
        environment=environment,
        description=description,
        settings=settings or {},
        overrides=overrides or {}
    )
    
    return profile


def create_environment_profile(environment: Environment) -> EnvironmentProfile:
    """
    Create template for specific environment.
    
    Args:
        environment: Target environment
        
    Returns:
        Environment-specific template
    """
    return EnvironmentProfile(environment)


def get_default_profile(environment: Environment) -> ConfigProfile:
    """
    Get default profile for environment.
    
    Args:
        environment: Target environment
        
    Returns:
        Default environment profile
    """
    manager = ProfileManager()
    return manager.get_environment_profile(environment)


def apply_profile_to_config(config: AppConfig, profile_name: str) -> AppConfig:
    """
    Apply profile to configuration.
    
    Args:
        config: Base configuration
        profile_name: Profile name to apply
        
    Returns:
        Modified configuration
    """
    manager = ProfileManager()
    profile = manager.get_profile(profile_name)
    
    if profile is None:
        raise ValueError(f"Profile not found: {profile_name}")
    
    return profile.apply_to_config(config)
