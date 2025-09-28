"""
Unit tests for configuration loading.

Tests hierarchical configuration loading with environment variables,
files, and defaults.
"""

import json
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.loader import ConfigLoader, load_config, create_config_loader
from src.config.settings import AppConfig, Environment


class TestConfigLoader:
    """Test ConfigLoader functionality."""
    
    def test_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader.config_dir.exists()
        assert ".json" in loader.supported_formats
        assert ".yaml" in loader.supported_formats
        assert ".yml" in loader.supported_formats
        assert ".toml" in loader.supported_formats
        assert ".env" in loader.supported_formats
    
    def test_custom_config_dir(self):
        """Test ConfigLoader with custom config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(Path(temp_dir))
            assert loader.config_dir == Path(temp_dir)
            assert loader.config_dir.exists()
    
    def test_add_file_source(self):
        """Test adding file sources."""
        loader = ConfigLoader()
        
        # Add existing file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "value"}, f)
            temp_file = Path(f.name)
        
        try:
            loader.add_file_source(temp_file, required=True)
            assert len(loader.sources) == 1
            assert loader.sources[0] == ("file", str(temp_file))
        finally:
            temp_file.unlink()
    
    def test_add_file_source_nonexistent_required(self):
        """Test adding required nonexistent file source."""
        loader = ConfigLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.add_file_source("nonexistent.json", required=True)
    
    def test_add_file_source_nonexistent_optional(self):
        """Test adding optional nonexistent file source."""
        loader = ConfigLoader()
        
        loader.add_file_source("nonexistent.json", required=False)
        assert len(loader.sources) == 0  # Should not add nonexistent file
    
    def test_add_env_source(self):
        """Test adding environment source."""
        loader = ConfigLoader()
        
        loader.add_env_source("TEST_")
        assert len(loader.sources) == 1
        assert loader.sources[0] == ("env", "TEST_")
    
    def test_add_env_source_with_file(self):
        """Test adding environment source with .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(Path(temp_dir))
            
            # Create .env file
            env_file = Path(temp_dir) / "test.env"
            env_file.write_text("TEST_VAR=test_value\n")
            
            loader.add_env_source("TEST_", env_file)
            assert len(loader.sources) == 2
            assert ("env_file", str(env_file)) in loader.sources
            assert ("env", "TEST_") in loader.sources
    
    def test_add_default_source(self):
        """Test adding default source."""
        loader = ConfigLoader()
        default_config = AppConfig()
        
        loader.add_default_source(default_config)
        assert len(loader.sources) == 1
        assert loader.sources[0] == ("default", default_config)
    
    def test_load_config_with_defaults(self):
        """Test loading configuration with defaults only."""
        loader = ConfigLoader()
        loader.add_default_source(AppConfig())
        
        config = loader.load_config()
        assert isinstance(config, AppConfig)
        assert config.environment == Environment.DEVELOPMENT
    
    def test_load_config_with_file(self):
        """Test loading configuration from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(Path(temp_dir))
            
            # Create config file
            config_file = Path(temp_dir) / "test.yaml"
            config_data = {
                "environment": "production",
                "debug": False,
                "security": {
                    "enabled": True
                },
                "server": {
                    "port": 9000,
                    "workers": 4
                }
            }
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            loader.add_file_source("test.yaml", required=True)
            config = loader.load_config()
            
            assert config.environment == Environment.PRODUCTION
            assert config.debug is False
            assert config.server.port == 9000
            assert config.server.workers == 4
    
    def test_load_config_with_env_vars(self):
        """Test loading configuration from environment variables."""
        loader = ConfigLoader()
        loader.add_default_source(AppConfig())
        
        with patch.dict(os.environ, {
            "APP_ENVIRONMENT": "staging",
            "APP_DEBUG": "true",
            "APP_SERVER_PORT": "8080",
            "APP_SERVER_WORKERS": "2"
        }):
            loader.add_env_source("APP_")
            config = loader.load_config()
            
            assert config.environment == Environment.STAGING
            assert config.debug is True
            assert config.server.port == 8080
            assert config.server.workers == 2
    
    def test_load_config_hierarchical_override(self):
        """Test hierarchical configuration loading with overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(Path(temp_dir))
            
            # Create config file
            config_file = Path(temp_dir) / "test.json"
            config_data = {
                "environment": "production",
                "security": {
                    "enabled": True
                },
                "server": {
                    "port": 9000
                }
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            # Set up sources: defaults -> file -> env
            loader.add_default_source(AppConfig())
            loader.add_file_source("test.json", required=True)
            
            with patch.dict(os.environ, {
                "APP_SERVER_PORT": "8080",
                "APP_DEBUG": "false"
            }):
                loader.add_env_source("APP_")
                config = loader.load_config()
                
                # Environment variables should override file and defaults
                assert config.server.port == 8080
                assert config.debug is False
                # File should override defaults
                assert config.environment == Environment.PRODUCTION
    
    def test_load_config_validation_error(self):
        """Test configuration loading with validation error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(Path(temp_dir))
            
            # Create invalid config data
            invalid_data = {
                "server": {
                    "port": 99999  # Invalid port
                }
            }
            
            # Create the file first
            invalid_file = Path(temp_dir) / "invalid.json"
            with open(invalid_file, 'w') as f:
                json.dump(invalid_data, f)
            
            loader.add_file_source("invalid.json", required=True)
            
            with pytest.raises(ValueError, match="Configuration validation failed"):
                loader.load_config()
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigLoader(Path(temp_dir))
            config = AppConfig()
            
            # Save as JSON
            json_file = Path(temp_dir) / "config.json"
            loader.save_config(config, json_file, "json")
            assert json_file.exists()
            
            # Save as YAML
            yaml_file = Path(temp_dir) / "config.yaml"
            loader.save_config(config, yaml_file, "yaml")
            assert yaml_file.exists()
            
            # Save as TOML
            toml_file = Path(temp_dir) / "config.toml"
            loader.save_config(config, toml_file, "toml")
            assert toml_file.exists()
    
    def test_convert_env_value(self):
        """Test environment variable value conversion."""
        loader = ConfigLoader()
        
        # Test boolean conversion
        assert loader._convert_env_value("true") is True
        assert loader._convert_env_value("false") is False
        assert loader._convert_env_value("yes") is True
        assert loader._convert_env_value("no") is False
        assert loader._convert_env_value("1") is True
        assert loader._convert_env_value("0") is False
        
        # Test null conversion
        assert loader._convert_env_value("null") is None
        assert loader._convert_env_value("none") is None
        assert loader._convert_env_value("") is None
        
        # Test numeric conversion
        assert loader._convert_env_value("123") == 123
        assert loader._convert_env_value("123.45") == 123.45
        
        # Test string conversion
        assert loader._convert_env_value("hello") == "hello"
        assert loader._convert_env_value("test value") == "test value"
    
    def test_set_nested_value(self):
        """Test setting nested dictionary values."""
        loader = ConfigLoader()
        data = {}
        
        # Test simple key
        loader._set_nested_value(data, "simple", "value")
        assert data["simple"] == "value"
        
        # Test nested key
        loader._set_nested_value(data, "nested.key", "nested_value")
        assert data["nested"]["key"] == "nested_value"
        
        # Test deeply nested key
        loader._set_nested_value(data, "deep.nested.structure", "deep_value")
        assert data["deep"]["nested"]["structure"] == "deep_value"
    
    def test_merge_configs(self):
        """Test configuration merging."""
        loader = ConfigLoader()
        
        base = {
            "a": 1,
            "b": {
                "x": 10,
                "y": 20
            }
        }
        
        override = {
            "b": {
                "y": 30,
                "z": 40
            },
            "c": 3
        }
        
        result = loader._merge_configs(base, override)
        
        assert result["a"] == 1  # From base
        assert result["b"]["x"] == 10  # From base
        assert result["b"]["y"] == 30  # Overridden
        assert result["b"]["z"] == 40  # From override
        assert result["c"] == 3  # From override


class TestLoadConfigFunction:
    """Test load_config convenience function."""
    
    def test_load_config_with_file(self):
        """Test load_config with file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"
            config_data = {
                "environment": "staging",
                "server": {
                    "port": 8080
                }
            }
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            config = load_config(config_file=config_file)
            assert config.environment == Environment.STAGING
            assert config.server.port == 8080
    
    def test_load_config_with_env_prefix(self):
        """Test load_config with environment prefix."""
        with patch.dict(os.environ, {
            "TEST_ENVIRONMENT": "development",
            "TEST_SERVER_PORT": "9000"
        }):
            config = load_config(env_prefix="TEST_")
            assert config.environment == Environment.DEVELOPMENT
            assert config.server.port == 9000
    
    def test_load_config_with_custom_class(self):
        """Test load_config with custom config class."""
        from src.config.settings import ServerConfig
        
        with patch.dict(os.environ, {
            "TEST_PORT": "9000"
        }):
            config = load_config(config_class=ServerConfig, env_prefix="TEST_")
            assert isinstance(config, ServerConfig)
            assert config.port == 9000


class TestCreateConfigLoader:
    """Test create_config_loader function."""
    
    def test_create_config_loader(self):
        """Test creating pre-configured loader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_files = ["config1.yaml", "config2.json"]
            
            # Create test files
            for file_name in config_files:
                file_path = Path(temp_dir) / file_name
                if file_name.endswith('.yaml'):
                    with open(file_path, 'w') as f:
                        yaml.dump({"test": "yaml"}, f)
                else:
                    with open(file_path, 'w') as f:
                        json.dump({"test": "json"}, f)
            
            loader = create_config_loader(
                config_dir=Path(temp_dir),
                env_prefix="TEST_",
                config_files=config_files
            )
            
            assert loader.config_dir == Path(temp_dir)
            assert len(loader.sources) == len(config_files) + 1  # Files + env
