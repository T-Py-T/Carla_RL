"""
Configuration hot-reloading without service restart.

Provides automatic configuration reloading when files change,
with callback support for handling configuration updates.
"""

import time
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .settings import AppConfig, BaseConfig
from .loader import ConfigLoader

# Type alias for hot reload callbacks
HotReloadCallback = Callable[[BaseConfig], None]


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration files."""
    
    def __init__(self, hot_reloader: "ConfigHotReloader"):
        """
        Initialize file handler.
        
        Args:
            hot_reloader: Hot reloader instance
        """
        self.hot_reloader = hot_reloader
        self.last_modified = {}  # Track last modification time
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self._is_config_file(event.src_path):
            self._handle_file_change(event.src_path)
    
    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory and self._is_config_file(event.dest_path):
            self._handle_file_change(event.dest_path)
    
    def _is_config_file(self, file_path: str) -> bool:
        """Check if file is a configuration file."""
        path = Path(file_path)
        return path.suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".env"}
    
    def _handle_file_change(self, file_path: str):
        """Handle configuration file change."""
        current_time = time.time()
        
        # Debounce rapid file changes
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 1.0:  # 1 second debounce
                return
        
        self.last_modified[file_path] = current_time
        
        # Trigger reload
        self.hot_reloader._reload_config()


class ConfigHotReloader:
    """Configuration hot-reloader with file watching."""
    
    def __init__(
        self,
        config_loader: ConfigLoader,
        config_class: type = AppConfig,
        watch_dirs: Optional[List[Union[str, Path]]] = None,
        debounce_time: float = 1.0
    ):
        """
        Initialize hot-reloader.
        
        Args:
            config_loader: Configuration loader instance
            config_class: Configuration class to instantiate
            watch_dirs: Directories to watch for changes
            debounce_time: Time to wait before reloading (seconds)
        """
        self.config_loader = config_loader
        self.config_class = config_class
        self.debounce_time = debounce_time
        
        # Current configuration
        self._current_config: Optional[BaseConfig] = None
        self._config_lock = threading.RLock()
        
        # Callbacks
        self._callbacks: List[HotReloadCallback] = []
        self._callback_lock = threading.Lock()
        
        # File watching
        self._observer: Optional[Observer] = None
        self._watch_dirs = watch_dirs or [config_loader.config_dir]
        self._watch_dirs = [Path(d) for d in self._watch_dirs]
        
        # Reload state
        self._reload_timer: Optional[threading.Timer] = None
        self._reload_lock = threading.Lock()
        
        # Error handling
        self._last_error: Optional[Exception] = None
        self._error_callbacks: List[Callable[[Exception], None]] = []
    
    def start(self) -> None:
        """Start hot-reloading."""
        if self._observer is not None:
            return  # Already started
        
        # Load initial configuration
        try:
            self._current_config = self.config_loader.load_config(self.config_class)
        except Exception as e:
            self._last_error = e
            raise
        
        # Start file watcher
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        
        for watch_dir in self._watch_dirs:
            if watch_dir.exists():
                self._observer.schedule(handler, str(watch_dir), recursive=True)
        
        self._observer.start()
    
    def stop(self) -> None:
        """Stop hot-reloading."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        
        # Cancel any pending reload
        with self._reload_lock:
            if self._reload_timer is not None:
                self._reload_timer.cancel()
                self._reload_timer = None
    
    def get_config(self) -> Optional[BaseConfig]:
        """
        Get current configuration.
        
        Returns:
            Current configuration object or None if not loaded
        """
        with self._config_lock:
            return self._current_config
    
    def add_callback(self, callback: HotReloadCallback) -> None:
        """
        Add configuration change callback.
        
        Args:
            callback: Function to call when configuration changes
        """
        with self._callback_lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: HotReloadCallback) -> None:
        """
        Remove configuration change callback.
        
        Args:
            callback: Callback to remove
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Add error callback for handling reload errors.
        
        Args:
            callback: Function to call when reload fails
        """
        self._error_callbacks.append(callback)
    
    def get_last_error(self) -> Optional[Exception]:
        """
        Get last reload error.
        
        Returns:
            Last exception that occurred during reload
        """
        return self._last_error
    
    def _reload_config(self) -> None:
        """Reload configuration (called by file handler)."""
        with self._reload_lock:
            # Cancel existing timer
            if self._reload_timer is not None:
                self._reload_timer.cancel()
            
            # Set new timer
            self._reload_timer = threading.Timer(
                self.debounce_time,
                self._perform_reload
            )
            self._reload_timer.start()
    
    def _perform_reload(self) -> None:
        """Perform the actual configuration reload."""
        try:
            # Load new configuration
            new_config = self.config_loader.load_config(self.config_class)
            
            # Update current configuration
            with self._config_lock:
                self._current_config = new_config
            
            # Clear last error
            self._last_error = None
            
            # Notify callbacks
            with self._callback_lock:
                for callback in self._callbacks:
                    try:
                        callback(new_config)
                    except Exception as e:
                        # Log callback error but don't fail the reload
                        print(f"Error in hot-reload callback: {e}")
            
        except Exception as e:
            self._last_error = e
            
            # Notify error callbacks
            for callback in self._error_callbacks:
                try:
                    callback(e)
                except Exception as callback_error:
                    print(f"Error in error callback: {callback_error}")
    
    def force_reload(self) -> None:
        """Force immediate configuration reload."""
        self._perform_reload()
    
    def is_watching(self) -> bool:
        """
        Check if hot-reloader is currently watching files.
        
        Returns:
            True if watching, False otherwise
        """
        return self._observer is not None and self._observer.is_alive()
    
    def get_watched_directories(self) -> List[Path]:
        """
        Get list of watched directories.
        
        Returns:
            List of watched directory paths
        """
        return self._watch_dirs.copy()
    
    def add_watch_directory(self, directory: Union[str, Path]) -> None:
        """
        Add directory to watch list.
        
        Args:
            directory: Directory to watch
        """
        directory = Path(directory)
        if directory not in self._watch_dirs:
            self._watch_dirs.append(directory)
            
            # Add to observer if running
            if self._observer is not None and directory.exists():
                handler = ConfigFileHandler(self)
                self._observer.schedule(handler, str(directory), recursive=True)
    
    def remove_watch_directory(self, directory: Union[str, Path]) -> None:
        """
        Remove directory from watch list.
        
        Args:
            directory: Directory to stop watching
        """
        directory = Path(directory)
        if directory in self._watch_dirs:
            self._watch_dirs.remove(directory)
            # Note: Cannot remove from observer while running, would need restart


def create_hot_reloader(
    config_file: Optional[Union[str, Path]] = None,
    env_prefix: str = "",
    config_class: type = AppConfig,
    config_dir: Optional[Path] = None,
    watch_dirs: Optional[List[Union[str, Path]]] = None,
    debounce_time: float = 1.0
) -> ConfigHotReloader:
    """
    Create a pre-configured hot-reloader.
    
    Args:
        config_file: Configuration file path
        env_prefix: Environment variable prefix
        config_class: Configuration class to instantiate
        config_dir: Configuration directory
        watch_dirs: Directories to watch for changes
        debounce_time: Time to wait before reloading (seconds)
        
    Returns:
        Configured ConfigHotReloader instance
    """
    from .loader import create_config_loader
    
    # Create config loader
    config_files = [config_file] if config_file else None
    loader = create_config_loader(config_dir, env_prefix, config_files)
    
    # Create hot reloader
    return ConfigHotReloader(
        config_loader=loader,
        config_class=config_class,
        watch_dirs=watch_dirs,
        debounce_time=debounce_time
    )


class HotReloadManager:
    """Manager for multiple hot-reloaders."""
    
    def __init__(self):
        """Initialize hot-reload manager."""
        self._reloaders: Dict[str, ConfigHotReloader] = {}
        self._lock = threading.Lock()
    
    def add_reloader(self, name: str, reloader: ConfigHotReloader) -> None:
        """
        Add hot-reloader to manager.
        
        Args:
            name: Name for the reloader
            reloader: Hot-reloader instance
        """
        with self._lock:
            self._reloaders[name] = reloader
    
    def get_reloader(self, name: str) -> Optional[ConfigHotReloader]:
        """
        Get hot-reloader by name.
        
        Args:
            name: Reloader name
            
        Returns:
            Hot-reloader instance or None if not found
        """
        with self._lock:
            return self._reloaders.get(name)
    
    def start_all(self) -> None:
        """Start all hot-reloaders."""
        with self._lock:
            for reloader in self._reloaders.values():
                reloader.start()
    
    def stop_all(self) -> None:
        """Stop all hot-reloaders."""
        with self._lock:
            for reloader in self._reloaders.values():
                reloader.stop()
    
    def remove_reloader(self, name: str) -> None:
        """
        Remove hot-reloader from manager.
        
        Args:
            name: Reloader name
        """
        with self._lock:
            if name in self._reloaders:
                self._reloaders[name].stop()
                del self._reloaders[name]
    
    def list_reloaders(self) -> List[str]:
        """
        List all reloader names.
        
        Returns:
            List of reloader names
        """
        with self._lock:
            return list(self._reloaders.keys())
