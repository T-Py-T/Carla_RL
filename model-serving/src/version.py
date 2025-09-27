"""
Version management and metadata for CarlaRL Policy-as-a-Service.

This module centralizes version information and git metadata
for consistent reporting across the application.
"""

import os
import subprocess
from typing import Optional

# Application metadata
APP_NAME = "carla-rl-serving"
MODEL_NAME = "carla-ppo"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1.0")

# Git information
def get_git_sha() -> str:
    """Get current git commit SHA, fallback to environment variable."""
    try:
        # Try to get git SHA from git command
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Fallback to environment variable
    return os.getenv("GIT_SHA", "unknown")

GIT_SHA = get_git_sha()

# Build information
BUILD_DATE = os.getenv("BUILD_DATE", "unknown")
BUILD_NUMBER = os.getenv("BUILD_NUMBER", "unknown")

# Version info dictionary for easy access
VERSION_INFO = {
    "app_name": APP_NAME,
    "model_name": MODEL_NAME, 
    "model_version": MODEL_VERSION,
    "git_sha": GIT_SHA,
    "build_date": BUILD_DATE,
    "build_number": BUILD_NUMBER
}


def get_version_string() -> str:
    """Get formatted version string."""
    return f"{APP_NAME} {MODEL_VERSION} ({GIT_SHA})"


def get_full_version_info() -> dict:
    """Get complete version information dictionary."""
    return VERSION_INFO.copy()
