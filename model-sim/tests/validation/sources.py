# Compatibility layer for scripts - imports from carla_rl package
# This allows the existing scripts to work without modification

import sys
import os

# Add src to path so we can import carla_rl (adjust for new location in tests/validation/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import everything from carla_rl and make it available as sources
from carla_rl import *