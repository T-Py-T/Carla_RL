#!/usr/bin/env python3
"""
CarlaRL Log Analysis CLI Tool

This script provides command-line access to the log aggregation and analysis
utilities for monitoring and debugging the CarlaRL serving infrastructure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.log_aggregation import main

if __name__ == '__main__':
    main()
