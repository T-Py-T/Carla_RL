# Legacy CARLA RL Implementation

This directory contains the original CARLA-based reinforcement learning implementation. 

## Platform Support

- **Windows**: Full native support
- **Linux**: Full native support  
- **macOS Intel**: Limited support (requires Docker or VM)
- **macOS Apple Silicon**: Not supported (fundamental incompatibility)

## Current Status

This implementation is maintained for compatibility with Windows/Linux systems where CARLA can run natively. For macOS development, especially on Apple Silicon, use the modern highway-env implementation in `src/highway_rl/`.

## Usage

For Windows/Linux users who want to train with the full CARLA simulator:

1. Install CARLA simulator natively
2. Use the training scripts in `training/carla/`
3. Refer to the main README for setup instructions

## Migration Note

New development focuses on the highway-env implementation which provides:
- Native Apple Silicon support
- Faster iteration cycles
- Multiple driving scenarios
- Better evaluation tools
- Modern ML stack integration

The algorithms and techniques developed here can be transferred to CARLA when full simulation fidelity is required.
