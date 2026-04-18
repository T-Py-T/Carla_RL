#!/usr/bin/env python3
"""Cross-platform compatibility tests for the `highway_rl` training stack.

Historically this file was copied from a legacy Carla project and still
referenced a `carla_rl` package that no longer exists. The assertions about
valid architectures also omitted Linux aarch64, which is what containers on
Apple Silicon report. The tests below have been rewritten to reflect the
actual package (`highway_rl`) and to accept every architecture we actually
ship to.
"""

import os
import platform
import sys
import unittest

# Make `src/highway_rl` importable when the file is run directly via
# `python tests/test_cross_platform.py` from the model-sim directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Include aarch64 so Linux/ARM containers (Apple Silicon hosts, AWS Graviton,
# etc.) pass the detection check.
SUPPORTED_ARCHES = {"x86_64", "amd64", "arm64", "aarch64", "i386", "i686"}


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Baseline platform and dependency checks."""

    def test_platform_detection(self):
        """Platform detection returns a supported system/arch pair."""
        system = platform.system()
        machine = platform.machine().lower()

        self.assertIn(system, {"Darwin", "Linux", "Windows"})
        self.assertIn(machine, SUPPORTED_ARCHES)

    def test_tensorflow_import(self):
        """TensorFlow imports and can evaluate a trivial graph."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed in this environment")

        self.assertIsNotNone(tf.__version__)
        tensor = tf.constant([1, 2, 3])
        self.assertEqual(tensor.numpy().tolist(), [1, 2, 3])

    def test_gpu_detection(self):
        """Device enumeration works without crashing on CPU-only hosts."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed in this environment")

        physical_devices = tf.config.list_physical_devices()
        self.assertIsInstance(physical_devices, list)

        gpu_devices = tf.config.list_physical_devices("GPU")
        self.assertIsInstance(gpu_devices, list)
        for gpu in gpu_devices:
            self.assertIsNotNone(gpu.name)

    def test_basic_dependencies(self):
        """Core third-party deps load and behave sanely."""
        import numpy as np
        import psutil
        import colorama

        arr = np.array([1, 2, 3])
        self.assertEqual(arr.sum(), 6)

        self.assertGreater(psutil.virtual_memory().total, 0)
        self.assertIsNotNone(colorama.__version__)

        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed in this environment")
        self.assertIsNotNone(cv2.__version__)

    def test_highway_rl_imports(self):
        """Public API of the `highway_rl` package is importable."""
        from highway_rl import (
            HighwayDQNAgent,
            HighwayEnvironment,
            HighwayTrainer,
            WandBLogger,
        )

        # Simply referencing the classes is enough to verify their modules
        # import cleanly on this platform.
        for cls in (HighwayDQNAgent, HighwayEnvironment, HighwayTrainer, WandBLogger):
            self.assertTrue(callable(cls))

    def test_environment_creation(self):
        """A HighwayEnvironment can be instantiated on this platform."""
        try:
            from highway_rl import HighwayEnvironment
        except ImportError:
            self.skipTest("highway_rl not importable in this environment")

        env = HighwayEnvironment(scenario="highway")
        try:
            self.assertIsNotNone(env.observation_space)
            self.assertIsNotNone(env.action_space)
        finally:
            env.close()


class TestPlatformSpecificFeatures(unittest.TestCase):
    """Optional platform-specific probes. These skip cleanly when unmet."""

    def test_apple_silicon_detection(self):
        """Metal GPU is reported on Darwin/arm64 when available."""
        if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
            self.skipTest("Not running on Apple Silicon")

        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed in this environment")

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            self.skipTest("No GPU visible to TensorFlow (CPU-only wheel)")

        details = tf.config.experimental.get_device_details(gpus[0])
        self.assertEqual(details.get("device_name"), "METAL")

    def test_cuda_detection(self):
        """On Linux/Windows, CUDA devices (if any) enumerate successfully."""
        if platform.system() not in {"Linux", "Windows"}:
            self.skipTest("CUDA probe only runs on Linux/Windows")

        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed in this environment")

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            self.skipTest("CUDA not available on this host")
        self.assertGreater(len(gpus), 0)


def run_platform_tests() -> bool:
    """Run all platform compatibility tests and report success."""
    print(
        f"Running cross-platform tests on {platform.system()} "
        f"{platform.machine()}"
    )

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCrossPlatformCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestPlatformSpecificFeatures))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_platform_tests()
    sys.exit(0 if success else 1)
