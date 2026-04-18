"""
Shared pytest fixtures for the model-serving test suite.

The `client` / `mock_app_state` / `mock_inference_engine` fixtures live here
(rather than in `test_api_endpoints.py`) so any test module that needs a
pre-wired FastAPI test client - notably `test_qa_plan.py` - can pull them in
automatically without duplicating the patching dance.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_inference_engine():
    """Return a ``Mock`` that stands in for the real ``InferenceEngine``."""
    engine = Mock()
    engine.device = "cpu"
    engine.predict = Mock(return_value=([Mock(throttle=0.7, brake=0.0, steer=0.1)], 8.5))
    return engine


@pytest.fixture
def mock_app_state(mock_inference_engine):
    """Mock the module-level ``app_state`` dict used by `src.server`."""
    return {
        "model_loaded": True,
        "inference_engine": mock_inference_engine,
        "startup_time": 1695825600.0,
        "warmup_completed": True,
        "selected_version": "v0.1.0",
    }


@pytest.fixture
def client(mock_app_state, mock_inference_engine):
    """Create a FastAPI ``TestClient`` with fully-mocked dependencies.

    The fixture also resets the global health-checker singleton so its
    cached reference to ``app_state`` matches the per-test mock.
    """
    from src.monitoring.health import initialize_health_checker

    initialize_health_checker(mock_app_state)

    with (
        patch("src.server.app_state", mock_app_state),
        patch("src.server.get_inference_engine", return_value=mock_inference_engine),
    ):
        from src.server import app

        yield TestClient(app)
