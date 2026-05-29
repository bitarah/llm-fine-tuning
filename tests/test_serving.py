"""Tests for the FastAPI serving endpoints."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401
from fastapi.testclient import TestClient
from src.data.preprocessor import KNOWN_INTENTS


@pytest.fixture
def client():
    """TestClient with mocked model loader — no real model loading."""
    with patch("src.serving.model_loader.load_model"):
        with patch("src.serving.model_loader.get_model") as mock_get:
            mock_get.return_value = (MagicMock(), MagicMock())
            with patch("src.serving.model_loader.is_model_loaded", return_value=True):
                with patch("src.serving.model_loader.get_device_info", return_value={
                    "device": "cpu",
                    "framework": "pytorch",
                    "base_model_id": "microsoft/Phi-3.5-mini-instruct",
                }):
                    from src.serving.app import create_app
                    app = create_app()
                    with TestClient(app) as c:
                        yield c


def test_health_returns_200(client: TestClient) -> None:
    """Health endpoint always returns HTTP 200."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200


def test_health_response_has_required_keys(client: TestClient) -> None:
    """Health response contains all required schema keys."""
    resp = client.get("/api/v1/health")
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert "base_model_id" in data


def test_classify_returns_200_with_valid_input(client: TestClient) -> None:
    """Classify endpoint returns 200 for valid text input."""
    with patch("src.serving.router._run_inference", return_value="cancel_order"):
        resp = client.post("/api/v1/classify", json={"text": "I want to cancel my order"})
    assert resp.status_code == 200
    data = resp.json()
    assert "intent" in data
    assert "confidence" in data
    assert "approach_used" in data
    assert "latency_ms" in data


def test_classify_rejects_empty_text(client: TestClient) -> None:
    """Classify endpoint rejects empty text with 422."""
    resp = client.post("/api/v1/classify", json={"text": ""})
    assert resp.status_code == 422


def test_classify_rejects_text_over_1000_chars(client: TestClient) -> None:
    """Classify endpoint rejects text longer than 1000 characters with 422."""
    resp = client.post("/api/v1/classify", json={"text": "x" * 1001})
    assert resp.status_code == 422


def test_classify_returns_500_on_model_error(client: TestClient) -> None:
    """Classify endpoint returns 500 when inference raises an exception."""
    with patch("src.serving.router._run_inference", side_effect=RuntimeError("model failed")):
        resp = client.post("/api/v1/classify", json={"text": "cancel my order"})
    assert resp.status_code == 500


def test_intents_endpoint_returns_27_items(client: TestClient) -> None:
    """Intents endpoint returns exactly 27 intent names."""
    resp = client.get("/api/v1/intents")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 27
    assert set(data) == set(KNOWN_INTENTS)


def test_benchmark_endpoint_returns_dict(client: TestClient, tmp_path) -> None:
    """Benchmark endpoint returns the summary JSON when file exists."""
    summary = {"approaches": {"zero_shot": {"accuracy": 0.72}}}
    benchmark_file = tmp_path / "benchmark_summary.json"
    benchmark_file.write_text(json.dumps(summary))

    with patch("src.serving.router._BENCHMARK_PATH", benchmark_file):
        resp = client.get("/api/v1/benchmark")

    assert resp.status_code == 200
    assert resp.json() == summary


def test_benchmark_endpoint_returns_404_when_missing(client: TestClient) -> None:
    """Benchmark endpoint returns 404 with helpful message when file is missing."""
    missing = __import__("pathlib").Path("/nonexistent/benchmark_summary.json")
    with patch("src.serving.router._BENCHMARK_PATH", missing):
        resp = client.get("/api/v1/benchmark")
    assert resp.status_code == 404
    assert "make evaluate" in resp.json()["detail"].lower()
