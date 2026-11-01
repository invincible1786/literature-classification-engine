"""Tier 1 API unit tests using FastAPI TestClient with mocked network and StubScorer."""
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Configure environment for stub scorer before importing main
os.environ["USE_STUB_SCORER"] = "1"
os.environ["MAX_CONCURRENT_JOBS"] = "1"
os.environ["QUEUE_WAIT_S"] = "0.5"

from app.main import app
from app.retrieval import RetrievalError


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """GET /health returns healthy status and model revision."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model_revision" in data


def test_e9_zero_search_hits_returns_200(client):
    """E9: Zero search hits returns empty results with HTTP 200, not an error."""
    with patch("app.main.search_europe_pmc", return_value=[]):
        payload = {
            "entity": "nonexistent_molecule_xyz",
            "criterion": "The paper describes the synthesis of nonexistent_molecule_xyz."
        }
        res = client.post("/classify", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["total_retrieved"] == 0
        assert data["total_classified"] == 0
        assert data["results"] == []


def test_e8_europe_pmc_upstream_outage_returns_502(client):
    """E8: Upstream 503 outage at Europe PMC surfaces as HTTP 502 Bad Gateway with diagnostic."""
    with patch("app.main.search_europe_pmc", side_effect=RetrievalError("Europe PMC temporary outage (HTTP 503).")):
        payload = {
            "entity": "vanillin",
            "criterion": "The paper describes the synthesis of vanillin."
        }
        res = client.post("/classify", json=payload)
        assert res.status_code == 502
        assert "Europe PMC" in res.json()["detail"]


def test_feedback_endpoint_records_correction(client, tmp_path):
    """POST /feedback persists user corrections atomically."""
    test_feedback_file = str(tmp_path / "test_feedback.jsonl")
    with patch("app.main.os.path.join", return_value=test_feedback_file):
        payload = {
            "paper_id": "PMC1234567",
            "entity": "vanillin",
            "criterion": "The paper describes the synthesis of vanillin.",
            "suggested_label": "does_not_qualify",
            "notes": "Vanillin was used as starting material in step 2."
        }
        res = client.post("/feedback", json=payload)
        assert res.status_code == 200
        assert res.json()["status"] == "success"

