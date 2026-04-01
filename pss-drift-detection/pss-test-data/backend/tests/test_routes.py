"""Tests for Healthcare Context Graph API."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def mock_neo4j():
    """Mock Neo4j connection for all tests."""
    with patch("app.context_graph_client.connect_neo4j", new_callable=AsyncMock), \
         patch("app.context_graph_client.close_neo4j", new_callable=AsyncMock), \
         patch("app.vector_client.create_vector_index", new_callable=AsyncMock):
        yield


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["domain"] == "healthcare"



def test_scenarios():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "domain" in data
    assert "scenarios" in data
    assert isinstance(data["scenarios"], list)
