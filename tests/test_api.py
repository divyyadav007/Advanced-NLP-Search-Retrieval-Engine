from fastapi.testclient import TestClient
from unittest.mock import patch
from src.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify that root index returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Enterprise Hybrid RAG" in response.text


def test_health_endpoint():
    """Verify that health check returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "engine": "enterprise_hybrid_rag"}


def test_ingest_endpoint_validation():
    """Verify endpoint validation for empty payload parameters."""
    response = client.post("/v1/ingest", json={"file_path": ""})
    assert response.status_code == 422  # Pydantic validation error due to min_length=1

    response = client.post("/v1/ingest", json={"file_path": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


@patch("src.main.app.state.parser_router")
def test_ingest_endpoint_file_not_found(mock_parser):
    """Verify endpoint error routing when file does not exist on disk."""
    # Configure mock to raise FileNotFoundError
    mock_parser.process_file.side_effect = FileNotFoundError(
        "Target track asset missing from disk: missing.txt"
    )

    response = client.post("/v1/ingest", json={"file_path": "missing.txt"})
    assert response.status_code == 404
    assert "missing from disk" in response.json()["detail"]


def test_ask_endpoint_validation():
    """Verify input validation constraints on query endpoint."""
    response = client.post("/v1/ask", json={"question": ""})
    assert response.status_code == 422  # Pydantic validation error due to min_length=1

    response = client.post("/v1/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


@patch("src.main.app.state.hybrid_retriever")
def test_ask_endpoint_empty_database(mock_retriever):
    """Verify answer fallback behavior when hybrid search database has zero indexed chunks."""
    mock_retriever.retrieve.return_value = []

    response = client.post("/v1/ask", json={"question": "What is the policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_context_sufficient"] is False
    assert "empty" in data["answer"].lower()
