"""
Integration tests for PolicyBridge API.
Run with: cd backend && python -m pytest tests/ -v

Note: These tests require the embedding model and ChromaDB to be available.
The LLM tests require Ollama to be running.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def client():
    """Create test client with mocked heavy services."""
    # Mock embedding service to avoid loading real models
    mock_embedding = MagicMock()
    mock_embedding.dim = 384
    mock_embedding._model_name = "test-model"
    mock_embedding.embed_text.return_value = [0.1] * 384
    mock_embedding.embed_batch.return_value = [[0.1] * 384]
    mock_embedding.get_collection_name.side_effect = lambda base: f"{base}_384"
    mock_embedding.is_bge_m3 = False

    with patch("app.services.embedding_service.embedding_service", mock_embedding), \
         patch("app.services.embedding_service.EmbeddingService", return_value=mock_embedding):

        from app.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    """Health endpoint returns status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_concepts_list(client):
    """Concept list returns all 10 concepts."""
    response = client.get("/api/concepts/list")
    assert response.status_code == 200
    concepts = response.json()
    assert len(concepts) == 10


def test_concept_by_id(client):
    """Get a specific concept by ID."""
    response = client.get("/api/concepts/algorithmic-bias")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "algorithmic-bias"
    assert data["term"] == "Algorithmic Bias"
    assert "morocco_context" in data


def test_concept_not_found(client):
    """Unknown concept ID returns 404."""
    response = client.get("/api/concepts/nonexistent")
    assert response.status_code == 404


def test_case_studies_list(client):
    """Case study list returns all 8 studies."""
    response = client.get("/api/case-studies/")
    assert response.status_code == 200
    studies = response.json()
    assert len(studies) == 8


def test_case_study_by_id(client):
    """Get a specific case study by ID."""
    response = client.get("/api/case-studies/eu_ai_act_2024")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "eu_ai_act_2024"
    assert data["country"] == "EU"
    assert "outcomes" in data
    assert "social_impact" in data["outcomes"]


def test_case_study_not_found(client):
    """Unknown case study ID returns 404."""
    response = client.get("/api/case-studies/nonexistent")
    assert response.status_code == 404


def test_simulator_templates(client):
    """Simulator templates return 5 templates."""
    response = client.get("/api/simulate/templates")
    assert response.status_code == 200
    templates = response.json()
    assert len(templates) == 5
    ids = [t["id"] for t in templates]
    assert "bias_testing" in ids
    assert "comprehensive_ai_act" in ids


def test_case_study_compare(client):
    """Compare endpoint returns details for requested IDs."""
    response = client.post(
        "/api/case-studies/compare",
        json={"ids": ["eu_ai_act_2024", "tunisia_ai_strategy_2023"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_data_files_valid():
    """Verify all JSON data files are valid and complete."""
    data_dir = Path(__file__).resolve().parent.parent / "app" / "data"

    with open(data_dir / "concepts.json") as f:
        concepts = json.load(f)
    assert len(concepts) == 10

    with open(data_dir / "case_studies.json") as f:
        case_studies = json.load(f)
    assert len(case_studies) == 8

    with open(data_dir / "morocco_context.json") as f:
        morocco = json.load(f)
    assert morocco["country"] == "Morocco"
    assert morocco["demographics"]["population"] == 37000000
