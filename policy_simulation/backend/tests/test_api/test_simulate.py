import sys
from unittest.mock import MagicMock, patch

import pytest

from policy_simulation.backend.app.schemas.simulation_result import (
    CostEstimate,
    DimensionScore,
    ImpactAnalysis,
    MatchedPolicy,
    SimulationResult,
    TimelineEstimate,
)

# Mock chromadb before any import of main triggers the full chain
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()

# Mock sentence_transformers and concept_translator services to avoid heavy ML imports
_mock_st = MagicMock()
sys.modules.setdefault("sentence_transformers", _mock_st)

# Now safe to import main
from main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


@pytest.fixture
def mock_simulation_result():
    return SimulationResult(
        policy_description="Test policy",
        matched_policies=[
            MatchedPolicy(
                id="eu-ai-act",
                name="EU AI Act",
                jurisdiction="European Union",
                approach="mandatory-horizontal",
                similarity_score=0.85,
            )
        ],
        cost_estimate=CostEstimate(
            annual_cost_per_system_mad=129170.0,
            annual_cost_per_system_usd=12917.0,
            sme_cost_per_system_mad=29700.0,
            sme_cost_per_system_usd=2970.0,
            ppp_adjustment_factor=0.229,
            source_policy="EU AI Act",
            methodology="PPP-adjusted",
        ),
        timeline_estimate=TimelineEstimate(
            announcement_to_enactment_years=5.7,
            enactment_to_enforcement_years=4.2,
            total_years=9.9,
            readiness_adjustment_factor=1.795,
            source_policy="EU AI Act",
            methodology="Readiness-adjusted",
        ),
        impact_analysis=ImpactAnalysis(
            dimensions=[
                DimensionScore(dimension="economic", score=-0.5, explanation="test"),
                DimensionScore(dimension="innovation", score=0.1, explanation="test"),
                DimensionScore(dimension="rights_protection", score=0.8, explanation="test"),
                DimensionScore(dimension="institutional_capacity", score=-0.3, explanation="test"),
                DimensionScore(dimension="international_alignment", score=0.5, explanation="test"),
            ],
            overall_score=0.1,
            methodology="Rule-based",
        ),
        narrative="Test narrative for Morocco.",
        data_sources=["World Bank WDI 2024"],
    )


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "The Bridge"


@patch("policy_simulation.backend.app.api.routes.simulate.get_engine")
def test_simulate_endpoint(mock_get_engine, mock_simulation_result):
    mock_engine = MagicMock()
    mock_engine.simulate.return_value = mock_simulation_result
    mock_get_engine.return_value = mock_engine

    response = client.post(
        "/api/v1/simulate",
        json={"policy_description": "Require explainability for high-risk AI in healthcare"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["matched_policies"]) == 1
    assert data["matched_policies"][0]["id"] == "eu-ai-act"
    assert data["cost_estimate"]["annual_cost_per_system_mad"] == 129170.0
    assert data["timeline_estimate"]["total_years"] == 9.9
    assert len(data["impact_analysis"]["dimensions"]) == 5
    assert data["narrative"] == "Test narrative for Morocco."


@patch("policy_simulation.backend.app.api.routes.simulate.get_engine")
def test_simulate_validation_error(mock_get_engine):
    response = client.post("/api/v1/simulate", json={"policy_description": "short"})
    assert response.status_code == 422


@patch("policy_simulation.backend.app.api.routes.simulate.get_engine")
def test_simulate_with_options(mock_get_engine, mock_simulation_result):
    mock_engine = MagicMock()
    mock_engine.simulate.return_value = mock_simulation_result
    mock_get_engine.return_value = mock_engine

    response = client.post(
        "/api/v1/simulate",
        json={
            "policy_description": "Voluntary AI ethics guidelines for agriculture sector",
            "target_sectors": ["agriculture"],
            "num_matches": 2,
        },
    )
    assert response.status_code == 200
    mock_engine.simulate.assert_called_once_with(
        policy_description="Voluntary AI ethics guidelines for agriculture sector",
        target_sectors=["agriculture"],
        num_matches=2,
    )
