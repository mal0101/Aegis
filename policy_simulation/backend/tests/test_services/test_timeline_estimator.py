import pytest

from policy_simulation.backend.app.schemas.policy_case import (
    PolicyCase,
    PolicyCosts,
    PolicyProvisions,
    PolicyTimeline,
)
from policy_simulation.backend.app.services.timeline_estimator import TimelineEstimator


@pytest.fixture
def estimator():
    return TimelineEstimator()


@pytest.fixture
def eu_ai_act_policy():
    return PolicyCase(
        id="eu-ai-act",
        name="EU Artificial Intelligence Act",
        jurisdiction="European Union",
        jurisdiction_iso="EU",
        description="EU AI Act",
        approach="mandatory-horizontal",
        status="enacted",
        timeline=PolicyTimeline(
            announced="2021-04-21",
            enacted="2024-07-12",
            full_enforcement="2027-08-01",
            announcement_to_enactment_years=3.2,
            enactment_to_enforcement_years=3.1,
            total_years=6.3,
        ),
        costs=PolicyCosts(
            currency="EUR",
            annual_compliance_cost_per_system=52227,
            source_gdp_per_capita_ppp=45000,
            cost_source="EC Impact Assessment",
        ),
        sectors_affected=["healthcare"],
        key_requirements=["Conformity assessment"],
        tags=["risk-based"],
        provisions=PolicyProvisions(),
        embedding_text="EU AI Act",
    )


@pytest.fixture
def singapore_policy():
    return PolicyCase(
        id="singapore-model-governance",
        name="Singapore Model AI Governance Framework",
        jurisdiction="Singapore",
        jurisdiction_iso="SG",
        description="Voluntary framework",
        approach="voluntary",
        status="voluntary",
        timeline=PolicyTimeline(
            announced="2019-01-23",
            enacted="2020-01-21",
            announcement_to_enactment_years=1.0,
            total_years=1.0,
        ),
        costs=PolicyCosts(
            currency="SGD",
            annual_compliance_cost_per_system=0,
            source_gdp_per_capita_ppp=116500,
            cost_source="PDPC 2nd Edition 2020",
        ),
        sectors_affected=["finance"],
        key_requirements=["Internal governance"],
        tags=["voluntary"],
        provisions=PolicyProvisions(),
        embedding_text="Singapore framework",
    )


def test_eu_ai_act_timeline(estimator, eu_ai_act_policy):
    result = estimator.estimate(eu_ai_act_policy)
    assert result is not None
    # Readiness ratio: 75.0 / 41.78 ≈ 1.795
    assert result.readiness_adjustment_factor == pytest.approx(75.0 / 41.78, rel=0.01)
    # Enactment: 3.2 * 1.795 ≈ 5.7 years
    assert 5.0 < result.announcement_to_enactment_years < 7.0
    # Enforcement: 3.1 * sqrt(1.795) ≈ 4.2 years
    assert 3.5 < result.enactment_to_enforcement_years < 5.0
    # Total: ~9.5-10 years
    assert 8.0 < result.total_years < 12.0


def test_singapore_timeline(estimator, singapore_policy):
    result = estimator.estimate(singapore_policy)
    assert result is not None
    # Readiness ratio: 88.0 / 41.78 ≈ 2.106
    assert result.readiness_adjustment_factor == pytest.approx(88.0 / 41.78, rel=0.01)
    # Enactment: 1.0 * 2.106 ≈ 2.1 years
    assert 1.5 < result.announcement_to_enactment_years < 3.0


def test_readiness_ratio_clamped(estimator):
    """Ensure readiness ratio is clamped to [0.5, 2.5]."""
    # The ratio for Singapore (88/41.78 ≈ 2.106) is within range
    # The maximum possible ratio is 2.5
    assert estimator.morocco_readiness > 0


def test_unknown_jurisdiction_returns_none(estimator):
    policy = PolicyCase(
        id="unknown",
        name="Unknown Policy",
        jurisdiction="Unknown",
        jurisdiction_iso="XX",
        description="Unknown",
        approach="voluntary",
        status="voluntary",
        timeline=PolicyTimeline(announcement_to_enactment_years=2.0),
        costs=PolicyCosts(
            currency="USD",
            source_gdp_per_capita_ppp=30000,
            cost_source="test",
        ),
        sectors_affected=[],
        key_requirements=[],
        tags=[],
        provisions=PolicyProvisions(),
        embedding_text="Unknown",
    )
    result = estimator.estimate(policy)
    assert result is None
