import pytest

from policy_simulation.backend.app.schemas.policy_case import (
    PolicyCase,
    PolicyCosts,
    PolicyProvisions,
    PolicyTimeline,
)
from policy_simulation.backend.app.services.cost_estimator import CostEstimator


@pytest.fixture
def estimator():
    return CostEstimator()


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
            sme_cost_per_system=12000,
            penalty_max_percent_turnover=6.0,
            penalty_max_fixed_eur=35000000,
            source_gdp_per_capita_ppp=45000,
            cost_source="EC Impact Assessment SWD(2021) 84",
        ),
        sectors_affected=["healthcare", "education"],
        key_requirements=["Conformity assessment"],
        tags=["risk-based"],
        provisions=PolicyProvisions(sandbox=True, rights_provisions=True),
        embedding_text="EU AI Act",
    )


@pytest.fixture
def voluntary_policy():
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
            sme_cost_per_system=0,
            source_gdp_per_capita_ppp=116500,
            cost_source="PDPC 2nd Edition 2020",
        ),
        sectors_affected=["finance"],
        key_requirements=["Internal governance"],
        tags=["voluntary"],
        provisions=PolicyProvisions(),
        embedding_text="Singapore framework",
    )


def test_eu_ai_act_cost_estimation(estimator, eu_ai_act_policy):
    result = estimator.estimate(eu_ai_act_policy)
    assert result is not None
    # EUR 52227 * 1.08 (EUR→USD) * (10305/45000) PPP * 10 (USD→MAD)
    # ≈ 52227 * 1.08 * 0.229 * 10 ≈ 129,131 MAD
    assert 120_000 < result.annual_cost_per_system_mad < 140_000
    assert result.ppp_adjustment_factor == pytest.approx(10305 / 45000, rel=0.01)
    assert result.source_policy == "EU Artificial Intelligence Act"
    assert result.sme_cost_per_system_mad > 0


def test_voluntary_policy_zero_cost(estimator, voluntary_policy):
    result = estimator.estimate(voluntary_policy)
    assert result is not None
    assert result.annual_cost_per_system_mad == 0
    assert result.annual_cost_per_system_usd == 0
    assert result.sme_cost_per_system_mad == 0
    assert "voluntary" in result.methodology.lower() or "no mandatory" in result.methodology.lower()


def test_ppp_factor_range(estimator, eu_ai_act_policy):
    result = estimator.estimate(eu_ai_act_policy)
    assert 0.1 < result.ppp_adjustment_factor < 0.5
