import pytest

from policy_simulation.backend.app.schemas.policy_case import (
    PolicyCase,
    PolicyCosts,
    PolicyProvisions,
    PolicyTimeline,
)
from policy_simulation.backend.app.services.impact_analyzer import ImpactAnalyzer


@pytest.fixture
def analyzer():
    return ImpactAnalyzer()


@pytest.fixture
def mandatory_policy():
    return PolicyCase(
        id="eu-ai-act",
        name="EU Artificial Intelligence Act",
        jurisdiction="European Union",
        jurisdiction_iso="EU",
        description="EU AI Act",
        approach="mandatory-horizontal",
        status="enacted",
        timeline=PolicyTimeline(
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
            cost_source="EC Impact Assessment",
        ),
        sectors_affected=["healthcare", "education", "finance"],
        key_requirements=["Conformity assessment"],
        tags=["risk-based"],
        provisions=PolicyProvisions(
            sandbox=True,
            rights_provisions=True,
            explainability_requirement=True,
            audit_requirement=True,
        ),
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
        timeline=PolicyTimeline(announcement_to_enactment_years=1.0, total_years=1.0),
        costs=PolicyCosts(
            currency="SGD",
            annual_compliance_cost_per_system=0,
            source_gdp_per_capita_ppp=116500,
            cost_source="PDPC 2nd Edition 2020",
        ),
        sectors_affected=["finance", "healthcare"],
        key_requirements=["Internal governance"],
        tags=["voluntary"],
        provisions=PolicyProvisions(sandbox=True),
        embedding_text="Singapore framework",
    )


def test_mandatory_negative_economic(analyzer, mandatory_policy):
    result = analyzer.analyze(mandatory_policy)
    economic = next(d for d in result.dimensions if d.dimension == "economic")
    assert economic.score < 0, "Mandatory-horizontal should have negative economic impact"


def test_mandatory_positive_rights(analyzer, mandatory_policy):
    result = analyzer.analyze(mandatory_policy)
    rights = next(d for d in result.dimensions if d.dimension == "rights_protection")
    assert rights.score > 0, "Mandatory-horizontal with rights provisions should be positive"


def test_voluntary_positive_economic(analyzer, voluntary_policy):
    result = analyzer.analyze(voluntary_policy)
    economic = next(d for d in result.dimensions if d.dimension == "economic")
    assert economic.score > 0, "Voluntary approach should have positive economic impact"


def test_voluntary_low_rights(analyzer, voluntary_policy):
    result = analyzer.analyze(voluntary_policy)
    rights = next(d for d in result.dimensions if d.dimension == "rights_protection")
    assert rights.score < 0.3, "Voluntary without rights provisions should have low rights score"


def test_eu_alignment_bonus(analyzer, mandatory_policy):
    result = analyzer.analyze(mandatory_policy)
    intl = next(d for d in result.dimensions if d.dimension == "international_alignment")
    assert intl.score >= 0.4, "EU policy should get strong alignment bonus"


def test_five_dimensions(analyzer, mandatory_policy):
    result = analyzer.analyze(mandatory_policy)
    assert len(result.dimensions) == 5
    dimension_names = {d.dimension for d in result.dimensions}
    assert dimension_names == {
        "economic",
        "innovation",
        "rights_protection",
        "institutional_capacity",
        "international_alignment",
    }


def test_overall_score_bounded(analyzer, mandatory_policy):
    result = analyzer.analyze(mandatory_policy)
    assert -1.0 <= result.overall_score <= 1.0


def test_sector_alignment_boosts_innovation(analyzer, mandatory_policy):
    result = analyzer.analyze(mandatory_policy)
    innovation = next(d for d in result.dimensions if d.dimension == "innovation")
    # Healthcare, education, finance are Maroc IA 2030 priority sectors → alignment bonus
    assert innovation.score > -0.2, "Sector alignment should mitigate innovation penalty"
