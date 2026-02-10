from pydantic import BaseModel
from typing import Optional, List, Literal


class PolicyTimeline(BaseModel):
    announced: Optional[str] = None
    enacted: Optional[str] = None
    full_enforcement: Optional[str] = None
    announcement_to_enactment_years: Optional[float] = None
    enactment_to_enforcement_years: Optional[float] = None
    total_years: Optional[float] = None


class PolicyCosts(BaseModel):
    currency: str
    annual_compliance_cost_per_system: float = 0
    compliance_breakdown: Optional[dict] = None
    sme_cost_per_system: float = 0
    penalty_max_percent_turnover: float = 0
    penalty_max_fixed_eur: float = 0
    source_gdp_per_capita_ppp: float
    cost_source: str


class PolicyProvisions(BaseModel):
    sandbox: bool = False
    rights_provisions: bool = False
    explainability_requirement: bool = False
    audit_requirement: bool = False


class PolicyAdoptionMetrics(BaseModel):
    compliance_rate: Optional[float] = None
    industry_support_level: Optional[str] = None


class PolicyCase(BaseModel):
    id: str
    name: str
    jurisdiction: str
    jurisdiction_iso: str
    description: str
    approach: Literal["mandatory-horizontal", "mandatory-sectoral", "voluntary", "principles-based"]
    status: Literal["enacted", "in-progress", "dead", "voluntary"]
    timeline: PolicyTimeline
    costs: PolicyCosts
    sectors_affected: List[str]
    key_requirements: List[str]
    tags: List[str]
    provisions: PolicyProvisions
    embedding_text: str
