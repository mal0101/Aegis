from pydantic import BaseModel, Field
from typing import List, Optional


class MatchedPolicy(BaseModel):
    id: str
    name: str
    jurisdiction: str
    approach: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class CostEstimate(BaseModel):
    annual_cost_per_system_mad: float
    annual_cost_per_system_usd: float
    sme_cost_per_system_mad: float
    sme_cost_per_system_usd: float
    ppp_adjustment_factor: float
    source_policy: str
    methodology: str


class TimelineEstimate(BaseModel):
    announcement_to_enactment_years: Optional[float] = None
    enactment_to_enforcement_years: Optional[float] = None
    total_years: Optional[float] = None
    readiness_adjustment_factor: float
    source_policy: str
    methodology: str


class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=-1.0, le=1.0)
    explanation: str


class ImpactAnalysis(BaseModel):
    dimensions: List[DimensionScore]
    overall_score: float = Field(ge=-1.0, le=1.0)
    methodology: str


class SimulationResult(BaseModel):
    policy_description: str
    matched_policies: List[MatchedPolicy]
    cost_estimate: Optional[CostEstimate] = None
    timeline_estimate: Optional[TimelineEstimate] = None
    impact_analysis: ImpactAnalysis
    narrative: str
    data_sources: List[str]
