from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# --- Concept Models ---

class ConceptQuestion(BaseModel):
    question: str
    difficulty: Optional[str] = None


class RelatedConcept(BaseModel):
    id: str
    term: str
    relevance: str = ""


class ConceptAnswer(BaseModel):
    question: str
    answer: str
    related_concepts: List[RelatedConcept] = []
    sources: List[str] = []
    processing_time_ms: int = 0


class ConceptSummary(BaseModel):
    id: str
    term: str
    definition: str
    difficulty: str = "intermediate"
    categories: List[str] = []


class ConceptDetail(BaseModel):
    id: str
    term: str
    definition: str
    simple_explanation: str
    examples: List[str] = []
    morocco_context: str = ""
    policy_relevance: str = ""
    related_concepts: List[str] = []
    metadata: Dict[str, Any] = {}


# --- Case Study Models ---

class CaseStudySearchQuery(BaseModel):
    query: str
    country: Optional[str] = None
    policy_type: Optional[str] = None
    top_k: int = 5


class CaseStudySummary(BaseModel):
    id: str
    country: str
    policy_name: str
    policy_type: str
    enacted_date: str
    data_quality: str = "medium"
    tags: List[str] = []
    relevance_score: Optional[float] = None


class CaseStudyDetail(BaseModel):
    id: str
    country: str
    policy: Dict[str, Any]
    outcomes: Dict[str, Any]
    metadata: Dict[str, Any]


class CaseStudyCompareRequest(BaseModel):
    ids: List[str]


class FindSimilarRequest(BaseModel):
    description: str
    policy_type: Optional[str] = None
    top_k: int = 5


# --- Simulator Models ---

class PolicyProposal(BaseModel):
    policy_name: str
    description: str
    sectors: List[str] = Field(default_factory=list)
    policy_type: str = "comprehensive"


class ImpactDimension(BaseModel):
    name: str
    score: float
    confidence: str = "medium"
    explanation: str = ""
    evidence_countries: List[str] = []


class ImpactPrediction(BaseModel):
    policy_name: str
    executive_summary: str = ""
    full_analysis: str = ""
    similar_policies: List[CaseStudySummary] = []
    impact_dimensions: List[ImpactDimension] = []
    recommendations: List[str] = []
    evidence_base_size: int = 0
    processing_time_ms: int = 0


class PolicyTemplate(BaseModel):
    id: str
    name: str
    description: str
    sectors: List[str]
    policy_type: str


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    search_engine: str = "bm25"
    collections: Dict[str, int] = {}
