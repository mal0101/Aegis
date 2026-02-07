from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal

class ConceptSchema(BaseModel):
    id : str = Field(
        ...,
        description="Unique identifier (kebab-case)",
        min_length=3,
        max_length=50,
        pattern=r'^[a-z0-9]+(-[a-z0-9]+)*$'
    )
    
    term: str = Field(
        ...,
        description="Human-readable term (title case)",
        min_length=3,
        max_length=100
    )
    
    definition: str = Field(
        ...,
        description="Academic/technical definition",
        min_length=50,
        max_length=1000
    )
    
    simple_explanation: str = Field(
        ...,
        description="Layperson-friendly explanation",
        min_length=20,
        max_length=500
    )
    
    examples: List[str] = Field(
        ...,
        description="Real-world examples (min2, at least 1 Morocco-relevant)",
        min_items=2,
        max_items=5
    )
    
    policy_relevance: str = Field(
        ...,
        description="Why this matters for Morocco specifically",
        min_length=50,
        max_length=500
    )
    
    related_concepts: List[str] = Field(
        default=[],
        description="IDs of related concepts",
        max_items=5
    )
    sources: List[str] = Field(
        ...,
        description="Credible sources (URLs to papers, legislation, etc.)",
        min_items=1,
        max_items=5
    )
    difficulty_level: Literal['Beginner', 'Intermediate', 'Advanced'] = Field(
        ...,
        description="Complexity level"
    )
    
    categories: List[Literal[
        "fairness",
        "ethics",
        "transparency",
        "governance",
        "technical",
        "risk",
        "data",
        "economic"
    ]] = Field(
        ...,
        description="Concept categories for filtering",
        min_items=1,
        max_items=3
    )
    
    created_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp"
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp"
    )
    
    @field_validator('id')
    def validate_id(cls,v):
        if not v.islower():
            raise ValueError('ID must be lowercase')
        if v.startswith('-') or v.endswith('-'):
            raise ValueError('ID cannot start or end with a hyphen')
        return v
    
    @field_validator('examples')
    def validate_morocco_relevance(cls,v):
        keywords = ['morocco','maroc','africa','african','jazari','developing','rural','urban', 'arabic']
        has_relevance = any(any(keyword in example.lower() for keyword in keywords)
                            for example in v)
        if not has_relevance:
            raise ValueError('At least one example must be relevant to Morocco')
        return v
    
    @field_validator('policy_relevance')
    def validate_morocco_mention(cls,v):
        if 'morocco' not in v.lower():
            raise ValueError(
                "policy_relevance must explicitly mention Morocco.",
                "This ensures the concept is contextualized for the target audience."
            )
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "bias-algorithmic",
                "term": "Algorithmic Bias",
                "definition": "Algorithmic bias occurs when...",
                "simple_explanation": "When AI treats different groups unfairly...",
                "examples": [
                    "Amazon's hiring AI...",
                    "Facial recognition systems..."
                ],
                "policy_relevance": "Morocco must ensure AI systems...",
                "related_concepts": ["fairness-metrics", "protected-attributes"],
                "sources": ["https://example.com/paper"],
                "difficulty_level": "beginner",
                "categories": ["fairness", "ethics"]
            }
        }