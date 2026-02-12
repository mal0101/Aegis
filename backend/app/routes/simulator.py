import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import PolicyProposal, ImpactPrediction, PolicyTemplate, ImpactDimension, CaseStudySummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/simulate", tags=["simulator"])


@router.post("/predict", response_model=ImpactPrediction)
async def predict_impact(proposal: PolicyProposal):
    """Predict multi-dimensional impact of a proposed policy on Morocco."""
    from app.services.impact_service import impact_service

    try:
        result = impact_service.predict(
            policy_name=proposal.policy_name,
            description=proposal.description,
            sectors=proposal.sectors,
            policy_type=proposal.policy_type
        )
        return ImpactPrediction(
            policy_name=result["policy_name"],
            executive_summary=result["executive_summary"],
            full_analysis=result["full_analysis"],
            similar_policies=[CaseStudySummary(**sp) for sp in result["similar_policies"]],
            impact_dimensions=[ImpactDimension(**d) for d in result["impact_dimensions"]],
            recommendations=result["recommendations"],
            evidence_base_size=result["evidence_base_size"],
            processing_time_ms=result["processing_time_ms"]
        )
    except Exception as e:
        logger.error(f"Impact prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=list[PolicyTemplate])
async def get_templates():
    """Get predefined policy templates for quick simulation."""
    from app.services.impact_service import impact_service

    return [PolicyTemplate(**t) for t in impact_service.get_templates()]
