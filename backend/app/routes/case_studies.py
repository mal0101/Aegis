import time
import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    CaseStudySearchQuery, CaseStudySummary, CaseStudyDetail,
    CaseStudyCompareRequest, FindSimilarRequest
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/case-studies", tags=["case-studies"])


@router.get("/", response_model=list[CaseStudySummary])
async def list_case_studies():
    """List all case study summaries."""
    from app.services.case_study_service import case_study_service

    return [CaseStudySummary(**cs) for cs in case_study_service.get_all()]


@router.post("/search", response_model=list[CaseStudySummary])
async def search_case_studies(query: CaseStudySearchQuery):
    """Search case studies with semantic + keyword matching."""
    from app.services.case_study_service import case_study_service

    try:
        results = case_study_service.search(
            query=query.query,
            country=query.country,
            policy_type=query.policy_type,
            top_k=query.top_k
        )
        return [CaseStudySummary(**r) for r in results]
    except Exception as e:
        logger.error(f"Case study search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/find-similar", response_model=list[CaseStudySummary])
async def find_similar(request: FindSimilarRequest):
    """Find case studies similar to a policy description."""
    from app.services.case_study_service import case_study_service

    try:
        results = case_study_service.find_similar(
            description=request.description,
            policy_type=request.policy_type,
            top_k=request.top_k
        )
        return [
            CaseStudySummary(
                id=r["id"],
                country=r["country"],
                policy_name=r["policy"]["name"],
                policy_type=r["policy"].get("type", ""),
                enacted_date=r["policy"].get("enacted_date", ""),
                data_quality=r["outcomes"].get("data_quality", "medium"),
                tags=r["metadata"].get("tags", []),
                relevance_score=r.get("relevance_score")
            )
            for r in results
        ]
    except Exception as e:
        logger.error(f"Find similar failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=list[CaseStudyDetail])
async def compare_case_studies(request: CaseStudyCompareRequest):
    """Compare multiple case studies side by side."""
    from app.services.case_study_service import case_study_service

    results = case_study_service.compare(request.ids)
    if not results:
        raise HTTPException(status_code=404, detail="No matching case studies found")
    return [CaseStudyDetail(**r) for r in results]


@router.get("/{study_id}", response_model=CaseStudyDetail)
async def get_case_study(study_id: str):
    """Get full details for a specific case study."""
    from app.services.case_study_service import case_study_service

    cs = case_study_service.get_by_id(study_id)
    if not cs:
        raise HTTPException(status_code=404, detail=f"Case study '{study_id}' not found")
    return CaseStudyDetail(**cs)
