import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import ConceptQuestion, ConceptAnswer, ConceptSummary, ConceptDetail, RelatedConcept

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/concepts", tags=["concepts"])


@router.post("/ask", response_model=ConceptAnswer)
async def ask_concept(question: ConceptQuestion):
    """Ask a question about AI policy concepts. Returns Morocco-contextualized answer."""
    from app.services.rag_service import rag_service

    try:
        result = rag_service.answer_question(question.question, question.difficulty)
        return ConceptAnswer(
            question=result["question"],
            answer=result["answer"],
            related_concepts=[RelatedConcept(**rc) for rc in result.get("related_concepts", [])],
            sources=result.get("sources", []),
            processing_time_ms=result.get("processing_time_ms", 0)
        )
    except Exception as e:
        logger.error(f"Concept Q&A failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=list[ConceptSummary])
async def list_concepts():
    """List all available AI policy concepts."""
    from app.services.rag_service import rag_service

    concepts = rag_service.get_all_concepts()
    return [ConceptSummary(**c) for c in concepts]


@router.get("/{concept_id}", response_model=ConceptDetail)
async def get_concept(concept_id: str):
    """Get full details for a specific concept."""
    from app.services.rag_service import rag_service

    concept = rag_service.get_concept_by_id(concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_id}' not found")
    return ConceptDetail(**concept)
