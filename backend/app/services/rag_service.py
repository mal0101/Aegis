import json
import time
import logging
from typing import List, Dict, Any

from app.config import MOROCCO_CONTEXT_FILE, CONCEPTS_FILE

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self._morocco_context = self._load_morocco_context()
        self._concepts_data = self._load_concepts()
        logger.info("RAG service initialized")

    def _load_morocco_context(self) -> dict:
        try:
            with open(MOROCCO_CONTEXT_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load Morocco context: {e}")
            return {}

    def _load_concepts(self) -> list:
        try:
            with open(CONCEPTS_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load concepts: {e}")
            return []

    def _build_system_prompt(self) -> str:
        ctx = self._morocco_context
        gov = ctx.get("governance", {})
        econ = ctx.get("economy", {})
        ai = ctx.get("ai_ecosystem", {})

        return (
            "You are an AI policy advisor for Moroccan policymakers. "
            "Explain AI concepts clearly in context of Morocco's situation.\n\n"
            f"Morocco: {econ.get('gdp_per_capita_usd', 3600)} GDP/capita, "
            f"population {ctx.get('demographics', {}).get('population', 37000000) / 1e6:.0f}M, "
            f"data protection: {gov.get('data_protection_law', 'Law 09-08')}, "
            f"digital strategy: {gov.get('digital_strategy', 'Digital Morocco 2030')}, "
            f"{ai.get('ai_startups', 45)} AI startups, {ai.get('ai_researchers', 350)} researchers.\n\n"
            "Rules: Be concise and practical. Reference Morocco's legal framework when relevant. "
            "Use examples from sectors important to Morocco (agriculture, tourism, healthcare, education). "
            "Answer in the same language as the question."
        )

    def answer_question(self, question: str, difficulty: str = None) -> dict:
        start = time.time()

        from app.services.search_service import search_service
        from app.services.llm_service import llm_service

        results = search_service.search(
            collection_name="concepts",
            query=question,
            top_k=3,
        )

        concept_context = "\n---\n".join(r["document"] for r in results) if results else "No relevant concepts found."
        sources = [r["metadata"].get("term", r["id"]) for r in results]

        # Find related concepts
        related = []
        if results:
            top_concept_id = results[0]["id"]
            for c in self._concepts_data:
                if c["id"] == top_concept_id:
                    for rel_id in c.get("related_concepts", [])[:3]:
                        for rc in self._concepts_data:
                            if rc["id"] == rel_id:
                                related.append({
                                    "id": rc["id"],
                                    "term": rc["term"],
                                    "relevance": "Related concept"
                                })
                    break

        system_prompt = self._build_system_prompt()
        user_prompt = (
            f"Question: {question}\n\n"
            f"Relevant context:\n{concept_context}\n\n"
            "Provide a clear, informative answer tailored to Moroccan policymakers."
        )

        answer = llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=1024,
        )

        elapsed = int((time.time() - start) * 1000)

        return {
            "question": question,
            "answer": answer,
            "related_concepts": related,
            "sources": sources,
            "processing_time_ms": elapsed,
        }

    def get_all_concepts(self) -> list:
        return [
            {
                "id": c["id"],
                "term": c["term"],
                "definition": c["definition"],
                "difficulty": c.get("metadata", {}).get("difficulty", "intermediate"),
                "categories": c.get("metadata", {}).get("categories", []),
            }
            for c in self._concepts_data
        ]

    def get_concept_by_id(self, concept_id: str) -> dict | None:
        for c in self._concepts_data:
            if c["id"] == concept_id:
                return c
        return None


# Singleton
rag_service = RAGService()
