import json
import logging
from typing import List, Dict, Optional, Any

from app.config import CASE_STUDIES_FILE

logger = logging.getLogger(__name__)


class CaseStudyService:
    def __init__(self):
        self._case_studies = self._load_case_studies()
        logger.info(f"Case study service initialized: {len(self._case_studies)} studies")

    def _load_case_studies(self) -> list:
        try:
            with open(CASE_STUDIES_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load case studies: {e}")
            return []

    def get_all(self) -> List[Dict]:
        return [
            {
                "id": cs["id"],
                "country": cs["country"],
                "policy_name": cs["policy"]["name"],
                "policy_type": cs["policy"].get("type", ""),
                "enacted_date": cs["policy"].get("enacted_date", ""),
                "data_quality": cs.get("outcomes", {}).get("data_quality", "medium"),
                "tags": cs.get("metadata", {}).get("tags", []),
                "relevance_score": None,
            }
            for cs in self._case_studies
        ]

    def get_by_id(self, study_id: str) -> Optional[Dict]:
        for cs in self._case_studies:
            if cs["id"] == study_id:
                return {
                    "id": cs["id"],
                    "country": cs["country"],
                    "policy": cs["policy"],
                    "outcomes": cs["outcomes"],
                    "metadata": cs["metadata"],
                }
        return None

    def search(self, query: str, country: str = None, policy_type: str = None, top_k: int = 5) -> List[Dict]:
        from app.services.search_service import search_service

        where = None
        if country:
            where = {"country": country}
        elif policy_type:
            where = {"policy_type": policy_type}

        results = search_service.search(
            collection_name="case_studies",
            query=query,
            top_k=top_k,
            where=where,
        )

        summaries = []
        for r in results:
            cs_data = self.get_by_id(r["id"])
            if cs_data:
                summaries.append({
                    "id": cs_data["id"],
                    "country": cs_data["country"],
                    "policy_name": cs_data["policy"]["name"],
                    "policy_type": cs_data["policy"].get("type", ""),
                    "enacted_date": cs_data["policy"].get("enacted_date", ""),
                    "data_quality": cs_data["outcomes"].get("data_quality", "medium"),
                    "tags": cs_data["metadata"].get("tags", []),
                    "relevance_score": r.get("score", 0.0),
                })
        return summaries

    def find_similar(self, description: str, policy_type: str = None, top_k: int = 5) -> List[Dict]:
        """Find case studies similar to a policy description. Used by impact simulator."""
        results = self.search(description, policy_type=policy_type, top_k=top_k)
        enriched = []
        for r in results:
            full = self.get_by_id(r["id"])
            if full:
                full["relevance_score"] = r.get("relevance_score", 0.0)
                enriched.append(full)
        return enriched

    def compare(self, ids: List[str]) -> List[Dict]:
        """Return full details for a list of case study IDs for side-by-side comparison."""
        results = []
        for study_id in ids:
            cs = self.get_by_id(study_id)
            if cs:
                results.append(cs)
        return results


# Singleton
case_study_service = CaseStudyService()
