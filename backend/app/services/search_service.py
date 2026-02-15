import json
import logging
from typing import List, Dict, Optional, Any

from rank_bm25 import BM25Okapi

from app.config import CONCEPTS_FILE, CASE_STUDIES_FILE, SEARCH_TOP_K

logger = logging.getLogger(__name__)


class SearchService:
    """BM25-based search service. Loads data from JSON files into memory."""

    def __init__(self):
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._bm25_docs: Dict[str, List[Dict]] = {}
        self._initialized = False

    def initialize(self):
        """Load JSON data and build BM25 indices. Called once on startup."""
        if self._initialized:
            return

        self._load_concepts()
        self._load_case_studies()
        self._initialized = True
        logger.info("Search service initialized with BM25 indices")

    def _load_concepts(self):
        try:
            with open(CONCEPTS_FILE) as f:
                concepts = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load concepts: {e}")
            return

        docs = []
        for c in concepts:
            searchable = c.get(
                "searchable_text",
                f"{c['term']} {c['definition']} {c.get('simple_explanation', '')} "
                f"{' '.join(c.get('examples', []))} {c.get('morocco_context', '')}"
            )
            docs.append({
                "id": c["id"],
                "text": searchable,
                "metadata": {
                    "term": c["term"],
                    "difficulty": c.get("metadata", {}).get("difficulty", "intermediate"),
                    "categories": ", ".join(c.get("metadata", {}).get("categories", [])),
                    "source_type": "concept",
                },
            })

        self._build_index("concepts", docs)

    def _load_case_studies(self):
        try:
            with open(CASE_STUDIES_FILE) as f:
                case_studies = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load case studies: {e}")
            return

        docs = []
        for cs in case_studies:
            policy = cs["policy"]
            outcomes = cs.get("outcomes", {})
            searchable = (
                f"{policy['name']} -- {cs['country']}\n"
                f"{policy['description']}\n"
                f"Key provisions: {'; '.join(policy.get('key_provisions', []))}\n"
                f"Insights: {'; '.join(outcomes.get('qualitative_insights', []))}"
            )
            docs.append({
                "id": cs["id"],
                "text": searchable,
                "metadata": {
                    "country": cs["country"],
                    "policy_name": policy["name"],
                    "policy_type": policy.get("type", ""),
                    "enacted_date": policy.get("enacted_date", ""),
                    "data_quality": outcomes.get("data_quality", "medium"),
                    "tags": ", ".join(cs.get("metadata", {}).get("tags", [])),
                    "source_type": "case_study",
                },
            })

        self._build_index("case_studies", docs)

    def _build_index(self, collection_name: str, documents: List[Dict[str, Any]]):
        if not documents:
            return
        tokenized = [doc["text"].lower().split() for doc in documents]
        self._bm25_indices[collection_name] = BM25Okapi(tokenized)
        self._bm25_docs[collection_name] = documents
        logger.info(f"Built BM25 index for '{collection_name}': {len(documents)} docs")

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        BM25 keyword search with optional metadata filtering.
        Returns list of {"id", "document", "metadata", "score"} dicts.
        """
        if not self._initialized:
            self.initialize()

        if top_k is None:
            top_k = SEARCH_TOP_K

        if collection_name not in self._bm25_indices:
            logger.warning(f"No BM25 index for '{collection_name}'")
            return []

        bm25 = self._bm25_indices[collection_name]
        docs = self._bm25_docs[collection_name]
        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            doc = docs[idx]

            if where:
                match = all(doc["metadata"].get(k) == v for k, v in where.items())
                if not match:
                    continue

            results.append({
                "id": doc["id"],
                "document": doc["text"],
                "metadata": doc["metadata"],
                "score": float(scores[idx]),
            })

            if len(results) >= top_k:
                break

        return results

    def get_document_count(self, collection_name: str) -> int:
        return len(self._bm25_docs.get(collection_name, []))


# Singleton
search_service = SearchService()
