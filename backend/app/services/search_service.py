import logging
from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi
from app.config import RRF_K, SEARCH_TOP_K, DENSE_WEIGHT, BM25_WEIGHT

logger = logging.getLogger(__name__)


class SearchService:
    """
    Hybrid search combining dense vector search (ChromaDB) with
    BM25 keyword search, merged via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._bm25_docs: Dict[str, List[Dict]] = {}

    def build_bm25_index(self, collection_name: str, documents: List[Dict[str, Any]]):
        """
        Build a BM25 index for a collection.
        documents: list of {"id": str, "text": str, "metadata": dict}
        """
        tokenized = [doc["text"].lower().split() for doc in documents]
        self._bm25_indices[collection_name] = BM25Okapi(tokenized)
        self._bm25_docs[collection_name] = documents
        logger.info(f"Built BM25 index for '{collection_name}': {len(documents)} docs")

    def bm25_search(self, collection_name: str, query: str, top_k: int = 10) -> List[Dict]:
        """Keyword search using BM25."""
        if collection_name not in self._bm25_indices:
            return []

        bm25 = self._bm25_indices[collection_name]
        docs = self._bm25_docs[collection_name]
        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": docs[idx]["id"],
                    "document": docs[idx]["text"],
                    "metadata": docs[idx]["metadata"],
                    "bm25_score": float(scores[idx])
                })
        return results

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        query_embedding: List[float],
        chromadb_collection,
        top_k: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Hybrid search: dense + BM25 merged with RRF.

        1. Run dense search (ChromaDB) -> 3x candidates
        2. Run BM25 search -> 3x candidates
        3. Merge with Reciprocal Rank Fusion
        4. Return top_k results
        """
        if top_k is None:
            top_k = SEARCH_TOP_K

        candidate_count = top_k * 3

        # Dense search
        from app.services.vectordb_service import vectordb_service
        dense_results = vectordb_service.search(
            chromadb_collection,
            query_embedding,
            top_k=candidate_count,
            where=where
        )

        # BM25 search
        bm25_results = self.bm25_search(collection_name, query, top_k=candidate_count)

        # Build ranked lists
        dense_ranked = {}
        for rank, doc_id in enumerate(dense_results["ids"]):
            dense_ranked[doc_id] = {
                "rank": rank + 1,
                "document": dense_results["documents"][rank],
                "metadata": dense_results["metadatas"][rank],
                "distance": dense_results["distances"][rank]
            }

        bm25_ranked = {}
        for rank, result in enumerate(bm25_results):
            bm25_ranked[result["id"]] = {
                "rank": rank + 1,
                "document": result["document"],
                "metadata": result["metadata"],
                "bm25_score": result["bm25_score"]
            }

        # RRF fusion
        all_ids = set(dense_ranked.keys()) | set(bm25_ranked.keys())
        fused = []

        for doc_id in all_ids:
            rrf_score = 0.0
            if doc_id in dense_ranked:
                rrf_score += DENSE_WEIGHT * (1.0 / (RRF_K + dense_ranked[doc_id]["rank"]))
            if doc_id in bm25_ranked:
                rrf_score += BM25_WEIGHT * (1.0 / (RRF_K + bm25_ranked[doc_id]["rank"]))

            doc_info = dense_ranked.get(doc_id) or bm25_ranked.get(doc_id)
            fused.append({
                "id": doc_id,
                "document": doc_info["document"],
                "metadata": doc_info["metadata"],
                "rrf_score": rrf_score,
                "in_dense": doc_id in dense_ranked,
                "in_bm25": doc_id in bm25_ranked,
            })

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:top_k]


# Singleton
search_service = SearchService()
