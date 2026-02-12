import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import logging
from app.config import CHROMADB_PATH

logger = logging.getLogger(__name__)


class VectorDBService:
    def __init__(self):
        logger.info(f"Initializing ChromaDB at: {CHROMADB_PATH}")
        self._client = chromadb.PersistentClient(
            path=CHROMADB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

    def get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def delete_collection(self, name: str):
        try:
            self._client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception:
            pass  # Collection didn't exist

    def add_documents(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        # ChromaDB requires metadata values to be str, int, float, or bool
        clean_metadatas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if isinstance(v, list):
                    clean[k] = ", ".join(str(x) for x in v)
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        # ChromaDB has batch limits — process in chunks of 500
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=clean_metadatas[i:end]
            )
        logger.info(f"Added {len(ids)} documents to collection '{collection.name}'")

    def search(
        self,
        collection,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        }

    def get_collection_count(self, collection) -> int:
        return collection.count()

    def get_by_id(self, collection, doc_id: str) -> Optional[Dict]:
        results = collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else None,
            }
        return None


# Singleton
vectordb_service = VectorDBService()
